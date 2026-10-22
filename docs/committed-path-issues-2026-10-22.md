# Committed-path issues (store-free verification follow-up)

Branch `feature/verify-no-forward`. The committed shard-dag path
(`prove_committed_shard_dag` / `verify_committed_shard_dag`) is now root-only and
serialized (P3/P4), and the weight-commit primitive (`commit_weights_batch`) that
already existed on the old `feature/batch-pcs` was finally wired into the shard-dag
flow. Wiring it to a real model exposed the following. These are follow-ups, not
part of the core store-free deliverable.

## 1. Full-weight WHIR commitment is the bottleneck (prove side)

`prove_committed_shard_dag` now commits every weight tensor (grouped by size). On
GPT-2 seq16 this did not finish in ~9 min; the dominant cost is `lm_head`
(2^26 elements): RS encoding + FFT + Merkle tree. This is the same WHIR-overhead
problem as before, now exercised on real weights.

## 2. p3 `OpeningProtocol` / `Whir` are not serde

`OpeningProtocol` (p3-sumcheck) has no `Serialize`/`Deserialize` and does not
expose its point schedule; `Whir` contains `Cell`. Worked around by storing
`(num_variables, security_level, pow_bits)` for `Whir` and reconstructing the
opening protocol from the tensor arity via `Whir::opening_protocol`.

## 3. mmap weight tensors panic on `Store::get` (prove side)

`collect_weight_tensors` read weights with `store.get(t)`, which panics on
mmap-backed tensors (`lm_head`). Fixed with `store.materialize(t).into_owned()`.

## 4. Boundary `Whir` num_variables must match exactly (prove side)

`prove_committed_shard_dag` commits every boundary tensor with the single passed
`whir`; p3-whir's adapter asserts the witness arity equals the configured
`num_variables` (e.g. arity 14 vs configured 16 panics). The committed path
therefore only works when all boundary tensors share one size.

## 5. Weight commitment uses `new_testing` (low security)

Weight batches are committed with `Whir::new_testing(len.trailing_zeros())`. Real
security parameters hit `PowBitsExceedBudget` for large tensors (2^26), the
pre-existing pow-budget issue.

## Status

Store-free verification (no forward re-run) is done and measured on the plain
path. The committed (root-only) path is implemented and unit-tested (tamper /
wrong-root / serialization roundtrip), but its real-model prove cost is
WHIR-bound and not yet measured end-to-end.
