# Committed-path issues (store-free verification follow-up)

Branch `feature/verify-no-forward`. The committed shard-dag path
(`prove_committed_shard_dag` / `verify_committed_shard_dag`) is root-only,
serialized (P3/P4), and the weight commitment is now split into a one-time
`commit_model_weights` (commit + cache) phase plus a prove phase that only
restores prover data from cache (no re-commit).

## Resolved

1. **`OpeningProtocol` / `Whir` not serde** — `Whir` is serialized as
   `(num_variables, security_level, pow_bits)` and reconstructed; the opening
   protocol is rebuilt from the tensor arity.
2. **mmap weight `Store::get` panic** — `commit_model_weights` uses
   `store.materialize(t).into_owned()`.
3. **Weight commitment used `new_testing` (32-bit)** — now `Whir::with_params(
   arity, 90, pow_bits=17)` plus horizontal merge, borrowed from
   `feature/batch-pcs` (`docs/vertical-batching-2026-10-21.md`).

## Borrowed from `feature/batch-pcs` (already absorbed)

- 90-bit security + `pow_bits=17`.
- Horizontal merge: pack same-arity weights into groups sized
  `1 << (26 - arity)` so `num_vars <= 26` stays within the PoW budget.
- Lazy opening (restore one group's Merkle tree at a time) and bulk POD
  serialization of prover data (both already in the cache path).

## Cannot change without a decision (recorded, not yet acted on)

1. **Full vs partial weight commitment.** `feature/batch-pcs` commits only
   `Op::Projection { w }` (q/k/v/o_proj/fc/proj), skipping `Op::MatMul { b }`
   (`lm_head`, 2^26) and biases — that is why its commit is ~7s while a full
   commit of the `lm_head` is minutes.
   **Decision (2026-10-22): commit `lm_head` too — full weight commitment.**
   The one-time commit stays expensive; that is accepted as the baseline.
2. **Boundary `Whir` num_variables must match exactly.** The committed path
   commits every boundary tensor with a single `whir`; p3-whir's adapter asserts
   arity == configured num_variables, so it only works when all boundary tensors
   share one size. Unrelated to weight commitment; not addressed by
   `feature/batch-pcs`.
3. **Affine bias committed but never opened.** `collect_weight_tensors` still
   collects the projection bias, but the Projection op only claims x/w/out (not
   bias), so a bias group has no opening (currently skipped as `proof: None`).
   If we keep committing biases this gap stays; `feature/batch-pcs` sidesteps it
   by not committing biases at all.

## Status

Store-free verification (no forward re-run) is done and measured on the plain
path. The committed (root-only) path is implemented, unit-tested (tamper /
wrong-root / serialization roundtrip, commit -> cache -> restore -> open), with
90-bit weight commitments + horizontal merge. Real-model end-to-end timing for
the committed path is still to be measured; first-run commit is dominated by the
`lm_head` if it is committed (see "Cannot change" #1).

## Baseline (2026-10-22)

GPT-2 seq=512 committed path, 90-bit + pow_bits=17 + horizontal merge, full
weight commitment (including lm_head). Release, 64-thread CPU.

| phase | time |
| --- | --- |
| commit (first run, writes cache) | 16.66 s |
| prove (restore + batch open) | 51.87 s |
| verify | 68.11 s |

argmax 511/512 (one near-tie fixed-point), peak RSS ~30.3 GB, proof bytes ~1.73 GB.

Note: an earlier argmax=0/512 was a bench bug (the argmax loop used vocab 65536
instead of the real 50257); the committed path itself was correct.
