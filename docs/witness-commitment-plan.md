# Witness commitment plan

## Goal
Replace every plaintext witness tensor embedded in op proofs (sel/d1/d2/e/indices/rstd_vals/rem_lo/hi/token_id/...) with a commit + batch opening, so a proof carries only sumcheck evals, claims and openings, and the verifier never re-runs forward or re-derives the LogUp fraction tree. Intermediate activations stay virtual (uncommitted).

## Baseline (GPT-2 seq16 committed path, measured)
Total proof ~100.9 MB: Argmax 50 MB, Projection 600x35 MB, Lookup 9.8 MB, Softmax 558 KB, LayerNormCentered 403 KB.

## Root cause
verify_fractional / verify_lookup_fractional (logup_gkr.rs) rebuild the full num/den fraction tree from full x/y witness vectors and mle::eval them at every layer. prove_virtual/verify_virtual are already store-free (they carry final_evals), so the fix is confined to the fractional-sumcheck core.

## Design decisions
1. Store-free verifier: verify_fractional uses layer.proof.final_evals instead of rebuilding the tree.
2. Leaf authentication: commit the per-layer num/den columns and open them at the challenge points. Flat approach commits every intermediate fraction-tree layer; simple and correct. A chained GKR committing only original x/y/table/m columns is the follow-up optimisation.
3. Lookup tables are committed too (DeepProve parity), not trusted public input.
4. Security 90 for all commitments (weights, tables, witness, multiplicity).
5. Witness columns are batch-committed per shard, grouped by size, reusing the existing weight commit_batch / prove_batch_open_committed machinery.

## Transcript order (Fiat-Shamir)
1. model id + shard/op structure (types + shapes, deterministic encoding)
2. commitment roots (weights, tables, witness, boundary)
3. in proof order: sumcheck round messages, claims, openings

## Witness column inventory per op
- TopKSelect / Argmax: sel, d1, d2
- RangeDecomp: c0, c1, c2
- Projection: rem_off
- Lookup: x (indices), y (values)
- SoftmaxRounded: indices, e
- LayernormChain: rsqrt_idx, rsqrt_vals
- LayerNormCentered: s_index, rstd_vals
- RmsNorm: s_index, rstd_vals, rem_lo, rem_hi
- RoPE: rem_f, rem_s
- Scatter: token_id, expert_of, sel
- ScatterAdd: weighted, gate, out, token_id
- GatherMatMul: counts, table

## Phases
- P0: this document.
- P1: store-free fractional-sumcheck core (logup_gkr.rs) + leaf claims.
- P2: table + multiplicity commitment (compose.rs committed path).
- P3: per-op witness commitment + drop plaintext vectors (ordered by size).
- P4: end-to-end regression + proof-size metric in README/benchmarks.

## Acceptance
- Each op witness bytes drop to ~K level (bench_committed breakdown).
- Tamper claim / opening / wrong table root / wrong witness root => reject.
- GPT-2 seq16 (16/16) and seq512 (511/512) argmax unchanged; prove/verify not slower than current committed baseline.
