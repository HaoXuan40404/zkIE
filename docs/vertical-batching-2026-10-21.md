# Vertical weight batching — end-to-end (2026-10-21)

## What changed

The vertical-batching primitives now drive the model prove/verify path instead
of a sidecar micro-benchmark.

- `models/benchmarks/vertical_weights_e2e.rs` groups same-shape Projection
  weights into one MLE, commits each group once (cacheable), then opens all of a
  model's weight claims in a single batch per group. It is split into three
  phases so the harness times model commitment, prove, and verify separately.
- Each model bench now runs this three-phase end-to-end path and reports
  `commit` / `prove` / `verify` separately. The old baseline-vs-packed sidecar
  measurement is removed.

## Results

Security level 90, PoW bits 17, group size 4.

| Model | Config | Commit | Prove | Verify | Argmax | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-2 | seq=16, 13 shards | 10.22s (cached) | 33.80s | 9.30s | 16/16 | - |
| GPT-2 | seq=16, 1 shard | 10.22s (cached) | 107.63s | 83.00s | 16/16 | - |
| GPT-2 | seq=512, 13 shards | 10.21s (cached) | 47.38s | 20.15s | 511/512 | - |
| GPT-2 | seq=512, 1 shard | 10.21s (cached) | 164.01s | 144.45s | 511/512 | - |
| TimesFM 200M | seq=16, pps=108 | 94.66s (first) | 72.59s | 11.81s | - | 48.4GB |
| Gemma 3 | seq=16, pps=53 | 28.95s (first) | 46.58s | 17.02s | 16/16 | 30.2GB |
| DeepSeek-V2-Lite | seq=16 | - | - | - | - | OOM (>415GB) |

`Commit` is the model-commitment phase: pack + WHIR commit + cache write on the
first run, or cache load on later runs. GPT-2's 10.2s is a cached run;
TimesFM/Gemma 3 numbers are first runs that write the cache.

Correctness holds: GPT-2 seq=512 argmax 511/512 (one near-tie fixed-point,
matching prior records), Gemma 3 16/16. The batch-opening proofs are now part
of the prove/verify path (`verify_weight_openings` is asserted), not a sidecar.

## Key finding: DeepSeek-V2-Lite OOM

DeepSeek-V2-Lite's 64-routed-expert MoE produces ~992 same-shape projection
weight groups (173GB of cached WHIR prover data). Keeping all of them resident
for batch opening pushed peak RSS past 415GB, which the 495GB machine could not
sustain, so the process was OOM-killed.

This confirms that full-resident weight commitments do not scale to large MoE
models. The prover should commit all expert weights once but only load/open the
activated experts (lazy opening), i.e. the "full commitment + lazy opening"
design. Memory should scale with activated parameters, not total parameters.

## Notes

- Forward recomputation in `verify_shard` is unchanged; store-free verification
  is a separate concern and out of scope for this change.
- Batch-opening is now the verification relation for weight claims, so its cost
  is the online cost the batching optimization targets.

## How to run

- GPT-2: `cargo run --release -p zkie-models-gpt2 --example bench_sharded`
  (`ZKIE_SEQ_LEN=16` for seq=16).
- TimesFM: `cargo run --release -p zkie-models-timesfm --example bench`.
- Gemma 3: `cargo run --release -p zkie-models-gemma3 --example bench`
  (`GEMMA_SEQ` overrides seq).
- DeepSeek: `cargo run --release -p zkie-models-deepseek-v2-lite --example bench`
  (`M` overrides seq).
- Cache dir: `/data/jimmyshi/zkie-cache/{model}-model-batches-v2`.
