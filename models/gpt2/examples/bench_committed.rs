//! GPT-2 seq16 end-to-end through the committed (root-only) shard-DAG path:
//! weights and shard boundaries are WHIR-committed and opened, and the verifier
//! holds only commitment roots + opening proofs (no weight/activation data).

use std::fs;

use zkie_core::common::field::{Goldilocks, XorShift64};
use zkie_core::common::fixed_point::{from_i32, to_i32, to_i64};
use zkie_core::pcs::whir::Whir;
use zkie_models_gpt2::build_gpt2;
use zkie_ops::compose::{commit_model_weights, prove_committed_shard_dag, verify_committed_shard_dag_bytes, Store};

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn peak_rss_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmHWM:") {
            return v.trim().trim_end_matches(" kB").parse().unwrap_or(0);
        }
    }
    0
}

fn main() {
    let m = 16usize;
    let shift = 16u32;
    let dir = "models/gpt2/weights";

    let mut store = Store::new();
    let mut ops = Vec::new();
    let (_x0, logits) = build_gpt2(&mut store, &mut ops, dir, m, shift);
    let gt = load_i32(&format!("{dir}/gt_argmax_i32.bin"));

    let whir = Whir::new_testing(14);
    let mut rng = XorShift64::new(0xBEEF);

    let t0 = std::time::Instant::now();
    let cached = commit_model_weights(&store, &ops, "/tmp/zkie-weight-cache");
    let proof = prove_committed_shard_dag(&mut store, &ops, 176, &whir, &cached, &mut rng);
    let prove_t = t0.elapsed();

    let t1 = std::time::Instant::now();
    let bytes = bincode::serialize(&proof).unwrap();
    assert!(
        verify_committed_shard_dag_bytes(&bytes, &store.owned_slices(), &ops, 176, &whir),
        "committed proof failed"
    );
    let verify_t = t1.elapsed();

    let lg = store.get(logits);
    let mut matches = 0;
    for i in 0..m {
        let mut best = 0usize;
        let mut best_v = lg[i * 65536];
        for j in 1..65536 {
            let v = lg[i * 65536 + j];
            if to_i64(v) > to_i64(best_v) {
                best_v = v;
                best = j;
            }
        }
        if best as i32 == to_i32(gt[i]) {
            matches += 1;
        }
    }

    println!(
        "prove={prove_t:?} verify={verify_t:?} argmax={matches}/{m} rss={}kB proof_bytes={}",
        peak_rss_kb(),
        bytes.len()
    );
}
