//! Prove + verify the full DeepSeek-V2-Lite graph at per-layer granularity and
//! report prove/verify time + peak RSS.

use zkie_core::common::field::XorShift64;
#[path = "../../benchmarks/vertical_weights_e2e.rs"]
mod vertical_weights_e2e;
use zkie_models_deepseek_v2_lite::build_deepseek;
use zkie_ops::compose::{prove_shard_dag, verify_shard_dag, Store};

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
    let m = std::env::var("M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16usize);
    let dir = "models/deepseek-v2-lite/weights";
    let mut store = Store::new();
    let mut ops = Vec::new();
    build_deepseek(&mut store, &mut ops, dir, m, 16);
    let total = ops.len();
    println!("seq={m} total ops: {total}");

    let group_size = std::env::var("ZKIE_VERTICAL_GROUP_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4usize);
    let pow_bits = std::env::var("ZKIE_VERTICAL_POW_BITS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(17usize);
    let t_commit = std::time::Instant::now();
    let batches = vertical_weights_e2e::commit_weight_batches(
        &store, &ops, group_size, pow_bits, "deepseek-v2-lite",
    );
    let commit_t = t_commit.elapsed();
    let mut rng = XorShift64::new(0xD5EED);
    let t0 = std::time::Instant::now();
    let proof = prove_shard_dag(&mut store, &ops, 718, &mut rng);
    let openings = vertical_weights_e2e::prove_weight_openings(&batches, &proof);
    let prove_t = t0.elapsed();
    println!(
        "commit {commit_t:?} prove {prove_t:?} shards={} batches={} openings={} rss={}kB",
        proof.shards.len(),
        batches.len(),
        openings.len(),
        peak_rss_kb()
    );

    let t1 = std::time::Instant::now();
    let ok = verify_shard_dag(&mut store, &ops, 718, &proof);
    let open_ok = vertical_weights_e2e::verify_weight_openings(&batches, &openings);
    let verify_t = t1.elapsed();
    println!(
        "verify {verify_t:?} ok={ok} open_ok={open_ok} rss={}kB",
        peak_rss_kb()
    );
}
