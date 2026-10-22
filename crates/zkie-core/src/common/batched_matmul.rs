//! Per-position (batched) matmul: `g[pos][j] = sum_d seg[pos][d] * w[pos][d][j]`,
//! where each position `pos` carries its own `K x N` weight matrix (already
//! gathered from the per-expert weights). This is the expert-major counterpart
//! of the ordinary matmul: the sumcheck is the same two-input product sumcheck,
//! but the weight slice is a data-dependent gather rather than a `partial_eval`
//! of a matrix shared by every row.

use crate::common::field::{Goldilocks, PrimeCharacteristicRing};
use crate::common::sumcheck::{self, SumcheckProof};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct BatchedMatmulProof {
    pub claimed: Goldilocks,
    pub sumcheck: SumcheckProof,
}

/// Prove `g[pos][j] = sum_{d in [0,K)} seg[pos][d] * w[pos][d][j]`.
///
/// Layouts (row-major):
/// - `seg`: `S x K`
/// - `w`:   `S x K x N` (per-position weight)
/// - `g`:   `S x N`
///
/// `pos` and `j` fix the output coordinate; `ch_d` carries
/// `K.trailing_zeros()` contraction challenges. `K` must be a power of two.
pub fn prove(
    seg: &[Goldilocks],
    w: &[Goldilocks],
    g: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
    pos: usize,
    j: usize,
    ch_d: &[Goldilocks],
) -> BatchedMatmulProof {
    assert_eq!(seg.len(), s * k);
    assert_eq!(w.len(), s * k * n);
    assert_eq!(g.len(), s * n);
    assert!(pos < s);
    assert!(j < n);
    assert_eq!(ch_d.len(), k.trailing_zeros() as usize);

    let a = &seg[pos * k..(pos + 1) * k];
    let b: Vec<Goldilocks> = (0..k).map(|d| w[pos * k * n + d * n + j]).collect();
    let claimed = g[pos * n + j];
    let sumcheck = sumcheck::prove(a, &b, claimed, ch_d);
    BatchedMatmulProof { claimed, sumcheck }
}

pub fn verify(proof: &BatchedMatmulProof, ch_d: &[Goldilocks]) -> bool {
    sumcheck::verify(
        &proof.sumcheck,
        proof.claimed,
        ch_d,
        proof.sumcheck.f_eval,
        proof.sumcheck.h_eval,
    )
}

/// Plain per-position matmul forward (also the test oracle).
pub fn forward(
    seg: &[Goldilocks],
    w: &[Goldilocks],
    s: usize,
    k: usize,
    n: usize,
) -> Vec<Goldilocks> {
    let mut g = vec![Goldilocks::ZERO; s * n];
    for pos in 0..s {
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for d in 0..k {
                acc = acc + seg[pos * k + d] * w[pos * k * n + d * n + j];
            }
            g[pos * n + j] = acc;
        }
    }
    g
}

/// `w_gathered[pos][d][j] = w_all[expert_of[pos]][d][j]`, with `w_all` laid out
/// expert-major as `n_experts x K x N`. Plain forward only: the *proof* of this
/// gather is the existing `Lookup` op (`out = table[idx]`), so it is not
/// duplicated here.
pub fn gather_weights(
    expert_of: &[u32],
    w_all: &[Goldilocks],
    s: usize,
    n_experts: usize,
    k: usize,
    n: usize,
) -> Vec<Goldilocks> {
    assert_eq!(expert_of.len(), s);
    assert_eq!(w_all.len(), n_experts * k * n);
    let mut w = vec![Goldilocks::ZERO; s * k * n];
    for pos in 0..s {
        let base = expert_of[pos] as usize * k * n;
        for d in 0..k {
            for j in 0..n {
                w[pos * k * n + d * n + j] = w_all[base + d * n + j];
            }
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::{ChallengeSource, XorShift64};

    #[test]
    fn batched_matmul_roundtrip() {
        let mut rng = XorShift64::new(0xbeef);
        let (s, k, n, n_experts) = (4usize, 4usize, 2usize, 3usize);

        let mut w_all = vec![Goldilocks::ZERO; n_experts * k * n];
        for i in 0..w_all.len() {
            w_all[i] = Goldilocks::from_u64((i as u64 % 7) + 1);
        }
        let expert_of: Vec<u32> = (0..s).map(|i| (i % n_experts) as u32).collect();
        let seg: Vec<Goldilocks> = (0..s * k)
            .map(|i| Goldilocks::from_u64((i as u64 % 11) + 1))
            .collect();

        let w = gather_weights(&expert_of, &w_all, s, n_experts, k, n);
        let g = forward(&seg, &w, s, k, n);

        // dense reference oracle
        let mut reference = vec![Goldilocks::ZERO; s * n];
        for pos in 0..s {
            let e = expert_of[pos] as usize;
            for j in 0..n {
                let mut acc = Goldilocks::ZERO;
                for d in 0..k {
                    acc = acc + seg[pos * k + d] * w_all[e * k * n + d * n + j];
                }
                reference[pos * n + j] = acc;
            }
        }
        assert_eq!(g, reference);

        let ch_d: Vec<Goldilocks> = (0..k.trailing_zeros() as usize)
            .map(|_| rng.field())
            .collect();
        for pos in 0..s {
            for j in 0..n {
                let proof = prove(&seg, &w, &g, s, k, n, pos, j, &ch_d);
                assert!(verify(&proof, &ch_d));
            }
        }

        // tamper the claimed output value -> reject
        let mut bad = prove(&seg, &w, &g, s, k, n, 0, 0, &ch_d);
        bad.claimed = bad.claimed + Goldilocks::ONE;
        assert!(!verify(&bad, &ch_d));
    }
}

