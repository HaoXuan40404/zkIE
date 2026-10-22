//! Expert-major gather (scatter-add): sum the weighted per-expert-segment
//! outputs back into token-major order, proved via the fractional histogram
//! identity `sum_pos weighted[pos]/(alpha - token_id[pos]) = sum_t out[t]/(alpha - t)`
//! (the reverse of a lookup).

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64, ChallengeSource};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::common::logup_gkr::{prove_fractional, verify_fractional, FractionalProof};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ScatterAddProof {
    pub frac: FractionalProof,
    pub alpha: Goldilocks,
    pub gamma: Vec<Goldilocks>,
    pub weighted: Vec<Goldilocks>,
    pub out: Vec<Goldilocks>,
    pub token_id: Vec<u32>,
}

/// `out[t][d] = sum_{pos: token_id[pos] == t} weighted[pos][d]`.
pub fn gather_forward(
    weighted: &[Goldilocks],
    token_id: &[u32],
    m: usize,
    h: usize,
) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; m * h];
    for (pos, &t) in token_id.iter().enumerate() {
        for d in 0..h {
            out[t as usize * h + d] = out[t as usize * h + d] + weighted[pos * h + d];
        }
    }
    out
}

pub fn prove_scatter_add(
    weighted: &[Goldilocks],
    token_id: &[u32],
    out: &[Goldilocks],
    m: usize,
    h: usize,
    rng: &mut impl ChallengeSource,
) -> ScatterAddProof {
    let s = token_id.len();
    assert_eq!(weighted.len(), s * h);
    assert_eq!(out.len(), m * h);
    let alpha = rng.field();
    let gamma: Vec<Goldilocks> = (0..h).map(|_| rng.field()).collect();
    let total = (s * h + m * h).next_power_of_two();
    let mut num = vec![Goldilocks::ZERO; total];
    let mut den = vec![Goldilocks::ONE; total];
    for pos in 0..s {
        for d in 0..h {
            num[pos * h + d] = gamma[d] * weighted[pos * h + d];
            den[pos * h + d] = alpha - Goldilocks::from_u64(token_id[pos] as u64);
        }
    }
    for t in 0..m {
        for d in 0..h {
            num[s * h + t * h + d] = from_i64(-1) * gamma[d] * out[t * h + d];
            den[s * h + t * h + d] = alpha - Goldilocks::from_u64(t as u64);
        }
    }
    let frac = prove_fractional(num, den, rng);
    ScatterAddProof {
        frac,
        alpha,
        gamma,
        weighted: weighted.to_vec(),
        out: out.to_vec(),
        token_id: token_id.to_vec(),
    }
}

pub fn verify_scatter_add(
    proof: &ScatterAddProof,
    weighted: &[Goldilocks],
    token_id: &[u32],
    out: &[Goldilocks],
    m: usize,
    h: usize,
) -> bool {
    let s = token_id.len();
    assert_eq!(weighted.len(), s * h);
    assert_eq!(out.len(), m * h);
    let total = (s * h + m * h).next_power_of_two();
    let mut num = vec![Goldilocks::ZERO; total];
    let mut den = vec![Goldilocks::ONE; total];
    for pos in 0..s {
        for d in 0..h {
            num[pos * h + d] = proof.gamma[d] * weighted[pos * h + d];
            den[pos * h + d] = proof.alpha - Goldilocks::from_u64(token_id[pos] as u64);
        }
    }
    for t in 0..m {
        for d in 0..h {
            num[s * h + t * h + d] = from_i64(-1) * proof.gamma[d] * out[t * h + d];
            den[s * h + t * h + d] = proof.alpha - Goldilocks::from_u64(t as u64);
        }
    }
    verify_fractional(&proof.frac, num, den, Goldilocks::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_add_roundtrip() {
        let mut rng = XorShift64::new(0x777);
        let (m, h) = (4usize, 2usize);
        // token_id: 4 positions routed to tokens [0, 2, 1, 2]
        let token_id = vec![0u32, 2, 1, 2];
        let weighted: Vec<Goldilocks> = (0..token_id.len() * h)
            .map(|i| Goldilocks::from_u64((i as u64 % 5) + 1))
            .collect();
        let out = gather_forward(&weighted, &token_id, m, h);
        // sanity: out[2] = weighted[1] + weighted[3]
        let mut expect = vec![Goldilocks::ZERO; m * h];
        for (pos, &t) in token_id.iter().enumerate() {
            for d in 0..h {
                expect[t as usize * h + d] = expect[t as usize * h + d] + weighted[pos * h + d];
            }
        }
        assert_eq!(out, expect);
        let proof = prove_scatter_add(&weighted, &token_id, &out, m, h, &mut rng);
        assert!(verify_scatter_add(&proof, &weighted, &token_id, &out, m, h));
        // tamper the output -> reject
        let mut bad = out.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        assert!(!verify_scatter_add(&proof, &weighted, &token_id, &bad, m, h));
    }
}


#[cfg(test)]
mod moe_minimal {
    use super::*;
    use zkie_core::common::field::XorShift64;

    #[test]
    fn moe_minimal_closure() {
        // synthetic MoE: n=4 experts, k=2 top-k, m=4 tokens, h=2 hidden dims
        let mut rng = XorShift64::new(0x999);
        let (m, n, h) = (4usize, 4usize, 2usize);
        let mut sel = vec![Goldilocks::ZERO; m * n];
        for t in 0..m {
            sel[t * n + (t % n)] = Goldilocks::ONE;
            sel[t * n + ((t + 1) % n)] = Goldilocks::ONE;
        }
        let gate: Vec<Goldilocks> = (0..m * n)
            .map(|i| Goldilocks::from_u64((i as u64 % 3) + 1))
            .collect();
        let x: Vec<Goldilocks> = (0..m * h)
            .map(|i| Goldilocks::from_u64((i as u64 % 7) + 1))
            .collect();

        let (token_id, expert_of) = crate::scatter::scatter_forward(&sel, m, n);
        let mut seg = Vec::with_capacity(token_id.len() * h);
        for &t in &token_id {
            let base = t as usize * h;
            for d in 0..h {
                seg.push(x[base + d]);
            }
        }
        let ffn_out: Vec<Goldilocks> = (0..token_id.len() * h)
            .map(|i| {
                let pos = i / h;
                seg[i] * Goldilocks::from_u64(expert_of[pos] as u64 + 1)
            })
            .collect();
        let weighted: Vec<Goldilocks> = (0..token_id.len() * h)
            .map(|i| {
                let pos = i / h;
                ffn_out[i] * gate[token_id[pos] as usize * n + expert_of[pos] as usize]
            })
            .collect();
        let out = gather_forward(&weighted, &token_id, m, h);

        // dense reference
        let mut reference = vec![Goldilocks::ZERO; m * h];
        for t in 0..m {
            for e in 0..n {
                if sel[t * n + e] == Goldilocks::ONE {
                    for d in 0..h {
                        reference[t * h + d] = reference[t * h + d]
                            + x[t * h + d] * Goldilocks::from_u64(e as u64 + 1) * gate[t * n + e];
                    }
                }
            }
        }
        assert_eq!(out, reference);

        let sp = crate::scatter::prove_scatter(&sel, &token_id, &expert_of, m, n, &mut rng);
        assert!(crate::scatter::verify_scatter(&sp, m, n));
        let gp = prove_scatter_add(&weighted, &token_id, &out, m, h, &mut rng);
        assert!(verify_scatter_add(&gp, &weighted, &token_id, &out, m, h));

        // tamper the gather output -> reject
        let mut bad = out.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        assert!(!verify_scatter_add(&gp, &weighted, &token_id, &bad, m, h));
    }
}
