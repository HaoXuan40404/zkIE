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
    pub gate: Vec<Goldilocks>,
    pub out: Vec<Goldilocks>,
    pub token_id: Vec<u32>,
}

/// `out[t][d] = sum_{pos: token_id[pos] == t} weighted[pos][d] * gate[pos]`.
pub fn gather_forward(
    weighted: &[Goldilocks],
    token_id: &[u32],
    gate: &[Goldilocks],
    m: usize,
    h: usize,
) -> Vec<Goldilocks> {
    assert_eq!(gate.len(), token_id.len());
    let mut out = vec![Goldilocks::ZERO; m * h];
    for (pos, &t) in token_id.iter().enumerate() {
        for d in 0..h {
            out[t as usize * h + d] = out[t as usize * h + d] + weighted[pos * h + d] * gate[pos];
        }
    }
    out
}

pub fn prove_scatter_add(
    weighted: &[Goldilocks],
    token_id: &[u32],
    gate: &[Goldilocks],
    out: &[Goldilocks],
    m: usize,
    h: usize,
    rng: &mut impl ChallengeSource,
) -> ScatterAddProof {
    let s = token_id.len();
    assert_eq!(weighted.len(), s * h);
    assert_eq!(gate.len(), s);
    assert_eq!(out.len(), m * h);
    let alpha = rng.field();
    let gamma: Vec<Goldilocks> = (0..h).map(|_| rng.field()).collect();
    let total = (s * h + m * h).next_power_of_two();
    let mut num = vec![Goldilocks::ZERO; total];
    let mut den = vec![Goldilocks::ONE; total];
    for pos in 0..s {
        for d in 0..h {
            num[pos * h + d] = gamma[d] * weighted[pos * h + d] * gate[pos];
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
        gate: gate.to_vec(),
        out: out.to_vec(),
        token_id: token_id.to_vec(),
    }
}

pub fn verify_scatter_add(
    proof: &ScatterAddProof,
    weighted: &[Goldilocks],
    token_id: &[u32],
    gate: &[Goldilocks],
    out: &[Goldilocks],
    m: usize,
    h: usize,
) -> bool {
    let s = token_id.len();
    assert_eq!(weighted.len(), s * h);
    assert_eq!(gate.len(), s);
    assert_eq!(out.len(), m * h);
    let total = (s * h + m * h).next_power_of_two();
    let mut num = vec![Goldilocks::ZERO; total];
    let mut den = vec![Goldilocks::ONE; total];
    for pos in 0..s {
        for d in 0..h {
            num[pos * h + d] = proof.gamma[d] * weighted[pos * h + d] * gate[pos];
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
    let ones: Vec<Goldilocks> = vec![Goldilocks::ONE; token_id.len()];
        let weighted: Vec<Goldilocks> = (0..token_id.len() * h)
            .map(|i| Goldilocks::from_u64((i as u64 % 5) + 1))
            .collect();
        let out = gather_forward(&weighted, &token_id, &ones, m, h);
        // sanity: out[2] = weighted[1] + weighted[3]
        let mut expect = vec![Goldilocks::ZERO; m * h];
        for (pos, &t) in token_id.iter().enumerate() {
            for d in 0..h {
                expect[t as usize * h + d] = expect[t as usize * h + d] + weighted[pos * h + d];
            }
        }
        assert_eq!(out, expect);
        let proof = prove_scatter_add(&weighted, &token_id, &ones, &out, m, h, &mut rng);
        assert!(verify_scatter_add(&proof, &weighted, &token_id, &ones, &out, m, h));
        // tamper the output -> reject
        let mut bad = out.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        assert!(!verify_scatter_add(&proof, &weighted, &token_id, &ones, &bad, m, h));
    }
}



    #[test]
    fn scatter_add_weighted_roundtrip() {
        let mut rng = XorShift64::new(0x888);
        let (m, h) = (4usize, 2usize);
        let token_id = vec![0u32, 2, 1, 2];
        let gate: Vec<Goldilocks> = vec![
            Goldilocks::from_u64(2),
            Goldilocks::from_u64(3),
            Goldilocks::from_u64(5),
            Goldilocks::from_u64(7),
        ];
        let weighted: Vec<Goldilocks> = (0..token_id.len() * h)
            .map(|i| Goldilocks::from_u64((i as u64 % 5) + 1))
            .collect();
        let out = gather_forward(&weighted, &token_id, &gate, m, h);
        let mut expect = vec![Goldilocks::ZERO; m * h];
        for (pos, &t) in token_id.iter().enumerate() {
            for d in 0..h {
                expect[t as usize * h + d] =
                    expect[t as usize * h + d] + weighted[pos * h + d] * gate[pos];
            }
        }
        assert_eq!(out, expect);
        let proof = prove_scatter_add(&weighted, &token_id, &gate, &out, m, h, &mut rng);
        assert!(verify_scatter_add(&proof, &weighted, &token_id, &gate, &out, m, h));
        let mut bad = out.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        assert!(!verify_scatter_add(&proof, &weighted, &token_id, &gate, &bad, m, h));
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
    let ones: Vec<Goldilocks> = vec![Goldilocks::ONE; token_id.len()];
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
        let out = gather_forward(&weighted, &token_id, &ones, m, h);

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
        let gp = prove_scatter_add(&weighted, &token_id, &ones, &out, m, h, &mut rng);
        assert!(verify_scatter_add(&gp, &weighted, &token_id, &ones, &out, m, h));

        // tamper the gather output -> reject
        let mut bad = out.clone();
        bad[0] = bad[0] + Goldilocks::ONE;
        assert!(!verify_scatter_add(&gp, &weighted, &token_id, &ones, &bad, m, h));
    }
}
#[cfg(test)]
mod moe_bmm {
    use super::*;
    use crate::scatter::{prove_scatter, scatter_forward, verify_scatter};
    use zkie_core::common::batched_matmul;
    use zkie_core::common::field::{ChallengeSource, Goldilocks, PrimeCharacteristicRing, XorShift64};

    #[test]
    fn expert_major_bmm_closure() {
        let mut rng = XorShift64::new(0x1234);
        let (m, n, h, p) = (4usize, 4usize, 4usize, 2usize);
        let k_top = 2usize;
        let s = m * k_top;

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
        let eg_all: Vec<Goldilocks> = (0..n * h * p)
            .map(|i| Goldilocks::from_u64((i as u64 % 5) + 1))
            .collect();

        let (token_id, expert_of) = scatter_forward(&sel, m, n);
    let ones: Vec<Goldilocks> = vec![Goldilocks::ONE; token_id.len()];
        assert_eq!(token_id.len(), s);

        let mut seg = vec![Goldilocks::ZERO; s * h];
        for pos in 0..s {
            let t = token_id[pos] as usize;
            for d in 0..h {
                seg[pos * h + d] = x[t * h + d];
            }
        }

        let eg = batched_matmul::gather_weights(&expert_of, &eg_all, s, n, h, p);
        let g = batched_matmul::forward(&seg, &eg, s, h, p);

        let weighted: Vec<Goldilocks> = (0..s * p)
            .map(|i| {
                let pos = i / p;
                g[i] * gate[token_id[pos] as usize * n + expert_of[pos] as usize]
            })
            .collect();
        let out = gather_forward(&weighted, &token_id, &ones, m, p);

        let mut reference = vec![Goldilocks::ZERO; m * p];
        for t in 0..m {
            for e in 0..n {
                if sel[t * n + e] == Goldilocks::ONE {
                    for j in 0..p {
                        let mut acc = Goldilocks::ZERO;
                        for d in 0..h {
                            acc = acc + x[t * h + d] * eg_all[e * h * p + d * p + j];
                        }
                        reference[t * p + j] = reference[t * p + j] + acc * gate[t * n + e];
                    }
                }
            }
        }
        assert_eq!(out, reference);

        let sp = prove_scatter(&sel, &token_id, &expert_of, m, n, &mut rng);
        assert!(verify_scatter(&sp, m, n));

        let u: Vec<Goldilocks> = (0..s.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let v: Vec<Goldilocks> = (0..p.trailing_zeros() as usize).map(|_| rng.field()).collect();
        let ch: Vec<Goldilocks> = (0..(s * h).trailing_zeros() as usize)
            .map(|_| rng.field())
            .collect();
        let bp = batched_matmul::prove(&seg, &eg, &g, s, h, p, &u, &v, &ch);
        assert!(batched_matmul::verify(&bp, s, h, &u, &ch));

        let gp = prove_scatter_add(&weighted, &token_id, &ones, &out, m, p, &mut rng);
        assert!(verify_scatter_add(&gp, &weighted, &token_id, &ones, &out, m, p));

        let mut bad_tid = token_id.clone();
        bad_tid[0] = (bad_tid[0] + 1) % m as u32;
        let bad_sp = prove_scatter(&sel, &bad_tid, &expert_of, m, n, &mut rng);
        assert!(!verify_scatter(&bad_sp, m, n));
    }
}

