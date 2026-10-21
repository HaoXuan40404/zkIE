//! logUp GKR: prove sum_i num[i]/den[i] == C via the fraction-addition tree
//! with eq-weighted virtual sumchecks (DeepProve "fractional sumcheck").
use crate::common::field::{Field, Goldilocks, PrimeCharacteristicRing, XorShift64};
use crate::common::fixed_point::from_i64;
use crate::common::sumcheck::{prove_virtual, verify_virtual, VirtualProof};
pub struct FractionalLayer {
    pub r: Vec<Goldilocks>,
    pub c: Goldilocks,
    pub proof: VirtualProof,
}
pub struct FractionalProof {
    pub layers: Vec<FractionalLayer>,
    pub final_num: Goldilocks,
    pub final_den: Goldilocks,
}
fn build_terms(c: Goldilocks) -> Vec<(Goldilocks, Vec<usize>)> {
    let neg = from_i64(-1);
    vec![
        (Goldilocks::from_u64(1), vec![0usize, 1]),
        (neg, vec![0, 3, 6]),
        (neg, vec![0, 4, 5]),
        (c, vec![0, 2]),
        (neg * c, vec![0, 5, 6]),
    ]
}
pub fn prove_fractional(
    mut num: Vec<Goldilocks>,
    mut den: Vec<Goldilocks>,
    rng: &mut XorShift64,
) -> FractionalProof {
    assert_eq!(num.len(), den.len());
    assert!(num.len().is_power_of_two());
    let mut layers = Vec::new();
    while num.len() > 1 {
        let half = num.len() / 2;
        let (num_low, num_high) = num.split_at(half);
        let (den_low, den_high) = den.split_at(half);
        let next_num: Vec<Goldilocks> = (0..half)
            .map(|s| num_low[s] * den_high[s] + num_high[s] * den_low[s])
            .collect();
        let next_den: Vec<Goldilocks> = (0..half).map(|s| den_low[s] * den_high[s]).collect();
        let m = half.trailing_zeros() as usize;
        let r: Vec<Goldilocks> = (0..m).map(|_| rng.field()).collect();
        let c = rng.field();
        let eq: Vec<Goldilocks> = crate::common::mle::eq_evals(&r);
        let mles: Vec<&[Goldilocks]> = vec![
            &eq, &next_num, &next_den, num_low, num_high, den_low, den_high,
        ];
        let terms = build_terms(c);
        let proof = prove_virtual(&mles, &terms, Goldilocks::from_u64(0), &r);
        layers.push(FractionalLayer { r, c, proof });
        num = next_num;
        den = next_den;
    }
    FractionalProof {
        layers,
        final_num: num[0],
        final_den: den[0],
    }
}
pub fn verify_fractional(
    proof: &FractionalProof,
    mut num: Vec<Goldilocks>,
    mut den: Vec<Goldilocks>,
    claimed: Goldilocks,
) -> bool {
    for layer in &proof.layers {
        let half = num.len() / 2;
        let num_low = &num[..half];
        let num_high = &num[half..];
        let den_low = &den[..half];
        let den_high = &den[half..];
        let next_num: Vec<Goldilocks> = (0..half)
            .map(|s| num_low[s] * den_high[s] + num_high[s] * den_low[s])
            .collect();
        let next_den: Vec<Goldilocks> = (0..half).map(|s| den_low[s] * den_high[s]).collect();
        let eq: Vec<Goldilocks> = crate::common::mle::eq_evals(&layer.r);
        let mles: Vec<&[Goldilocks]> = vec![
            &eq, &next_num, &next_den, num_low, num_high, den_low, den_high,
        ];
        let terms = build_terms(layer.c);
        let final_evals = vec![
            crate::common::mle::eval(&eq, &layer.r),
            crate::common::mle::eval(&next_num, &layer.r),
            crate::common::mle::eval(&next_den, &layer.r),
            crate::common::mle::eval(num_low, &layer.r),
            crate::common::mle::eval(num_high, &layer.r),
            crate::common::mle::eval(den_low, &layer.r),
            crate::common::mle::eval(den_high, &layer.r),
        ];
        if !verify_virtual(
            &layer.proof,
            &terms,
            Goldilocks::from_u64(0),
            &layer.r,
            &final_evals,
        ) {
            return false;
        }
        num = next_num;
        den = next_den;
    }
    proof.final_num == claimed * proof.final_den
}

/// Prove `y_i == table[x_i]` for every `i` via the fractional-sumcheck logUp:
/// `sum_i 1/(alpha+x_i+beta*y_i) = sum_j m_j/(alpha+j+beta*table[j])`, encoded as
/// a combined fraction list whose sum is 0 and proven with `prove_fractional`.
/// This replaces the standalone grand-product form (`prove_product`).
pub fn prove_lookup_fractional(
    x: &[u32],
    y: &[Goldilocks],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    rng: &mut XorShift64,
) -> FractionalProof {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let t = table.len();
    let total = (n + t).next_power_of_two();
    let mut num = vec![Goldilocks::from_u64(0); total];
    let mut den = vec![Goldilocks::from_u64(1); total];
    for i in 0..n {
        num[i] = Goldilocks::from_u64(1);
        den[i] = alpha + Goldilocks::from_u64(x[i] as u64) + beta * y[i];
    }
    let mut m = vec![0u64; t];
    for &i in x {
        m[i as usize] += 1;
    }
    for j in 0..t {
        num[n + j] = from_i64(-(m[j] as i64));
        den[n + j] = alpha + Goldilocks::from_u64(j as u64) + beta * table[j];
    }
    prove_fractional(num, den, rng)
}
pub fn verify_lookup_fractional(
    proof: &FractionalProof,
    x: &[u32],
    y: &[Goldilocks],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
) -> bool {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let t = table.len();
    let total = (n + t).next_power_of_two();
    let mut num = vec![Goldilocks::from_u64(0); total];
    let mut den = vec![Goldilocks::from_u64(1); total];
    for i in 0..n {
        num[i] = Goldilocks::from_u64(1);
        den[i] = alpha + Goldilocks::from_u64(x[i] as u64) + beta * y[i];
    }
    let mut m = vec![0u64; t];
    for &i in x {
        m[i as usize] += 1;
    }
    for j in 0..t {
        num[n + j] = from_i64(-(m[j] as i64));
        den[n + j] = alpha + Goldilocks::from_u64(j as u64) + beta * table[j];
    }
    verify_fractional(proof, num, den, Goldilocks::from_u64(0))
}

/// Batch several lookups into the SAME `table` into one fractional-sumcheck
/// proof. All sample lists share one random challenge `(alpha, beta)`, so the
/// sumcheck count drops from `xs.len()` to one and the table multiplicity is
/// built only once.
pub fn prove_lookup_fractional_batch(
    xs: &[&[u32]],
    ys: &[&[Goldilocks]],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
    rng: &mut XorShift64,
) -> FractionalProof {
    assert_eq!(xs.len(), ys.len());
    let n: usize = xs.iter().map(|x| x.len()).sum();
    let t = table.len();
    let total = (n + t).next_power_of_two();
    let mut num = vec![Goldilocks::from_u64(0); total];
    let mut den = vec![Goldilocks::from_u64(1); total];
    let mut pos = 0usize;
    for (x, y) in xs.iter().zip(ys.iter()) {
        assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            num[pos] = Goldilocks::from_u64(1);
            den[pos] = alpha + Goldilocks::from_u64(x[i] as u64) + beta * y[i];
            pos += 1;
        }
    }
    let mut m = vec![0u64; t];
    for x in xs {
        for &i in *x {
            m[i as usize] += 1;
        }
    }
    for j in 0..t {
        num[n + j] = from_i64(-(m[j] as i64));
        den[n + j] = alpha + Goldilocks::from_u64(j as u64) + beta * table[j];
    }
    prove_fractional(num, den, rng)
}

pub fn verify_lookup_fractional_batch(
    proof: &FractionalProof,
    xs: &[&[u32]],
    ys: &[&[Goldilocks]],
    table: &[Goldilocks],
    alpha: Goldilocks,
    beta: Goldilocks,
) -> bool {
    assert_eq!(xs.len(), ys.len());
    let n: usize = xs.iter().map(|x| x.len()).sum();
    let t = table.len();
    let total = (n + t).next_power_of_two();
    let mut num = vec![Goldilocks::from_u64(0); total];
    let mut den = vec![Goldilocks::from_u64(1); total];
    let mut pos = 0usize;
    for (x, y) in xs.iter().zip(ys.iter()) {
        assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            num[pos] = Goldilocks::from_u64(1);
            den[pos] = alpha + Goldilocks::from_u64(x[i] as u64) + beta * y[i];
            pos += 1;
        }
    }
    let mut m = vec![0u64; t];
    for x in xs {
        for &i in *x {
            m[i as usize] += 1;
        }
    }
    for j in 0..t {
        num[n + j] = from_i64(-(m[j] as i64));
        den[n + j] = alpha + Goldilocks::from_u64(j as u64) + beta * table[j];
    }
    verify_fractional(proof, num, den, Goldilocks::from_u64(0))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fractional_roundtrip() {
        let mut rng = XorShift64::new(0xDEC0);
        let n = 1usize << 6;
        let num: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
        let den: Vec<Goldilocks> = (0..n).map(|i| from_i64((i % 7) as i64 + 1)).collect();
        let claimed: Goldilocks = (0..n).fold(Goldilocks::from_u64(0), |acc, i| {
            acc + num[i] * den[i].inverse()
        });
        let proof = prove_fractional(num.clone(), den.clone(), &mut rng);
        assert!(verify_fractional(&proof, num.clone(), den.clone(), claimed));
        let tampered = FractionalProof {
            final_num: proof.final_num + Goldilocks::from_u64(1),
            final_den: proof.final_den,
            layers: proof.layers,
        };
        assert!(!verify_fractional(
            &tampered,
            num.clone(),
            den.clone(),
            claimed
        ));
    }
    #[test]
    fn lookup_fractional_roundtrip() {
        let mut rng = XorShift64::new(0x10AD);
        let n = 64usize;
        let t = 16usize;
        let table: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let x: Vec<u32> = (0..n).map(|_| (rng.next_u64() % t as u64) as u32).collect();
        let y: Vec<Goldilocks> = x.iter().map(|&i| table[i as usize]).collect();
        let alpha = rng.field();
        let beta = rng.field();
        let proof = prove_lookup_fractional(&x, &y, &table, alpha, beta, &mut rng);
        assert!(verify_lookup_fractional(
            &proof, &x, &y, &table, alpha, beta
        ));
        let mut bad_y = y.clone();
        bad_y[0] = bad_y[0] + Goldilocks::from_u64(1);
        let proof_bad = prove_lookup_fractional(&x, &bad_y, &table, alpha, beta, &mut rng);
        assert!(!verify_lookup_fractional(
            &proof_bad, &x, &bad_y, &table, alpha, beta
        ));
    }
    #[test]
    fn committed_lookup_roundtrip() {
        use crate::pcs::committed::commit;
        use crate::pcs::whir::Whir;
        let mut rng = XorShift64::new(0xEE33);
        let n = 32usize;
        let t = 16usize;
        let table: Vec<Goldilocks> = (0..t).map(|_| rng.field()).collect();
        let x: Vec<u32> = (0..n).map(|_| (rng.next_u64() % t as u64) as u32).collect();
        let y: Vec<Goldilocks> = x.iter().map(|&i| table[i as usize]).collect();
        let alpha = rng.field();
        let beta = rng.field();
        let proof = prove_lookup_fractional(&x, &y, &table, alpha, beta, &mut rng);
        assert!(verify_lookup_fractional(
            &proof, &x, &y, &table, alpha, beta
        ));
        let whir = Whir::new_testing(n.trailing_zeros() as usize);
        let c_y = commit(&whir, &y);
        let r: Vec<Goldilocks> = (0..n.trailing_zeros() as usize)
            .map(|_| rng.field())
            .collect();
        let (open, ev) = whir.open(c_y.prover_data.clone(), &c_y.protocol, &r);
        assert_eq!(
            whir.verify(&c_y.commitment, &open, &c_y.protocol, &r)
                .unwrap(),
            ev
        );
        assert_eq!(ev, crate::common::mle::eval(&y, &r));
    }
}
