# Batched-matmul (per-position expert weight): correctness note

Status: the concrete-index primitive in `crates/zkie-core/src/common/batched_matmul.rs`
proves ONE coordinate `g[pos][j] = sum_d seg[pos][d] * w[pos][d][j]` correctly; it is
NOT yet a full, sound batched-matmul proof.

## The problem

The expert-major FFN needs `g[pos][j] = sum_d seg[pos][d] * W[expert_of[pos]][d][j]`,
i.e. a matmul whose weight matrix is per-position (depends on `pos`). The ordinary
matmul sumcheck does not transfer: for shared weights the MLE identity

    C~(u,v) = sum_d A~(u,d) * B~(d,v)

is multilinear in `u` because `B~` does not depend on `u`. With a per-position weight
both `A~` and `B~` depend on `u`, so

    G~(u,v) = sum_d S~(u,d) * W~(u,d,v)

is degree-2 in `u` and is NOT the MLE of `g` (verified numerically and symbolically on
a tiny case). So a single-contraction sumcheck at a random MLE point is unsound for
per-position weights.

## Correct approaches (to build next)

1. eq-selector virtual sumcheck over (S, K, N):
   `G~(u,v) = sum_{pos,j,d} seg[pos][d] * w[pos][d][j] * eq((pos,j),(u,v))`.
   Reuses `sumcheck::prove_virtual`; cost is O(S+K+N) rounds, but needs tensor
   broadcasting to the (S,K,N) domain.
2. Random linear combination over positions (batch S matmuls into one):
   `sum_pos r_pos g[pos][v] = sum_{pos,d} (r_pos seg[pos][d]) * w[pos][d][v]`.
   Sound with random `r`, but the `r o seg` folding needs an extra weighted claim on
   `seg`.

The concrete-index primitive is the correct inner dot-product for either approach.

## Related

The weight gather `W_gathered = Lookup(idx(expert_of), W_all)` reuses the existing
`Lookup` op and needs no new primitive. `Scatter` / `Gather` / permutation argument are
already implemented and integrated.

