# Algorithm Overview

Adaptive-OCGCN replaces the single global *Winner Membership Closeness* (WMC)
threshold of Overlapping Cluster-GCN with node-specific thresholds derived
from each node's community-membership ambiguity.

## Pipeline

1. **Soft community assignment (DANMF).**
   DANMF factorizes the adjacency matrix into a soft membership matrix
   `P ∈ R^(N×K)`, where `P(i,c)` is the strength of node `i` in community `c`
   (rows are L1-normalized).

2. **Overlap assignment (adaptive thresholding).**
   For every node `i`, a per-node threshold is computed as

   ```
   WMC_i = WMC_base · (1 − λ · a_i)
   ```

   and node `i` is assigned to cluster `c` iff `P(i,c) ≥ WMC_i · max_c' P(i,c')`.
   `λ = 0` reduces to the original global-WMC OCGCN. `WMC_i` is floored at
   0.01 to avoid degenerate all-cluster assignments.

3. **Ambiguity scores** (all in `[0,1]`):

   - *Normalized entropy*: `H_norm(i) = −Σ_c P(i,c)·log P(i,c) / log K` — global
     spread of the membership distribution.
   - *Margin ambiguity*: `A(i) = 1 − (p1 − p2)` where `p1,p2` are the top two
     memberships — competition between the two dominant communities.
   - *Hybrid*: `a_i = max(H_norm(i), A(i))` — reacts to whichever signal is
     stronger for the node.

4. **Overlapping subgraph construction.**
   Overlap nodes are duplicated into each assigned cluster subgraph,
   preserving features and labels.

5. **Training and evaluation.**
   A shared-weight multi-layer GCN is trained cluster-by-cluster (Adam,
   NLL loss on train nodes). Micro and macro F1 are computed on the
   concatenated test predictions.

## Reproducibility design

- Every run is fully seeded: `torch`, `numpy`, `random`, and DANMF's internal
  seed are all set from the run seed.
- The DANMF decomposition is cached per `(dataset, seed)` and shared across
  all overlap strategies, giving a clean **paired design**: strategies on the
  same seed see identical community memberships and train/test splits, so
  paired statistical tests are valid.

## Statistical tests

- Paired t-test and Wilcoxon signed-rank test per dataset:
  adaptive (default λ=0.5) vs. best fixed WMC (oracle) **and** vs. the
  untuned default WMC=0.30.
- Matched-overlap "fair comparison": each adaptive method is compared to the
  fixed WMC whose mean overlap ratio is closest, isolating overlap *quality*
  from overlap *quantity*.
- Aggregate across datasets: binomial sign test on matched-overlap winners
  and a paired t-test on per-dataset mean differences.
