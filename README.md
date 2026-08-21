# Adaptive-OCGCN

Adaptive overlap thresholding for Overlapping Cluster-GCN using community-membership ambiguity.

## Problem

Overlapping Cluster-GCN (OCGCN) extends Cluster-GCN by allowing nodes to participate in multiple clusters. The original OCGCN uses a single global Winner Membership Closeness (WMC) threshold applied uniformly to all nodes. This requires per-dataset tuning and ignores the fact that nodes vary enormously in their membership ambiguity.

## Method

We propose three adaptive overlap selection methods that replace the global WMC with a node-specific threshold derived from each node's membership distribution:

**Entropy-Adaptive WMC** — scales the threshold by normalized membership entropy:

```
WMC_i = WMC_base * (1 - lambda * H_norm(i))
```

Nodes with high entropy (spread across many clusters) get a lower threshold, allowing them to join more clusters.

**Margin-Adaptive WMC** — scales by the ambiguity of the top-two membership gap:

```
A(i) = 1 - (p1 - p2)
WMC_i = WMC_base * (1 - lambda * A(i))
```

Nodes caught between two dominant communities get a lower threshold.

**Hybrid-Adaptive WMC** — uses the stronger of the two signals per node:

```
a_i = max(H_norm(i), A(i))
WMC_i = WMC_base * (1 - lambda * a_i)
```

All methods eliminate the need for manual WMC tuning while achieving competitive performance across diverse graph datasets.

## Project Structure

```
Adaptive-OCGCN/
├── src/
│   ├── overlap_selection/           # Core: overlap selection strategies
│   │   ├── common.py                # Base class + entropy/margin utilities
│   │   ├── original_wmc.py          # Global WMC baseline
│   │   ├── entropy_adaptive_wmc.py  # Entropy-adaptive variant
│   │   ├── margin_adaptive_wmc.py   # Margin-adaptive variant
│   │   └── hybrid_adaptive_wmc.py   # Hybrid (max) variant
│   ├── clustergcn.py                # Cluster-GCN implementation (micro+macro F1)
│   ├── clustering.py                # Graph clustering (DANMF, spectral, random)
│   ├── ra_ocgcn_clustering.py       # Adaptive-WMC clustering pipeline
│   ├── layers.py                    # GCN layers
│   ├── utils.py                     # Data loading, evaluation utilities
│   ├── hetero_utils.py              # Heterogeneous graph utilities
│   ├── experiment_utils.py          # Shared experiment helpers (seeding, DANMF cache)
│   ├── parser.py                    # Argument parsing
│   ├── main.py                      # Main entry point
│   ├── main_ra_ocgcn.py             # Adaptive-OCGCN strategy comparison entry point
│   ├── run_full_experiments.py      # Full experiment suite
│   ├── run_ablation.py              # Ablation study runner (λ sensitivity)
│   ├── run_ambiguity_analysis.py    # Ambiguity analysis
│   ├── run_full_ambiguity_analysis.py
│   ├── run_fixed_wmc_comparison.py  # Fixed vs adaptive WMC comparison + stats
│   ├── run_standard_baselines.py    # Standard full-batch baselines
│   ├── measure_runtime.py           # Runtime measurement
│   ├── generate_tsne.py             # t-SNE visualization
│   ├── generate_tsne_comparison.py  # Comparative t-SNE (baseline vs adaptive)
│   └── generate_tsne_danmf_clusters.py  # DANMF cluster t-SNE
├── results/                         # Experimental results
│   ├── plots/                       # Generated figures
│   └── *.csv                        # Raw and summary results
├── docs/
│   ├── ALGORITHM_OVERVIEW.md        # Algorithm description
│   └── CODEBASE_OVERVIEW.md         # Codebase documentation
└── tmp/                             # Datasets (gitignored)
```

## Datasets

| Dataset  | Type          | Nodes   | Edges    | Classes |
|----------|---------------|---------|----------|---------|
| Cora     | Citation      | 2,708   | 5,278    | 7       |
| CiteSeer | Citation      | 3,327   | 4,552    | 6       |
| PubMed   | Citation      | 19,717  | 44,324   | 3       |
| ACM      | Heterogeneous | 2,363   | 5,318    | 3       |
| DBLP     | Heterogeneous | 2,591   | 3,528    | 4       |
| IMDB     | Heterogeneous | 4,630   | 54,576   | 5       |

## Usage

### Overlap Selection

```python
from src.overlap_selection import create_selector

# Entropy-adaptive with default lambda=0.5
selector = create_selector("entropy_adaptive_wmc", membership_closeness=0.3, lam=0.5)
assignments = selector.select_overlap(P, valid_clusters)

# Margin-adaptive
selector = create_selector("margin_adaptive_wmc", membership_closeness=0.3, lam=0.5)
assignments = selector.select_overlap(P, valid_clusters)

# Hybrid (max of entropy and margin)
selector = create_selector("hybrid_adaptive_wmc", membership_closeness=0.3, lam=0.5)
assignments = selector.select_overlap(P, valid_clusters)

# Original global WMC (baseline)
selector = create_selector("original_wmc", membership_closeness=0.3)
assignments = selector.select_overlap(P, valid_clusters)
```

### Running Experiments

```bash
# Full experiment suite (6 datasets, 10 seeds)
python src/run_full_experiments.py --seeds 10

# Ablation study (lambda sensitivity)
python src/run_ablation.py --seeds 10

# Fixed WMC vs adaptive comparison + statistical tests (20 seeds)
python src/run_fixed_wmc_comparison.py --seeds 20

# Ambiguity analysis
python src/run_full_ambiguity_analysis.py
```

### Compiling the Paper

```bash
cd adaptive-ocgcn-latex
pdflatex sn-article.tex
```

## Key Results

- Adaptive thresholding eliminates the WMC hyperparameter while maintaining competitive performance
- Margin-Adaptive (and Hybrid-Adaptive) with lambda=0.50 are reasonable defaults across all datasets
- High-entropy nodes are 41-95% overlap candidates vs 0-17% for low-entropy nodes
- At matched overlap ratios, adaptive methods win on most datasets
- Fully seeded runs (torch/numpy/python/DANMF) + cached DANMF per (dataset, seed) give a clean paired design
- Both micro and macro F1 are reported

## Installation

```bash
pip install -r requirements.txt
```

The `src/` modules must be importable; run scripts from the repository root or add `src/` to `PYTHONPATH`.

## Citation

If you use this code, please cite:

> M. Amintoosi, *Adaptive-OCGCN: Ambiguity-Aware Overlap Assignment for Scalable Graph Neural Networks*, (submitted).
