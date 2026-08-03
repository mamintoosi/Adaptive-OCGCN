"""
Measure DANMF one-time fitting cost per dataset and combine it with the
per-method GCN training runtimes already saved in the experiment CSVs to
produce results/runtime_summary.csv for the paper.

The DANMF decomposition is cached across all strategies and seeds (fixed
seed 42), so its cost is a ONE-TIME preprocessing expense per dataset,
separate from the per-run training time reported in fixed_wmc_summary.csv.

Run: python measure_runtime.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_utils import NUM_LABELS, load_dataset
from clustering import fit_danmf

DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'ACM', 'DBLP', 'IMDB']


def main():
    ds_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

    rows = []
    for ds in DATASETS:
        print(f'\n=== {ds} ===', flush=True)
        graph, features, target = load_dataset(ds, ds_root)
        # Record graph stats BEFORE fitting: fit_danmf mutates the graph by
        # adding self-loops (karateclub behaviour), which would inflate the
        # edge count if measured afterwards.
        n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
        t0 = time.time()
        fit_danmf(graph, NUM_LABELS[ds], seed=42)
        danmf_time = time.time() - t0
        print(f'  DANMF fit: {danmf_time:.2f}s (graph {n_nodes}N/{n_edges}E)', flush=True)

        # per-method training time (mean over 20 seeds) from the fixed-WMC study
        fw = pd.read_csv(os.path.join(results_dir, 'fixed_wmc_summary.csv'))
        fw = fw[fw['dataset'] == ds]
        training = {row['label']: row['runtime_mean'] for _, row in fw.iterrows()}

        rows.append({
            'dataset': ds,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'danmf_fit_s': round(danmf_time, 2),
            'train_no_overlap_s': round(training.get('no_overlap', float('nan')), 2),
            'train_wmc030_s': round(training.get('WMC_0.30', float('nan')), 2),
            'train_margin_s': round(training.get('margin_l050', float('nan')), 2),
            'train_hybrid_s': round(training.get('hybrid_l050', float('nan')), 2),
            'train_entropy_s': round(training.get('entropy_l050', float('nan')), 2),
        })

    df = pd.DataFrame(rows)
    out = os.path.join(results_dir, 'runtime_summary.csv')
    df.to_csv(out, index=False)
    print(f'\nSaved: {out}')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
