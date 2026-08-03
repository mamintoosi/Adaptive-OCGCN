"""
Full experiments: all 6 datasets, all methods.

Methods:
1. Cluster-GCN (no overlap)
2. Overlapping-Cluster-GCN (original_wmc)
3. Entropy-Adaptive WMC
4. Margin-Adaptive WMC
5. Hybrid-Adaptive WMC

Datasets: Cora, CiteSeer, PubMed, ACM, DBLP, IMDB

Run: python run_full_experiments.py [--seeds 5]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_utils import (
    NUM_LABELS,
    SimpleArgs,
    fit_danmf_cached,
    load_dataset,
    run_single,
)

DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'ACM', 'DBLP', 'IMDB']
METHODS = ["no_overlap", "original_wmc", "entropy_adaptive_wmc", "margin_adaptive_wmc", "hybrid_adaptive_wmc"]


def main():
    parser = argparse.ArgumentParser(description="Full experiment suite")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    datasets = args.datasets
    wmc = 0.3
    lam = 0.5

    ds_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Loading: {ds_name}")
        print(f"{'='*70}")

        graph, features, target = load_dataset(ds_name, ds_root)
        print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

        for method in METHODS:
            print(f"\n  Method: {method}")

            for seed in seeds:
                kw = dict(
                    dataset_name=ds_name, ds_root=ds_root, clustering_method='danmf',
                    epochs=10, test_ratio=0.3, seed=seed, dropout=0.5,
                    learning_rate=0.01, cluster_number=NUM_LABELS[ds_name],
                    num_trial=1, layers=[16, 16, 16],
                    membership_closeness=wmc, overlap_strategy=method,
                    adaptation_lambda=lam,
                )
                kw['clustering_overlap'] = (method != "no_overlap")

                args0 = SimpleArgs(**kw)
                danmf_result = fit_danmf_cached(graph, args0, seed)
                result = run_single(graph, features, target, args0, method, danmf_result)
                result["dataset"] = ds_name
                result["method"] = method
                result["seed"] = seed
                result["wmc"] = wmc
                result["lambda"] = lam
                all_results.append(result)

                print(f"    seed={seed}: F1(micro)={result['f1_micro']:.4f} "
                      f"F1(macro)={result['f1_macro']:.4f} overlap={result['overlap_ratio']:.2f}x "
                      f"time={result['runtime']:.1f}s")

    # Save raw results
    df = pd.DataFrame(all_results)
    raw_csv = os.path.join(results_dir, 'full_experiment_raw.csv')
    df.to_csv(raw_csv, index=False)

    # Summary
    print(f"\n{'='*70}")
    print("  FULL RESULTS SUMMARY")
    print(f"{'='*70}")

    summary = df.groupby(['dataset', 'method']).agg(
        f1_micro_mean=('f1_micro', 'mean'),
        f1_micro_std=('f1_micro', 'std'),
        f1_macro_mean=('f1_macro', 'mean'),
        f1_macro_std=('f1_macro', 'std'),
        overlap_mean=('overlap_ratio', 'mean'),
        runtime_mean=('runtime', 'mean'),
    ).round(4)
    print(summary.to_string())

    # Pivot tables for both metrics
    print(f"\n{'='*70}")
    print("  F1 MICRO (mean ± std)")
    print(f"{'='*70}")
    pivot = df.groupby(['dataset', 'method'])['f1_micro'].agg(['mean', 'std']).unstack('method')
    pivot.columns = [f"{m}" for m in pivot.columns.get_level_values(1)]
    print(pivot.round(4).to_string())

    print(f"\n{'='*70}")
    print("  F1 MACRO (mean ± std)")
    print(f"{'='*70}")
    pivot_macro = df.groupby(['dataset', 'method'])['f1_macro'].agg(['mean', 'std']).unstack('method')
    pivot_macro.columns = [f"{m}" for m in pivot_macro.columns.get_level_values(1)]
    print(pivot_macro.round(4).to_string())

    summary_csv = os.path.join(results_dir, 'full_experiment_summary.csv')
    summary.to_csv(summary_csv)

    print(f"\nRaw results: {raw_csv}")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
