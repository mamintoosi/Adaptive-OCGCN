"""
Ablation study for Adaptive-OCGCN.

Configurations:
  A0: no overlap (WMC=1.0)
  A1: original_wmc (WMC=0.3)
  A2-A4: entropy_adaptive (λ = 0.25 / 0.50 / 0.75)
  A5-A7: margin_adaptive (λ = 0.25 / 0.50 / 0.75)
  A8-A10: hybrid_adaptive (λ = 0.25 / 0.50 / 0.75)

Datasets: Cora, CiteSeer, PubMed, ACM, DBLP, IMDB
Run: python run_ablation.py [--seeds 10]
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

CONFIGS = [
    ("A0_no_overlap",     "no_overlap",          {},                       {}),
    ("A1_original_wmc",   "original_wmc",        {"membership_closeness": 0.3}, {}),
    ("A2_entropy_l025",   "entropy_adaptive_wmc", {"membership_closeness": 0.3}, {"lam": 0.25}),
    ("A3_entropy_l050",   "entropy_adaptive_wmc", {"membership_closeness": 0.3}, {"lam": 0.50}),
    ("A4_entropy_l075",   "entropy_adaptive_wmc", {"membership_closeness": 0.3}, {"lam": 0.75}),
    ("A5_margin_l025",    "margin_adaptive_wmc",  {"membership_closeness": 0.3}, {"lam": 0.25}),
    ("A6_margin_l050",    "margin_adaptive_wmc",  {"membership_closeness": 0.3}, {"lam": 0.50}),
    ("A7_margin_l075",    "margin_adaptive_wmc",  {"membership_closeness": 0.3}, {"lam": 0.75}),
    ("A8_hybrid_l025",    "hybrid_adaptive_wmc",  {"membership_closeness": 0.3}, {"lam": 0.25}),
    ("A9_hybrid_l050",    "hybrid_adaptive_wmc",  {"membership_closeness": 0.3}, {"lam": 0.50}),
    ("A10_hybrid_l075",   "hybrid_adaptive_wmc",  {"membership_closeness": 0.3}, {"lam": 0.75}),
]


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    datasets = args.datasets

    ds_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    total = len(datasets) * len(CONFIGS) * len(seeds)
    done = 0

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*70}")

        graph, features, target = load_dataset(ds_name, ds_root)

        for config_name, strategy, base_kw, extra_kw in CONFIGS:
            for seed in seeds:
                done += 1
                kw = dict(
                    dataset_name=ds_name, ds_root=ds_root, clustering_method='danmf',
                    epochs=10, test_ratio=0.3, seed=seed, dropout=0.5,
                    learning_rate=0.01, cluster_number=NUM_LABELS[ds_name],
                    num_trial=1, layers=[16, 16, 16],
                    overlap_strategy=strategy,
                    adaptation_lambda=extra_kw.get('lam', 0.5),
                    **base_kw,
                )
                kw['clustering_overlap'] = (strategy != "no_overlap")

                args0 = SimpleArgs(**kw)
                danmf_result = fit_danmf_cached(graph, args0, seed)
                result = run_single(graph, features, target, args0, strategy, danmf_result)
                result["dataset"] = ds_name
                result["config"] = config_name
                result["strategy"] = strategy
                result["lambda"] = extra_kw.get('lam', 0.0 if strategy in ("no_overlap", "original_wmc") else 0.5)
                result["seed"] = seed
                all_results.append(result)

                if done % 30 == 0 or done == total:
                    print(f"  [{done}/{total}] {ds_name} {config_name} seed={seed}: "
                          f"F1(micro)={result['f1_micro']:.4f} F1(macro)={result['f1_macro']:.4f} "
                          f"overlap={result['overlap_ratio']:.2f}x")

    # Save raw results
    df = pd.DataFrame(all_results)
    raw_csv = os.path.join(results_dir, 'ablation_raw.csv')
    df.to_csv(raw_csv, index=False)

    # Summary
    print(f"\n{'='*70}")
    print("  ABLATION SUMMARY")
    print(f"{'='*70}")

    summary = df.groupby(['dataset', 'config']).agg(
        f1_micro_mean=('f1_micro', 'mean'),
        f1_micro_std=('f1_micro', 'std'),
        f1_macro_mean=('f1_macro', 'mean'),
        f1_macro_std=('f1_macro', 'std'),
        overlap_mean=('overlap_ratio', 'mean'),
        runtime_mean=('runtime', 'mean'),
    ).round(4)
    print(summary.to_string())

    # Pivot (macro)
    pivot = df.groupby(['dataset', 'config'])['f1_macro'].agg(['mean', 'std']).unstack('config')
    print(f"\n{'='*70}")
    print("  F1 MACRO PIVOT TABLE")
    print(f"{'='*70}")
    print(pivot.round(4).to_string())

    summary_csv = os.path.join(results_dir, 'ablation_summary.csv')
    summary.to_csv(summary_csv)

    print(f"\nRaw: {raw_csv}")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
