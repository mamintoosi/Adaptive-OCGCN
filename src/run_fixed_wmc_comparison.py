"""
Fixed-WMC Baseline Study for Adaptive-OCGCN.

Determines whether Adaptive-WMC improves because of:
A) smarter overlap-node selection
B) simply generating more overlap

Tasks:
1. Fixed-WMC sweep (0.1-0.5)
2. Compare with best adaptive variants
3. Overlap efficiency analysis
4. Fair comparison at similar overlap ratios
5. Statistical testing (adaptive vs. best fixed WMC AND adaptive vs. default WMC=0.30)

Key enhancements over the original script:
  - fully seeded runs (torch/numpy/python/DANMF)
  - DANMF decomposition cached per (dataset, seed) -> clean paired design
  - both micro and macro F1 reported
  - hybrid adaptive strategy included
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_utils import (
    NUM_LABELS,
    SimpleArgs,
    fit_danmf_cached,
    load_dataset,
    run_single,
)

DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'ACM', 'DBLP', 'IMDB']

FIXED_WMC_VALUES = [0.10, 0.20, 0.30, 0.40, 0.50]

ADAPTIVE_CONFIGS = [
    ("entropy_l050", "entropy_adaptive_wmc", {"membership_closeness": 0.3, "adaptation_lambda": 0.5}),
    ("margin_l050", "margin_adaptive_wmc", {"membership_closeness": 0.3, "adaptation_lambda": 0.5}),
    ("hybrid_l050", "hybrid_adaptive_wmc", {"membership_closeness": 0.3, "adaptation_lambda": 0.5}),
]


def make_args(ds_name, ds_root, seed, strategy, **extra):
    kw = dict(
        dataset_name=ds_name, ds_root=ds_root, clustering_method='danmf',
        epochs=10, test_ratio=0.3, seed=seed, dropout=0.5,
        learning_rate=0.01, cluster_number=NUM_LABELS[ds_name],
        num_trial=1, layers=[16, 16, 16],
        overlap_strategy=strategy,
    )
    kw.update(extra)
    return SimpleArgs(**kw)


def main():
    parser = argparse.ArgumentParser(description="Fixed-WMC baseline study")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds")
    parser.add_argument("--datasets", nargs="*", default=DATASETS,
                        help="Datasets to evaluate (default: all six)")
    parser.add_argument("--recompute-only", action="store_true",
                        help="Skip training; recompute all analysis CSVs from the saved fixed_wmc_raw.csv")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    datasets = args.datasets

    ds_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    if args.recompute_only:
        raw_path = os.path.join(results_dir, 'fixed_wmc_raw.csv')
        if not os.path.exists(raw_path):
            parser.error(f"{raw_path} not found; run without --recompute-only first")
        print(f"Recomputing analysis from {raw_path} (no training)...")
        df = pd.read_csv(raw_path)
        present = set(df['dataset'].unique())
        datasets = [d for d in DATASETS if d in present]
    else:
        for ds_name in datasets:
            print(f"\n{'='*70}")
            print(f"  Dataset: {ds_name}")
            print(f"{'='*70}")

            graph, features, target = load_dataset(ds_name, ds_root)

            for seed in seeds:
                # Cache DANMF once per (dataset, seed); every method below shares it.
                args0 = make_args(ds_name, ds_root, seed, "original_wmc")
                danmf_result = fit_danmf_cached(graph, args0, seed)

                # Baseline: no overlap
                args_no = make_args(ds_name, ds_root, seed, "no_overlap", clustering_overlap=False)
                result = run_single(graph, features, target, args_no, "no_overlap", danmf_result)
                result.update({"dataset": ds_name, "method": "fixed_wmc",
                               "wmc": 1.0, "seed": seed, "label": "no_overlap"})
                all_results.append(result)

                # Fixed WMC sweep
                for wmc in FIXED_WMC_VALUES:
                    args_f = make_args(ds_name, ds_root, seed, "original_wmc",
                                       clustering_overlap=True, membership_closeness=wmc)
                    result = run_single(graph, features, target, args_f, "original_wmc", danmf_result)
                    result.update({"dataset": ds_name, "method": "fixed_wmc",
                                   "wmc": wmc, "seed": seed, "label": f"WMC_{wmc:.2f}"})
                    all_results.append(result)

                # Adaptive methods
                for label, strategy, extra_kw in ADAPTIVE_CONFIGS:
                    args_a = make_args(ds_name, ds_root, seed, strategy,
                                       clustering_overlap=True, **extra_kw)
                    result = run_single(graph, features, target, args_a, strategy, danmf_result)
                    result.update({"dataset": ds_name, "method": "adaptive",
                                   "wmc": extra_kw['membership_closeness'], "seed": seed,
                                   "label": label})
                    all_results.append(result)

        # Save raw results
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(results_dir, 'fixed_wmc_raw.csv'), index=False)

    # ============================================================
    # TASK 2: Summary comparison
    # ============================================================
    print(f"\n{'='*70}")
    print("  TASK 2: METHOD COMPARISON")
    print(f"{'='*70}")

    summary = df.groupby(['dataset', 'label']).agg(
        f1_micro_mean=('f1_micro', 'mean'),
        f1_micro_std=('f1_micro', 'std'),
        f1_macro_mean=('f1_macro', 'mean'),
        f1_macro_std=('f1_macro', 'std'),
        overlap_mean=('overlap_ratio', 'mean'),
        runtime_mean=('runtime', 'mean'),
    ).round(4)
    summary.to_csv(os.path.join(results_dir, 'fixed_wmc_summary.csv'))
    print(summary[['f1_micro_mean', 'f1_micro_std', 'overlap_mean']].to_string())

    # ============================================================
    # TASK 3: Overlap Efficiency
    # ============================================================
    print(f"\n{'='*70}")
    print("  TASK 3: OVERLAP EFFICIENCY")
    print(f"{'='*70}")

    no_overlap = df[df['label'] == 'no_overlap'].groupby('dataset')['f1_micro'].mean()
    efficiency_results = []

    for label in df['label'].unique():
        if label == 'no_overlap':
            continue
        method_df = df[df['label'] == label]
        for ds in datasets:
            ds_method = method_df[method_df['dataset'] == ds]
            ds_nooverlap = no_overlap.get(ds, 0)
            if len(ds_method) == 0 or ds_nooverlap == 0:
                continue
            f1_gain = ds_method['f1_micro'].mean() - ds_nooverlap
            overlap_gain = ds_method['overlap_ratio'].mean() - 1.0
            efficiency = f1_gain / max(overlap_gain, 0.001)
            efficiency_results.append({
                'dataset': ds, 'label': label,
                'f1_gain': f1_gain, 'overlap_gain': overlap_gain,
                'efficiency': efficiency,
            })

    eff_df = pd.DataFrame(efficiency_results)
    eff_df.to_csv(os.path.join(results_dir, 'overlap_efficiency.csv'), index=False)
    print(eff_df.pivot_table(index='label', columns='dataset', values='efficiency').round(4).to_string())

    # ============================================================
    # TASK 4: Fair comparison at similar overlap ratios
    # ============================================================
    print(f"\n{'='*70}")
    print("  TASK 4: FAIR COMPARISON AT SIMILAR OVERLAP")
    print(f"{'='*70}")

    fair_results = []
    for ds in datasets:
        ds_df = df[df['dataset'] == ds]
        for adapt_label, _, _ in ADAPTIVE_CONFIGS:
            adapt_data = ds_df[ds_df['label'] == adapt_label]
            if len(adapt_data) == 0:
                continue
            adapt_overlap = adapt_data['overlap_ratio'].mean()
            adapt_f1 = adapt_data['f1_micro'].mean()

            fixed_data = ds_df[ds_df['method'] == 'fixed_wmc']
            fixed_overlaps = fixed_data.groupby('wmc')['overlap_ratio'].mean()
            closest_wmc = fixed_overlaps.iloc[(fixed_overlaps - adapt_overlap).abs().argsort()[:1]].index[0]
            closest_fixed = ds_df[(ds_df['method'] == 'fixed_wmc') & (ds_df['wmc'] == closest_wmc)]
            fixed_f1 = closest_fixed['f1_micro'].mean()
            fixed_overlap = closest_fixed['overlap_ratio'].mean()

            fair_results.append({
                'dataset': ds, 'adaptive': adapt_label,
                'adapt_overlap': adapt_overlap, 'adapt_f1': adapt_f1,
                'fixed_wmc': closest_wmc, 'fixed_overlap': fixed_overlap, 'fixed_f1': fixed_f1,
                'f1_diff': adapt_f1 - fixed_f1,
            })

    fair_df = pd.DataFrame(fair_results)
    fair_df.to_csv(os.path.join(results_dir, 'fair_comparison.csv'), index=False)
    print(fair_df.to_string(index=False))

    # ============================================================
    # TASK 5: Statistical testing
    # ============================================================
    print(f"\n{'='*70}")
    print("  TASK 5: STATISTICAL TESTING")
    print(f"{'='*70}")

    stat_results = []
    for ds in datasets:
        ds_df = df[df['dataset'] == ds]
        for adapt_label, _, _ in ADAPTIVE_CONFIGS:
            # Align runs by seed so pairs are always correct, regardless of
            # the row order of the raw frame.
            adapt_series = ds_df[ds_df['label'] == adapt_label].set_index('seed')['f1_micro']
            if len(adapt_series) == 0:
                continue

            # (a) vs BEST fixed WMC (oracle baseline)
            fixed_summary = ds_df[ds_df['method'] == 'fixed_wmc'].groupby('wmc')['f1_micro'].mean()
            best_fixed_wmc = fixed_summary.idxmax()
            best_fixed_series = ds_df[(ds_df['method'] == 'fixed_wmc') &
                                      (ds_df['wmc'] == best_fixed_wmc)].set_index('seed')['f1_micro']
            common = adapt_series.index.intersection(best_fixed_series.index)
            adapt_f1s = adapt_series.loc[common].values
            best_fixed_f1s = best_fixed_series.loc[common].values

            if len(adapt_f1s) > 1:
                t_stat, t_pval = ttest_rel(adapt_f1s, best_fixed_f1s)
                try:
                    w_stat, w_pval = wilcoxon(adapt_f1s, best_fixed_f1s)
                except ValueError:
                    w_stat, w_pval = 0, 1.0
                effect_size = (adapt_f1s.mean() - best_fixed_f1s.mean()) / max(
                    adapt_f1s.std(), best_fixed_f1s.std(), 1e-10)

                stat_results.append({
                    'dataset': ds, 'adaptive': adapt_label,
                    'vs': 'best_fixed',
                    'baseline': f'WMC={best_fixed_wmc:.2f}',
                    'adapt_mean': adapt_f1s.mean(), 'base_mean': best_fixed_f1s.mean(),
                    't_stat': t_stat, 't_pval': t_pval,
                    'w_stat': w_stat, 'w_pval': w_pval,
                    'effect_size': effect_size,
                })

            # (b) vs DEFAULT (untuned) fixed WMC=0.30
            default_series = ds_df[(ds_df['method'] == 'fixed_wmc') &
                                   (ds_df['wmc'] == 0.30)].set_index('seed')['f1_micro']
            common = adapt_series.index.intersection(default_series.index)
            adapt_f1s = adapt_series.loc[common].values
            default_f1s = default_series.loc[common].values
            if len(adapt_f1s) > 1:
                t_stat, t_pval = ttest_rel(adapt_f1s, default_f1s)
                try:
                    w_stat, w_pval = wilcoxon(adapt_f1s, default_f1s)
                except ValueError:
                    w_stat, w_pval = 0, 1.0
                effect_size = (adapt_f1s.mean() - default_f1s.mean()) / max(
                    adapt_f1s.std(), default_f1s.std(), 1e-10)

                stat_results.append({
                    'dataset': ds, 'adaptive': adapt_label,
                    'vs': 'default_wmc030',
                    'baseline': 'WMC=0.30',
                    'adapt_mean': adapt_f1s.mean(), 'base_mean': default_f1s.mean(),
                    't_stat': t_stat, 't_pval': t_pval,
                    'w_stat': w_stat, 'w_pval': w_pval,
                    'effect_size': effect_size,
                })

    stat_df = pd.DataFrame(stat_results)
    stat_df.to_csv(os.path.join(results_dir, 'statistical_tests.csv'), index=False)
    print(stat_df.round(4).to_string(index=False))

    # ============================================================
    # TASK 6: Aggregate (across-dataset) significance
    # ============================================================
    print(f"\n{'='*70}")
    print("  TASK 6: AGGREGATE ACROSS-DATASET ANALYSIS")
    print(f"{'='*70}")

    # 6a. Sign test on the matched-overlap comparison winners
    for adapt_label, _, _ in ADAPTIVE_CONFIGS:
        f = fair_df[fair_df['adaptive'] == adapt_label]
        wins = (f['f1_diff'] > 0).sum()
        n = len(f)
        if n > 0:
            p = binomtest(wins, n, 0.5, alternative='greater').pvalue
            print(f"  {adapt_label}: matched-overlap wins {wins}/{n} "
                  f"(sign-test p={p:.4f})")

    # 6b. Paired test on per-dataset mean differences (adaptive - default WMC)
    print("\n  Per-dataset mean difference (adaptive micro-F1 minus default WMC=0.30):")
    agg_rows = []
    for adapt_label, _, _ in ADAPTIVE_CONFIGS:
        diffs = []
        for ds in datasets:
            ds_df = df[df['dataset'] == ds]
            a = ds_df[ds_df['label'] == adapt_label]['f1_micro'].mean()
            b = ds_df[(ds_df['method'] == 'fixed_wmc') &
                      (ds_df['wmc'] == 0.30)]['f1_micro'].mean()
            diffs.append(a - b)
            print(f"    {ds:<10} {adapt_label:<16} {a-b:+.4f}")
        if len(diffs) >= 3:
            t_stat, t_pval = ttest_rel(diffs, np.zeros(len(diffs)))
            agg_rows.append({'adaptive': adapt_label, 'mean_diff': np.mean(diffs),
                             'n_datasets': len(diffs), 't_stat': t_stat, 't_pval': t_pval})
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(os.path.join(results_dir, 'aggregate_tests.csv'), index=False)
    if len(agg_df):
        print(agg_df.round(4).to_string(index=False))

    print(f"\nAll results saved to: {results_dir}/")
    print(f"  fixed_wmc_raw.csv, fixed_wmc_summary.csv, overlap_efficiency.csv")
    print(f"  fair_comparison.csv, statistical_tests.csv, aggregate_tests.csv")


if __name__ == "__main__":
    main()
