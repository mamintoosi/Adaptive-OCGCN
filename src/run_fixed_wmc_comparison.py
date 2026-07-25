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
5. Statistical testing
"""
import sys
import os
import time
import torch
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clustering import ClusteringMachine
from ra_ocgcn_clustering import AdaptiveWMCClusteringMachine
from clustergcn import ClusterGCNTrainer
from hetero_utils import HETERO_LOADERS


class SimpleArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def load_citation_dataset(ds_name, ds_root):
    import sys as _sys
    _sys.argv = ['main.py', '--dataset-name', ds_name, '--ds-root', ds_root]
    from parser import parameter_parser
    from utils import dataset_reader
    args = parameter_parser()
    graph, features, target = dataset_reader(args)
    return graph, features, target


def load_hetero_dataset(ds_name, ds_root):
    loader = HETERO_LOADERS[ds_name]
    graph, features, target = loader(ds_root)
    isolates = list(nx.isolates(graph))
    if isolates:
        graph.remove_nodes_from(isolates)
        features = np.delete(features, isolates, axis=0)
        non_isolates = [n for n in range(len(target)) if n not in set(isolates)]
        target = target[non_isolates]
    mapping = {old: new for new, old in enumerate(sorted(graph.nodes()))}
    graph = nx.relabel_nodes(graph, mapping)
    return graph, features, target


def run_single(graph, features, target, args, method):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    start = time.time()
    if method == "no_overlap":
        cm = ClusteringMachine(args, graph, features, target)
    else:
        cm = AdaptiveWMCClusteringMachine(args, graph, features, target)
    cm.decompose()
    trainer = ClusterGCNTrainer(args, cm)
    trainer.train()
    score = trainer.test()
    elapsed = time.time() - start
    avg_overlap = np.sum(cm.ClusterNodes) / len(graph.nodes())
    return {"f1": score, "runtime": elapsed, "overlap_ratio": avg_overlap}


def main():
    ds_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    seeds = list(range(10))
    num_labels = {'Cora': 7, 'CiteSeer': 6, 'PubMed': 3, 'ACM': 3, 'DBLP': 4, 'IMDB': 5}
    datasets = ['Cora', 'CiteSeer', 'PubMed', 'ACM', 'DBLP', 'IMDB']

    # Fixed WMC values to sweep
    fixed_wmc_values = [0.10, 0.20, 0.30, 0.40, 0.50]

    # Best adaptive variants (from previous experiments)
    adaptive_configs = [
        ("entropy_l050", "entropy_adaptive_wmc", {"membership_closeness": 0.3, "adaptation_lambda": 0.5}),
        ("margin_l050",  "margin_adaptive_wmc",  {"membership_closeness": 0.3, "adaptation_lambda": 0.5}),
    ]

    all_results = []
    total_experiments = len(datasets) * (len(fixed_wmc_values) + len(adaptive_configs) + 1) * len(seeds)
    done = 0

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*70}")

        is_hetero = ds_name in ('ACM', 'DBLP', 'IMDB')
        if is_hetero:
            graph, features, target = load_hetero_dataset(ds_name, ds_root)
        else:
            graph, features, target = load_citation_dataset(ds_name, ds_root)

        # Baseline: no overlap
        for seed in seeds:
            done += 1
            kw = dict(
                dataset_name=ds_name, ds_root=ds_root, clustering_method='danmf',
                epochs=10, test_ratio=0.3, seed=seed, dropout=0.5,
                learning_rate=0.01, cluster_number=num_labels[ds_name],
                num_trial=1, layers=[16, 16, 16],
                clustering_overlap=False,
            )
            args = SimpleArgs(**kw)
            result = run_single(graph, features, target, args, "no_overlap")
            result.update({"dataset": ds_name, "method": "fixed_wmc",
                           "wmc": 1.0, "seed": seed, "label": "no_overlap"})
            all_results.append(result)
            if done % 20 == 0:
                print(f"  [{done}/{total_experiments}] {ds_name} no_overlap seed={seed}: F1={result['f1']:.4f}")

        # Fixed WMC sweep
        for wmc in fixed_wmc_values:
            for seed in seeds:
                done += 1
                kw = dict(
                    dataset_name=ds_name, ds_root=ds_root, clustering_method='danmf',
                    epochs=10, test_ratio=0.3, seed=seed, dropout=0.5,
                    learning_rate=0.01, cluster_number=num_labels[ds_name],
                    num_trial=1, layers=[16, 16, 16],
                    clustering_overlap=True, membership_closeness=wmc,
                    overlap_strategy='original_wmc',
                )
                args = SimpleArgs(**kw)
                result = run_single(graph, features, target, args, "original_wmc")
                result.update({"dataset": ds_name, "method": "fixed_wmc",
                               "wmc": wmc, "seed": seed, "label": f"WMC_{wmc:.2f}"})
                all_results.append(result)
                if done % 20 == 0:
                    print(f"  [{done}/{total_experiments}] {ds_name} WMC={wmc:.2f} seed={seed}: F1={result['f1']:.4f}")

        # Adaptive methods
        for label, strategy, extra_kw in adaptive_configs:
            for seed in seeds:
                done += 1
                kw = dict(
                    dataset_name=ds_name, ds_root=ds_root, clustering_method='danmf',
                    epochs=10, test_ratio=0.3, seed=seed, dropout=0.5,
                    learning_rate=0.01, cluster_number=num_labels[ds_name],
                    num_trial=1, layers=[16, 16, 16],
                    clustering_overlap=True, overlap_strategy=strategy,
                    **extra_kw,
                )
                args = SimpleArgs(**kw)
                result = run_single(graph, features, target, args, strategy)
                result.update({"dataset": ds_name, "method": "adaptive",
                               "wmc": extra_kw['membership_closeness'], "seed": seed,
                               "label": label})
                all_results.append(result)
                if done % 20 == 0:
                    print(f"  [{done}/{total_experiments}] {ds_name} {label} seed={seed}: F1={result['f1']:.4f}")

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
        f1_mean=('f1', 'mean'),
        f1_std=('f1', 'std'),
        overlap_mean=('overlap_ratio', 'mean'),
        runtime_mean=('runtime', 'mean'),
    ).round(4)
    summary.to_csv(os.path.join(results_dir, 'fixed_wmc_summary.csv'))
    print(summary.to_string())

    # ============================================================
    # TASK 3: Overlap Efficiency
    # ============================================================
    print(f"\n{'='*70}")
    print("  TASK 3: OVERLAP EFFICIENCY")
    print(f"{'='*70}")

    no_overlap = df[df['label'] == 'no_overlap'].groupby('dataset')['f1'].mean()
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
            f1_gain = ds_method['f1'].mean() - ds_nooverlap
            overlap_gain = ds_method['overlap_ratio'].mean() - 1.0  # relative to no-overlap
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
        for adapt_label in ['entropy_l050', 'margin_l050']:
            adapt_data = ds_df[ds_df['label'] == adapt_label]
            if len(adapt_data) == 0:
                continue
            adapt_overlap = adapt_data['overlap_ratio'].mean()
            adapt_f1 = adapt_data['f1'].mean()

            # Find closest fixed WMC
            fixed_data = ds_df[ds_df['method'] == 'fixed_wmc']
            fixed_overlaps = fixed_data.groupby('wmc')['overlap_ratio'].mean()
            closest_wmc = fixed_overlaps.iloc[(fixed_overlaps - adapt_overlap).abs().argsort()[:1]].index[0]
            closest_fixed = ds_df[ds_df['wmc'] == closest_wmc]
            fixed_f1 = closest_fixed['f1'].mean()
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

        # Best adaptive vs best fixed
        adapt_labels = ['entropy_l050', 'margin_l050']
        for adapt_label in adapt_labels:
            adapt_f1s = ds_df[ds_df['label'] == adapt_label]['f1'].values
            if len(adapt_f1s) == 0:
                continue

            # Find best fixed WMC
            fixed_summary = ds_df[ds_df['method'] == 'fixed_wmc'].groupby('wmc')['f1'].mean()
            best_fixed_wmc = fixed_summary.idxmax()
            best_fixed_f1s = ds_df[ds_df['wmc'] == best_fixed_wmc]['f1'].values

            if len(adapt_f1s) == len(best_fixed_f1s):
                t_stat, t_pval = ttest_rel(adapt_f1s, best_fixed_f1s)
                try:
                    w_stat, w_pval = wilcoxon(adapt_f1s, best_fixed_f1s)
                except ValueError:
                    w_stat, w_pval = 0, 1.0

                effect_size = (adapt_f1s.mean() - best_fixed_f1s.mean()) / max(adapt_f1s.std(), best_fixed_f1s.std(), 1e-10)

                stat_results.append({
                    'dataset': ds, 'adaptive': adapt_label,
                    'best_fixed_wmc': best_fixed_wmc,
                    'adapt_mean': adapt_f1s.mean(), 'fixed_mean': best_fixed_f1s.mean(),
                    't_stat': t_stat, 't_pval': t_pval,
                    'w_stat': w_stat, 'w_pval': w_pval,
                    'effect_size': effect_size,
                })

    stat_df = pd.DataFrame(stat_results)
    stat_df.to_csv(os.path.join(results_dir, 'statistical_tests.csv'), index=False)
    print(stat_df.to_string(index=False))

    print(f"\nAll results saved to: {results_dir}/")
    print(f"  fixed_wmc_raw.csv")
    print(f"  fixed_wmc_summary.csv")
    print(f"  overlap_efficiency.csv")
    print(f"  fair_comparison.csv")
    print(f"  statistical_tests.csv")


if __name__ == "__main__":
    main()
