"""
Ablation study for Adaptive-OCGCN.

Configurations:
  A0: no overlap (WMC=1.0)
  A1: original_wmc (WMC=0.3)
  A2: entropy_adaptive (λ=0.25)
  A3: entropy_adaptive (λ=0.50)
  A4: entropy_adaptive (λ=0.75)
  A5: margin_adaptive (λ=0.25)
  A6: margin_adaptive (λ=0.50)
  A7: margin_adaptive (λ=0.75)

Datasets: Cora, CiteSeer, PubMed, ACM, DBLP, IMDB
Seeds: 10 per configuration
"""
import sys
import os
import time
import torch
import numpy as np
import pandas as pd
import networkx as nx

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

    return {
        "f1": score,
        "runtime": elapsed,
        "overlap_ratio": avg_overlap,
        "num_clusters": len(cm.clusters),
    }


def main():
    ds_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    seeds = list(range(10))  # 0-9
    num_labels = {'Cora': 7, 'CiteSeer': 6, 'PubMed': 3, 'ACM': 3, 'DBLP': 4, 'IMDB': 5}
    datasets = ['Cora', 'CiteSeer', 'PubMed', 'ACM', 'DBLP', 'IMDB']

    configs = [
        ("A0_no_overlap",     "no_overlap",          {},                    {}),
        ("A1_original_wmc",   "original_wmc",        {"membership_closeness": 0.3}, {}),
        ("A2_entropy_l025",   "entropy_adaptive_wmc", {"membership_closeness": 0.3}, {"lam": 0.25}),
        ("A3_entropy_l050",   "entropy_adaptive_wmc", {"membership_closeness": 0.3}, {"lam": 0.50}),
        ("A4_entropy_l075",   "entropy_adaptive_wmc", {"membership_closeness": 0.3}, {"lam": 0.75}),
        ("A5_margin_l025",    "margin_adaptive_wmc",  {"membership_closeness": 0.3}, {"lam": 0.25}),
        ("A6_margin_l050",    "margin_adaptive_wmc",  {"membership_closeness": 0.3}, {"lam": 0.50}),
        ("A7_margin_l075",    "margin_adaptive_wmc",  {"membership_closeness": 0.3}, {"lam": 0.75}),
    ]

    all_results = []
    total = len(datasets) * len(configs) * len(seeds)
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

        for config_name, strategy, base_kw, extra_kw in configs:
            for seed in seeds:
                done += 1
                kw = dict(
                    dataset_name=ds_name, ds_root=ds_root, clustering_method='danmf',
                    epochs=10, test_ratio=0.3, seed=seed, dropout=0.5,
                    learning_rate=0.01, cluster_number=num_labels[ds_name],
                    num_trial=1, layers=[16, 16, 16],
                    overlap_strategy=strategy, adaptation_lambda=extra_kw.get('lam', 0.5),
                    **base_kw,
                )
                if strategy == "no_overlap":
                    kw['clustering_overlap'] = False
                else:
                    kw['clustering_overlap'] = True

                args = SimpleArgs(**kw)
                result = run_single(graph, features, target, args, strategy)
                result["dataset"] = ds_name
                result["config"] = config_name
                result["strategy"] = strategy
                result["lambda"] = extra_kw.get('lam', 0.0 if strategy in ("no_overlap", "original_wmc") else 0.5)
                result["seed"] = seed
                all_results.append(result)

                if done % 30 == 0 or done == total:
                    print(f"  [{done}/{total}] {ds_name} {config_name} seed={seed}: "
                          f"F1={result['f1']:.4f} overlap={result['overlap_ratio']:.2f}x")

    # Save raw results
    df = pd.DataFrame(all_results)
    raw_csv = os.path.join(results_dir, 'ablation_raw.csv')
    df.to_csv(raw_csv, index=False)

    # Summary
    print(f"\n{'='*70}")
    print("  ABLATION SUMMARY")
    print(f"{'='*70}")

    summary = df.groupby(['dataset', 'config']).agg(
        f1_mean=('f1', 'mean'),
        f1_std=('f1', 'std'),
        overlap_mean=('overlap_ratio', 'mean'),
        runtime_mean=('runtime', 'mean'),
    ).round(4)
    print(summary.to_string())

    # Pivot
    pivot = df.groupby(['dataset', 'config'])['f1'].agg(['mean', 'std']).unstack('config')
    print(f"\n{'='*70}")
    print("  F1 PIVOT TABLE")
    print(f"{'='*70}")
    print(pivot.round(4).to_string())

    summary_csv = os.path.join(results_dir, 'ablation_summary.csv')
    summary.to_csv(summary_csv)

    print(f"\nRaw: {raw_csv}")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
