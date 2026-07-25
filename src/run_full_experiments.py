"""
Full experiments: all 6 datasets, all methods.

Methods:
1. Cluster-GCN (no overlap)
2. Overlapping-Cluster-GCN (original_wmc)
3. Entropy-Adaptive WMC
4. Margin-Adaptive WMC

Datasets: Cora, CiteSeer, PubMed, ACM, DBLP, IMDB
Seeds: 5 per configuration
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
    """Load citation dataset via torch_geometric."""
    import sys as _sys
    _sys.argv = ['main.py', '--dataset-name', ds_name, '--ds-root', ds_root]
    from parser import parameter_parser
    from utils import dataset_reader
    args = parameter_parser()
    graph, features, target = dataset_reader(args)
    return graph, features, target


def load_hetero_dataset(ds_name, ds_root):
    """Load heterogeneous dataset and convert to homogeneous graph."""
    loader = HETERO_LOADERS[ds_name]
    graph, features, target = loader(ds_root)

    # Remove isolates and relabel
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
    """Run one experiment and return F1 score."""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    start = time.time()

    if method == "no_overlap":
        cm = ClusteringMachine(args, graph, features, target)
    elif method in ("original_wmc", "entropy_adaptive_wmc", "margin_adaptive_wmc"):
        cm = AdaptiveWMCClusteringMachine(args, graph, features, target)
    else:
        raise ValueError(f"Unknown method: {method}")

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
    }


def main():
    ds_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    seeds = [42, 123, 456, 789, 1024]
    wmc = 0.3
    lam = 0.5
    epochs = 10
    test_ratio = 0.3

    num_labels = {
        'Cora': 7, 'CiteSeer': 6, 'PubMed': 3,
        'ACM': 3, 'DBLP': 4, 'IMDB': 5,
    }

    datasets = ['Cora', 'CiteSeer', 'PubMed', 'ACM', 'DBLP', 'IMDB']
    methods = ["no_overlap", "original_wmc", "entropy_adaptive_wmc", "margin_adaptive_wmc"]

    all_results = []

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Loading: {ds_name}")
        print(f"{'='*70}")

        is_hetero = ds_name in ('ACM', 'DBLP', 'IMDB')
        if is_hetero:
            graph, features, target = load_hetero_dataset(ds_name, ds_root)
        else:
            graph, features, target = load_citation_dataset(ds_name, ds_root)

        print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}, "
              f"Classes: {np.max(target)+1 if target.ndim==1 else target.shape[1]}")

        for method in methods:
            print(f"\n  Method: {method}")

            for seed in seeds:
                kw = dict(
                    dataset_name=ds_name,
                    ds_root=ds_root,
                    clustering_method='danmf',
                    epochs=epochs,
                    test_ratio=test_ratio,
                    seed=seed,
                    dropout=0.5,
                    learning_rate=0.01,
                    cluster_number=num_labels[ds_name],
                    num_trial=1,
                    layers=[16, 16, 16],
                    membership_closeness=wmc,
                    overlap_strategy=method,
                    adaptation_lambda=lam,
                )

                if method == "no_overlap":
                    kw['clustering_overlap'] = False
                else:
                    kw['clustering_overlap'] = True

                args = SimpleArgs(**kw)
                result = run_single(graph, features, target, args, method)
                result["dataset"] = ds_name
                result["method"] = method
                result["seed"] = seed
                result["wmc"] = wmc
                result["lambda"] = lam
                all_results.append(result)

                print(f"    seed={seed}: F1={result['f1']:.4f}  overlap={result['overlap_ratio']:.2f}x  "
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
        f1_mean=('f1', 'mean'),
        f1_std=('f1', 'std'),
        overlap_mean=('overlap_ratio', 'mean'),
        runtime_mean=('runtime', 'mean'),
    ).round(4)
    print(summary.to_string())

    # Pivot table
    print(f"\n{'='*70}")
    print("  F1 SCORES (mean ± std)")
    print(f"{'='*70}")
    pivot = df.groupby(['dataset', 'method'])['f1'].agg(['mean', 'std']).unstack('method')
    pivot.columns = [f"{m}" for m in pivot.columns.get_level_values(1)]
    print(pivot.round(4).to_string())

    # Save summary
    summary_csv = os.path.join(results_dir, 'full_experiment_summary.csv')
    summary.to_csv(summary_csv)

    print(f"\nRaw results: {raw_csv}")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
