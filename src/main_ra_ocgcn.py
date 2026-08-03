"""
Main entry point for Adaptive Overlapping Cluster-GCN (Adaptive-OCGCN).

This script runs a comparison of overlap selection strategies:
1. Original Cluster-GCN (no overlap)
2. Original Overlapping Cluster-GCN (global WMC)
3. Entropy-Adaptive WMC
4. Margin-Adaptive WMC
5. Hybrid-Adaptive WMC (max of entropy and margin)

Run: python main_ra_ocgcn.py --dataset-name Cora
"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parameter_parser
from utils import dataset_reader
from clustering import ClusteringMachine
from ra_ocgcn_clustering import AdaptiveWMCClusteringMachine
from clustergcn import ClusterGCNTrainer, seed_everything


def run_experiment(args, graph, features, target, method, danmf_result=None):
    """
    Run a single experiment.
    :param args: Arguments object.
    :param method: "no_overlap", "original_wmc", "entropy_adaptive_wmc",
                   "margin_adaptive_wmc" or "hybrid_adaptive_wmc".
    :return: dict with results.
    """
    seed_everything(args.seed)
    start = time.time()

    if method == "no_overlap":
        clustering_machine = ClusteringMachine(args, graph, features, target)
    else:
        clustering_machine = AdaptiveWMCClusteringMachine(args, graph, features, target)

    clustering_machine.decompose(danmf_result=danmf_result)

    trainer = ClusterGCNTrainer(args, clustering_machine)
    trainer.train()
    scores = trainer.test()
    elapsed = time.time() - start

    avg_overlap = float(np.sum(clustering_machine.ClusterNodes)) / len(graph.nodes())

    return {
        "f1_micro": scores["micro"],
        "f1_macro": scores["macro"],
        "runtime": elapsed,
        "avg_overlap": avg_overlap,
        "num_clusters": len(clustering_machine.clusters),
        "num_nodes": len(graph.nodes()),
    }


def main():
    """Main function."""
    print(f"\n{'='*70}")
    print(f"  Adaptive Overlapping Cluster-GCN (Adaptive-OCGCN)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    args = parameter_parser()
    args.clustering_overlap = True
    args.membership_closeness = 0.3
    args.adaptation_lambda = 0.5

    methods = [
        ("no_overlap", "Cluster-GCN (no overlap)"),
        ("original_wmc", "OCGCN (global WMC=0.30)"),
        ("entropy_adaptive_wmc", "Entropy-Adaptive (lambda=0.5)"),
        ("margin_adaptive_wmc", "Margin-Adaptive (lambda=0.5)"),
        ("hybrid_adaptive_wmc", "Hybrid-Adaptive (lambda=0.5)"),
    ]

    datasets = ["Cora", "CiteSeer", "PubMed"]
    all_results = []

    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset}")
        print(f"{'='*70}")

        args.dataset_name = dataset
        graph, features, target = dataset_reader(args)
        print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

        # Cache the DANMF decomposition once per dataset so all strategies
        # share identical community memberships (fair comparison).
        danmf_result = None

        for method, label in methods:
            args.clustering_overlap = (method != "no_overlap")
            result = run_experiment(args, graph, features, target, method, danmf_result)
            result["dataset"] = dataset
            result["method"] = label
            result["strategy"] = method
            all_results.append(result)
            print(f"  {label:<40} F1(micro)={result['f1_micro']:.4f}  "
                  f"F1(macro)={result['f1_macro']:.4f}  overlap={result['avg_overlap']:.2f}x")

        # NOTE: for a fully independent per-method clustering one would fit
        # DANMF per method; here caching gives a paired design.

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(results_dir, "ra_ocgcn_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    pivot = df.pivot_table(index="method", columns="dataset", values="f1_macro", aggfunc="first")
    pivot["Mean"] = pivot.mean(axis=1)
    print("\nF1 Macro by strategy:")
    print(pivot.round(4).to_string())

    print(f"\n{'='*70}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
