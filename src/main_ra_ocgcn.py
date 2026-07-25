"""
Main entry point for Role-Aware Overlapping Cluster-GCN (RA-OCGCN).

This script runs experiments comparing:
1. Original Cluster-GCN (no overlap)
2. Original Overlapping Cluster-GCN
3. Role-Aware Overlapping Cluster-GCN (RA-OCGCN)
"""
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import parameter_parser
from utils import dataset_reader
from clustering import ClusteringMachine
from ra_ocgcn_clustering import RAOClusteringMachine
from clustergcn import ClusterGCNTrainer


def run_experiment(args, method="original"):
    """
    Run a single experiment.
    
    Args:
        args: Arguments object
        method: "original" or "ra_ocgcn"
    
    Returns:
        dict with results
    """
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    start = time.time()
    
    # Load dataset
    graph, features, target = dataset_reader(args)
    
    # Create clustering machine
    if method == "ra_ocgcn":
        clustering_machine = RAOClusteringMachine(args, graph, features, target)
    else:
        clustering_machine = ClusteringMachine(args, graph, features, target)
    
    clustering_machine.decompose()
    
    # Train and evaluate
    trainer = ClusterGCNTrainer(args, clustering_machine)
    trainer.train()
    score = trainer.test()
    
    elapsed = time.time() - start
    
    # Calculate overlap statistics
    avg_overlap = np.sum(clustering_machine.ClusterNodes) / len(graph.nodes())
    
    return {
        "accuracy": score,
        "runtime": elapsed,
        "avg_overlap": avg_overlap,
        "num_clusters": len(clustering_machine.clusters),
        "num_nodes": len(graph.nodes()),
    }


def run_ablation_study(args):
    """
    Run ablation study for RA-OCGCN.
    
    Ablations:
    A. Original method (no overlap)
    B. Boundary only
    C. Boundary + uncertainty
    D. Boundary + entropy
    E. Boundary + uncertainty + entropy (full RA-OCGCN)
    """
    results = []
    
    # A. Original method (no overlap)
    print("  A. Original Cluster-GCN (no overlap)...")
    args.clustering_overlap = False
    result = run_experiment(args, method="original")
    result["variant"] = "A_Original"
    result["description"] = "No overlap"
    results.append(result)
    print(f"    Accuracy: {result['accuracy']:.4f}")
    
    # B. Boundary only
    print("  B. Boundary only...")
    args.clustering_overlap = True
    args.alpha = 1.0
    args.beta = 0.0
    args.gamma = 0.0
    result = run_experiment(args, method="ra_ocgcn")
    result["variant"] = "B_Boundary"
    result["description"] = "Boundary only"
    results.append(result)
    print(f"    Accuracy: {result['accuracy']:.4f}")
    
    # C. Boundary + uncertainty
    print("  C. Boundary + uncertainty...")
    args.alpha = 1.0
    args.beta = 0.5
    args.gamma = 0.0
    result = run_experiment(args, method="ra_ocgcn")
    result["variant"] = "C_BoundaryUncertainty"
    result["description"] = "Boundary + uncertainty"
    results.append(result)
    print(f"    Accuracy: {result['accuracy']:.4f}")
    
    # D. Boundary + entropy
    print("  D. Boundary + entropy...")
    args.alpha = 1.0
    args.beta = 0.0
    args.gamma = 0.5
    result = run_experiment(args, method="ra_ocgcn")
    result["variant"] = "D_BoundaryEntropy"
    result["description"] = "Boundary + entropy"
    results.append(result)
    print(f"    Accuracy: {result['accuracy']:.4f}")
    
    # E. Full RA-OCGCN
    print("  E. Full RA-OCGCN...")
    args.alpha = 1.0
    args.beta = 0.5
    args.gamma = 0.5
    result = run_experiment(args, method="ra_ocgcn")
    result["variant"] = "E_FullRAOCGCN"
    result["description"] = "Boundary + uncertainty + entropy"
    results.append(result)
    print(f"    Accuracy: {result['accuracy']:.4f}")
    
    return results


def main():
    """Main function."""
    print(f"\n{'='*70}")
    print(f"  Role-Aware Overlapping Cluster-GCN (RA-OCGCN)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Datasets
    datasets = ["Cora", "CiteSeer", "PubMed"]
    
    all_results = []
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {dataset}")
        print(f"{'='*70}")
        
        # Parse arguments
        sys.argv = [
            'main_ra_ocgcn.py',
            '--dataset-name', dataset,
            '--epochs', '10',
            '--ds-root', './tmp',
            '--clustering-overlap', 'True',
            '--membership-closeness', '0.3',
            '--test-ratio', '0.3',
        ]
        
        args = parameter_parser()
        
        # Add role-aware parameters
        args.alpha = 1.0
        args.beta = 0.5
        args.gamma = 0.5
        args.overlap_strategy = 'adaptive'
        args.overlap_threshold = 0.5
        args.warmup_epochs = 10
        
        # Run ablation study
        print("\nRunning ablation study...")
        ablation_results = run_ablation_study(args)
        
        for result in ablation_results:
            result["dataset"] = dataset
            all_results.append(result)
        
        print(f"\nDataset {dataset} completed.")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(results_dir, "ra_ocgcn_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")
    
    # Display summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    
    pivot = df.pivot_table(
        index='variant',
        columns='dataset',
        values='accuracy',
        aggfunc='first'
    )
    pivot['Mean'] = pivot.mean(axis=1)
    print("\nAccuracy by variant:")
    print(pivot.round(4).to_string())
    
    # Improvement analysis
    print("\n\nImprovement over Original:")
    for dataset in datasets:
        ds_data = df[df['dataset'] == dataset]
        original = ds_data[ds_data['variant'] == 'A_Original']['accuracy'].values[0]
        full_ra = ds_data[ds_data['variant'] == 'E_FullRAOCGCN']['accuracy'].values[0]
        improvement = ((full_ra - original) / original) * 100
        print(f"  {dataset}: {improvement:+.2f}%")
    
    print(f"\n{'='*70}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
