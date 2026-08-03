"""
Membership Ambiguity Analysis.

For every node compute:
1. Membership entropy
2. Top1-Top2 margin
3. Number of assigned clusters
4. Final prediction correctness
5. Cluster participation count

Perform correlation analysis, stratified analysis, and dataset comparison.
"""
import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import normalize
from karateclub.community_detection.overlapping import DANMF
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlap_selection.common import compute_membership_entropy, compute_membership_margin
from overlap_selection import create_selector
from clustering import fit_danmf
from experiment_utils import load_dataset as shared_load_dataset, NUM_LABELS


def run_danmf(graph, cluster_count, seed=42):
    result = fit_danmf(graph, cluster_count, seed=seed)
    return result["P"], result["clusters"]


def compute_node_metrics(P, graph, cluster_membership, clusters):
    """Compute per-node metrics."""
    n_nodes = P.shape[0]
    n_clusters = len(clusters)

    metrics = {
        'entropy': np.zeros(n_nodes),
        'margin': np.zeros(n_nodes),
        'ambiguity': np.zeros(n_nodes),
        'n_clusters_assigned': np.zeros(n_nodes, dtype=int),
    }

    for i in range(n_nodes):
        row = P[i]
        metrics['entropy'][i] = compute_membership_entropy(row)
        metrics['margin'][i] = compute_membership_margin(row)
        metrics['ambiguity'][i] = 1.0 - (np.sort(row)[::-1][0] - np.sort(row)[::-1][1]) if len(row) > 1 else 0.0
        metrics['n_clusters_assigned'][i] = len(cluster_membership.get(i, []))

    return metrics


def stratify_and_analyze(metrics, n_bins=3):
    """Stratify nodes by entropy and margin, compare overlap and accuracy."""
    results = []

    for metric_name in ['entropy', 'ambiguity']:
        values = metrics[metric_name]
        percentiles = np.percentile(values, np.linspace(0, 100, n_bins + 1))

        for b in range(n_bins):
            mask = (values >= percentiles[b]) & (values < percentiles[b + 1])
            if b == n_bins - 1:
                mask = (values >= percentiles[b]) & (values <= percentiles[b + 1])

            if mask.sum() == 0:
                continue

            results.append({
                'metric': metric_name,
                'bin': f'{metric_name}_P{int(percentiles[b]*100):02d}-P{int(percentiles[b+1]*100):02d}',
                'n_nodes': int(mask.sum()),
                'mean_clusters_assigned': metrics['n_clusters_assigned'][mask].mean(),
                'mean_entropy': metrics['entropy'][mask].mean(),
                'mean_margin': metrics['ambiguity'][mask].mean(),
            })

    return results


def analyze_dataset(ds_name, ds_root, num_labels):
    """Full analysis for one dataset."""
    print(f"\n{'='*60}")
    print(f"  Analyzing: {ds_name}")
    print(f"{'='*60}")

    graph, features, target = shared_load_dataset(ds_name, ds_root)

    P, clusters = run_danmf(graph, num_labels[ds_name])

    # Original WMC membership
    selector = create_selector('original_wmc', membership_closeness=0.3)
    cluster_list = selector.select_overlap(P, clusters)
    cluster_membership = {i: m for i, m in enumerate(cluster_list)}

    # Compute metrics
    metrics = compute_node_metrics(P, graph, cluster_membership, clusters)

    # Summary stats
    print(f"  Nodes: {len(graph.nodes())}, Clusters: {len(clusters)}")
    print(f"  Entropy: mean={metrics['entropy'].mean():.3f} std={metrics['entropy'].std():.3f}")
    print(f"  Margin ambiguity: mean={metrics['ambiguity'].mean():.3f} std={metrics['ambiguity'].std():.3f}")
    print(f"  Clusters assigned: mean={metrics['n_clusters_assigned'].mean():.2f} "
          f"min={metrics['n_clusters_assigned'].min()} max={metrics['n_clusters_assigned'].max()}")

    # Correlation analysis
    corr_ent_margin, p_ent_margin = spearmanr(metrics['entropy'], metrics['ambiguity'])
    corr_ent_clusters, p_ent_clusters = spearmanr(metrics['entropy'], metrics['n_clusters_assigned'])
    corr_margin_clusters, p_margin_clusters = spearmanr(metrics['ambiguity'], metrics['n_clusters_assigned'])

    print(f"\n  Correlations (Spearman):")
    print(f"    entropy vs ambiguity: r={corr_ent_margin:.3f} (p={p_ent_margin:.2e})")
    print(f"    entropy vs n_clusters: r={corr_ent_clusters:.3f} (p={p_ent_clusters:.2e})")
    print(f"    ambiguity vs n_clusters: r={corr_margin_clusters:.3f} (p={p_margin_clusters:.2e})")

    # Stratified analysis
    strat_results = stratify_and_analyze(metrics)
    print(f"\n  Stratified analysis:")
    for r in strat_results:
        print(f"    {r['bin']}: n={r['n_nodes']}, clusters={r['mean_clusters_assigned']:.2f}")

    return {
        'dataset': ds_name,
        'n_nodes': len(graph.nodes()),
        'n_clusters': len(clusters),
        'entropy_mean': metrics['entropy'].mean(),
        'entropy_std': metrics['entropy'].std(),
        'ambiguity_mean': metrics['ambiguity'].mean(),
        'ambiguity_std': metrics['ambiguity'].std(),
        'n_clusters_mean': metrics['n_clusters_assigned'].mean(),
        'corr_ent_margin': corr_ent_margin,
        'corr_ent_clusters': corr_ent_clusters,
        'corr_margin_clusters': corr_margin_clusters,
    }, metrics, strat_results


def main():
    ds_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    num_labels = NUM_LABELS
    all_stats = []
    all_strat = []

    for ds_name in ['ACM', 'DBLP', 'IMDB', 'Cora', 'CiteSeer', 'PubMed']:
        stats, metrics, strat = analyze_dataset(ds_name, ds_root, num_labels)
        all_stats.append(stats)
        all_strat.extend([{**r, 'dataset': ds_name} for r in strat])

    # Save
    stats_df = pd.DataFrame(all_stats)
    strat_df = pd.DataFrame(all_strat)
    stats_df.to_csv(os.path.join(results_dir, 'ambiguity_stats.csv'), index=False)
    strat_df.to_csv(os.path.join(results_dir, 'ambiguity_stratified.csv'), index=False)

    print(f"\n{'='*60}")
    print("  DATASET COMPARISON")
    print(f"{'='*60}")
    print(stats_df.to_string(index=False))

    print(f"\nStats: {results_dir}/ambiguity_stats.csv")
    print(f"Stratified: {results_dir}/ambiguity_stratified.csv")


if __name__ == "__main__":
    main()
