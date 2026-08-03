"""
Comprehensive Membership Ambiguity Analysis for Adaptive-OCGCN.

Tasks:
1. Collect node-level statistics
2. Correlation analysis
3. Stratified ambiguity study
4. Dataset-level comparison
5. Visualizations
"""
import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import normalize
from karateclub.community_detection.overlapping import DANMF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlap_selection.common import compute_membership_entropy, compute_membership_margin
from overlap_selection import create_selector
from clustering import fit_danmf
from experiment_utils import load_dataset as shared_load_dataset, NUM_LABELS


# ============================================================
# TASK 1: Collect Node-Level Statistics
# ============================================================

def run_danmf(graph, cluster_count, seed=42):
    result = fit_danmf(graph, cluster_count, seed=seed)
    return result["P"], result["clusters"]


def load_dataset(ds_name, ds_root, num_labels):
    """Load dataset and return graph, features, target (shared loader)."""
    return shared_load_dataset(ds_name, ds_root)


def compute_node_statistics(ds_name, graph, P, clusters, cluster_membership, logits=None):
    """Compute per-node statistics for all nodes."""
    n_nodes = P.shape[0]
    n_clusters = len(clusters)

    records = []
    for i in range(n_nodes):
        row = P[i]

        # 1. Membership entropy
        entropy = compute_membership_entropy(row)

        # 2. Top1-Top2 margin
        sorted_probs = np.sort(row)[::-1]
        p1 = sorted_probs[0]
        p2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin = p1 - p2

        # 3. Ambiguity score
        ambiguity = 1.0 - margin

        # 4. Number of assigned clusters
        assigned_clusters = len(cluster_membership.get(i, []))

        # 5. Whether overlap node
        is_overlap = 1 if assigned_clusters > 1 else 0

        # 6. Prediction correctness (using argmax of membership as proxy)
        predicted_cluster = np.argmax(row)
        true_cluster = None  # Would need labels for true classification

        # 7. Boundary-node indicator
        node_clusters = set(cluster_membership.get(i, []))
        neighbor_clusters = set()
        for neighbor in graph.neighbors(i):
            if neighbor in cluster_membership:
                neighbor_clusters.update(cluster_membership[neighbor])
        diff_clusters = neighbor_clusters - node_clusters
        boundary_score = len(diff_clusters) / max(len(neighbor_clusters), 1)
        is_boundary = 1 if boundary_score > 0 else 0

        records.append({
            'dataset': ds_name,
            'node_id': i,
            'entropy': entropy,
            'margin': margin,
            'ambiguity': ambiguity,
            'n_clusters_assigned': assigned_clusters,
            'is_overlap': is_overlap,
            'boundary_score': boundary_score,
            'is_boundary': is_boundary,
        })

    return pd.DataFrame(records)


# ============================================================
# TASK 2: Correlation Analysis
# ============================================================

def correlation_analysis(df):
    """Compute Pearson and Spearman correlations."""
    results = []

    pairs = [
        ('entropy', 'is_overlap'),
        ('entropy', 'n_clusters_assigned'),
        ('ambiguity', 'is_overlap'),
        ('ambiguity', 'n_clusters_assigned'),
        ('entropy', 'ambiguity'),
        ('margin', 'ambiguity'),
    ]

    for var1, var2 in pairs:
        if var1 not in df.columns or var2 not in df.columns:
            continue
        pearson_r, pearson_p = pearsonr(df[var1], df[var2])
        spearman_r, spearman_p = spearmanr(df[var1], df[var2])
        results.append({
            'variable_1': var1,
            'variable_2': var2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
        })

    return pd.DataFrame(results)


# ============================================================
# TASK 3: Stratified Ambiguity Study
# ============================================================

def stratified_analysis(df, n_terciles=3):
    """Split nodes into terciles by ambiguity and compare."""
    results = []

    for metric in ['entropy', 'ambiguity']:
        values = df[metric].values
        terciles = np.percentile(values, np.linspace(0, 100, n_terciles + 1))

        for t in range(n_terciles):
            mask = (values >= terciles[t]) & (values < terciles[t + 1])
            if t == n_terciles - 1:
                mask = (values >= terciles[t]) & (values <= terciles[t + 1])

            subset = df[mask]
            if len(subset) == 0:
                continue

            results.append({
                'metric': metric,
                'tercile': f'T{t+1}' if t < n_terciles - 1 else f'T{n_terciles}',
                'range_low': terciles[t],
                'range_high': terciles[t + 1],
                'n_nodes': len(subset),
                'overlap_ratio': subset['is_overlap'].mean(),
                'mean_clusters': subset['n_clusters_assigned'].mean(),
                'mean_boundary': subset['boundary_score'].mean(),
                'mean_entropy': subset['entropy'].mean(),
                'mean_ambiguity': subset['ambiguity'].mean(),
            })

    return pd.DataFrame(results)


# ============================================================
# TASK 4: Dataset-Level Comparison
# ============================================================

def dataset_comparison(all_stats):
    """Compare datasets at aggregate level."""
    results = []
    for ds_name, df in all_stats.items():
        results.append({
            'dataset': ds_name,
            'n_nodes': len(df),
            'entropy_mean': df['entropy'].mean(),
            'entropy_std': df['entropy'].std(),
            'entropy_var': df['entropy'].var(),
            'ambiguity_mean': df['ambiguity'].mean(),
            'ambiguity_std': df['ambiguity'].std(),
            'ambiguity_var': df['ambiguity'].var(),
            'overlap_ratio': df['is_overlap'].mean(),
            'mean_clusters': df['n_clusters_assigned'].mean(),
            'boundary_ratio': df['is_boundary'].mean(),
        })
    return pd.DataFrame(results)


def _savefig(fig, path):
    """
    Save a figure, robust to Windows file-lock errors.

    Overwriting an existing PNG on Windows can raise OSError [Errno 22]
    when the file is momentarily held (thumbnail cache, viewers, etc.).
    We remove any stale file first and retry once.
    """
    for attempt in range(2):
        try:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            fig.savefig(path, dpi=150)
            return
        except OSError:
            import time
            time.sleep(0.5)
    # Final fallback: save without touching the existing file name.
    fig.savefig(path, dpi=150)


def _safe_terciles(series):
    """
    Split a series into up to three terciles, robust to duplicate bin edges.

    pd.qcut(..., duplicates='drop') collapses bins when many values are
    identical (common for entropy, where many nodes have entropy ~0), which
    raises ValueError when a fixed label list is supplied. We instead bin
    without labels and then map integer codes to 'Low'/'Med'/'High'.
    """
    labels = ['Low', 'Med', 'High']
    try:
        codes = pd.qcut(series, 3, labels=False, duplicates='drop')
    except ValueError:
        return pd.Series('Low', index=series.index)
    n_bins = int(codes.max()) + 1 if len(codes) else 1
    mapping = {i: labels[i] for i in range(min(n_bins, 3))}
    return codes.map(mapping)


# ============================================================
# TASK 5: Visualizations
# ============================================================

def create_visualizations(all_stats, results_dir):
    """Create all plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping visualizations")
        return

    plot_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    datasets = list(all_stats.keys())
    n_ds = len(datasets)

    # 1. Entropy histograms
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, ds_name in enumerate(datasets):
        ax = axes[idx // 3, idx % 3]
        df = all_stats[ds_name]
        ax.hist(df['entropy'], bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(f'{ds_name}')
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Count')
    plt.suptitle('Membership Entropy Distributions', fontsize=14)
    plt.tight_layout()
    _savefig(plt.gcf(), os.path.join(plot_dir, 'entropy_histograms.png'))
    plt.close()

    # 2. Ambiguity histograms
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, ds_name in enumerate(datasets):
        ax = axes[idx // 3, idx % 3]
        df = all_stats[ds_name]
        ax.hist(df['ambiguity'], bins=30, alpha=0.7, edgecolor='black', color='orange')
        ax.set_title(f'{ds_name}')
        ax.set_xlabel('Ambiguity (1 - margin)')
        ax.set_ylabel('Count')
    plt.suptitle('Ambiguity Score Distributions', fontsize=14)
    plt.tight_layout()
    _savefig(plt.gcf(), os.path.join(plot_dir, 'ambiguity_histograms.png'))
    plt.close()

    # 3. Overlap ratio vs entropy (box plots by tercile)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, ds_name in enumerate(datasets):
        ax = axes[idx // 3, idx % 3]
        df = all_stats[ds_name].copy()
        df['entropy_tercile'] = _safe_terciles(df['entropy'])
        overlap_by_tercile = df.groupby('entropy_tercile')['is_overlap'].mean()
        overlap_by_tercile.plot(kind='bar', ax=ax, color=['#2196F3', '#FFC107', '#F44336'])
        ax.set_title(f'{ds_name}')
        ax.set_xlabel('Entropy Tercile')
        ax.set_ylabel('Overlap Ratio')
        ax.set_ylim(0, 1)
    plt.suptitle('Overlap Ratio by Entropy Tercile', fontsize=14)
    plt.tight_layout()
    _savefig(plt.gcf(), os.path.join(plot_dir, 'overlap_vs_entropy.png'))
    plt.close()

    # 4. Scatter: entropy vs n_clusters
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, ds_name in enumerate(datasets):
        ax = axes[idx // 3, idx % 3]
        df = all_stats[ds_name]
        ax.scatter(df['entropy'], df['n_clusters_assigned'], alpha=0.1, s=5)
        ax.set_title(f'{ds_name}')
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Clusters Assigned')
    plt.suptitle('Entropy vs Clusters Assigned', fontsize=14)
    plt.tight_layout()
    _savefig(plt.gcf(), os.path.join(plot_dir, 'entropy_vs_clusters.png'))
    plt.close()

    # 5. Scatter: ambiguity vs overlap
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, ds_name in enumerate(datasets):
        ax = axes[idx // 3, idx % 3]
        df = all_stats[ds_name]
        ax.scatter(df['ambiguity'], df['is_overlap'], alpha=0.1, s=5, color='green')
        ax.set_title(f'{ds_name}')
        ax.set_xlabel('Ambiguity')
        ax.set_ylabel('Is Overlap')
    plt.suptitle('Ambiguity vs Overlap Assignment', fontsize=14)
    plt.tight_layout()
    _savefig(plt.gcf(), os.path.join(plot_dir, 'ambiguity_vs_overlap.png'))
    plt.close()

    # 6. Dataset comparison bar chart
    ds_comp = dataset_comparison(all_stats)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ds_comp.set_index('dataset')[['entropy_mean', 'ambiguity_mean', 'overlap_ratio']].plot(
        kind='bar', ax=axes[0], rot=45
    )
    axes[0].set_title('Mean Scores by Dataset')
    axes[0].set_ylabel('Score')

    ds_comp.set_index('dataset')[['entropy_var', 'ambiguity_var']].plot(
        kind='bar', ax=axes[1], rot=45
    )
    axes[1].set_title('Variance by Dataset')
    axes[1].set_ylabel('Variance')

    ds_comp.set_index('dataset')[['mean_clusters', 'boundary_ratio']].plot(
        kind='bar', ax=axes[2], rot=45
    )
    axes[2].set_title('Cluster Stats by Dataset')
    axes[2].set_ylabel('Value')

    plt.suptitle('Dataset Comparison', fontsize=14)
    plt.tight_layout()
    _savefig(plt.gcf(), os.path.join(plot_dir, 'dataset_comparison.png'))
    plt.close()

    print(f"  Plots saved to: {plot_dir}/")


# ============================================================
# MAIN
# ============================================================

def main():
    ds_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    num_labels = NUM_LABELS
    datasets = ['ACM', 'DBLP', 'IMDB', 'Cora', 'CiteSeer', 'PubMed']

    all_stats = {}
    all_correlations = []
    all_stratified = []

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"  Processing: {ds_name}")
        print(f"{'='*60}")

        graph, features, target = load_dataset(ds_name, ds_root, num_labels)
        print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

        P, clusters = run_danmf(graph, num_labels[ds_name])
        print(f"  Clusters: {len(clusters)}")

        # Original WMC membership
        selector = create_selector('original_wmc', membership_closeness=0.3)
        cluster_list = selector.select_overlap(P, clusters)
        cluster_membership = {i: m for i, m in enumerate(cluster_list)}

        # Compute statistics
        stats_df = compute_node_statistics(ds_name, graph, P, clusters, cluster_membership)
        all_stats[ds_name] = stats_df

        # Correlation analysis
        corr_df = correlation_analysis(stats_df)
        all_correlations.append(corr_df)

        # Stratified analysis
        strat_df = stratified_analysis(stats_df)
        all_stratified.append(strat_df)

        # Print summary
        print(f"  Entropy: mean={stats_df['entropy'].mean():.3f} std={stats_df['entropy'].std():.3f}")
        print(f"  Ambiguity: mean={stats_df['ambiguity'].mean():.3f} std={stats_df['ambiguity'].std():.3f}")
        print(f"  Overlap ratio: {stats_df['is_overlap'].mean():.3f}")
        print(f"  Mean clusters: {stats_df['n_clusters_assigned'].mean():.2f}")

    # Save node-level statistics
    all_stats_df = pd.concat(all_stats.values(), ignore_index=True)
    all_stats_df.to_csv(os.path.join(results_dir, 'node_statistics.csv'), index=False)
    print(f"\nNode statistics saved to: {results_dir}/node_statistics.csv")

    # Save correlation results
    corr_summary = pd.concat(all_correlations, ignore_index=True)
    corr_summary.to_csv(os.path.join(results_dir, 'correlation_analysis.csv'), index=False)
    print(f"Correlation analysis saved to: {results_dir}/correlation_analysis.csv")

    # Print correlation tables
    print(f"\n{'='*60}")
    print("  CORRELATION ANALYSIS")
    print(f"{'='*60}")
    for ds_name in datasets:
        ds_corr = corr_summary.copy()
        print(f"\n  {ds_name}:")
        print(f"    {'Pair':<30} {'Pearson r':>10} {'Spearman r':>12} {'p-value':>12}")
        print(f"    {'-'*64}")
        for _, row in corr_summary.iterrows():
            print(f"    {row['variable_1']+' vs '+row['variable_2']:<30} "
                  f"{row['pearson_r']:>10.3f} {row['spearman_r']:>12.3f} {row['pearson_p']:>12.2e}")

    # Save stratified results
    strat_summary = pd.concat(all_stratified, ignore_index=True)
    strat_summary.to_csv(os.path.join(results_dir, 'stratified_analysis.csv'), index=False)

    # Print stratified tables
    print(f"\n{'='*60}")
    print("  STRATIFIED ANALYSIS")
    print(f"{'='*60}")
    for ds_name in datasets:
        ds_strat = strat_summary[strat_summary.index.isin(
            strat_summary.loc[strat_summary['metric'] == 'entropy'].index
        )]
        print(f"\n  {ds_name} (by entropy terciles):")
        ds_data = all_stats[ds_name].copy()
        ds_data['tercile'] = _safe_terciles(ds_data['entropy'])
        for t in ['Low', 'Med', 'High']:
            subset = ds_data[ds_data['tercile'] == t]
            print(f"    {t}: n={len(subset)}, overlap={subset['is_overlap'].mean():.3f}, "
                  f"clusters={subset['n_clusters_assigned'].mean():.2f}")

    # Dataset comparison
    ds_comp = dataset_comparison(all_stats)
    ds_comp.to_csv(os.path.join(results_dir, 'dataset_comparison.csv'), index=False)

    print(f"\n{'='*60}")
    print("  DATASET COMPARISON")
    print(f"{'='*60}")
    print(ds_comp.to_string(index=False))

    # Visualizations
    print(f"\n{'='*60}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    create_visualizations(all_stats, results_dir)

    print(f"\n{'='*60}")
    print("  ALL RESULTS SAVED")
    print(f"{'='*60}")
    print(f"  Node statistics: {results_dir}/node_statistics.csv")
    print(f"  Correlations:    {results_dir}/correlation_analysis.csv")
    print(f"  Stratified:      {results_dir}/stratified_analysis.csv")
    print(f"  Dataset compare:  {results_dir}/dataset_comparison.csv")
    print(f"  Plots:           {results_dir}/plots/")


if __name__ == "__main__":
    main()
