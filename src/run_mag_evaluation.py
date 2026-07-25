"""
OGBN-MAG Evaluation for Adaptive-OCGCN.
Uses 50K node subsample for tractability.
"""
import sys
import os
import time
import gzip
import random
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clustering import ClusteringMachine
from ra_ocgcn_clustering import AdaptiveWMCClusteringMachine
from clustergcn import ClusterGCNTrainer
from overlap_selection import create_selector
from overlap_selection.common import compute_membership_entropy, compute_membership_margin
from scipy.stats import spearmanr


class SimpleArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def load_ogbn_mag_subsample(ds_root, sample_size=50000):
    """Load OGBN-MAG and create a subsampled paper-paper citation graph."""
    raw_dir = os.path.join(ds_root, 'OGB_MAG', 'mag', 'raw')

    # Load labels
    labels_df = pd.read_csv(os.path.join(raw_dir, 'node-label', 'paper', 'node-label.csv.gz'),
                            compression='gzip', header=None)
    all_labels = labels_df.values.flatten()

    # Load features
    features_df = pd.read_csv(os.path.join(raw_dir, 'node-feat', 'paper', 'node-feat.csv.gz'),
                              compression='gzip', header=None, dtype=np.float32)
    all_features = features_df.values

    # Load citation edges
    edges_df = pd.read_csv(os.path.join(raw_dir, 'relations', 'paper___cites___paper', 'edge.csv.gz'),
                           compression='gzip', header=None)
    all_edges = edges_df.values

    # Build full graph to find nodes with edges
    print("  Building full citation graph...")
    full_G = nx.DiGraph()
    full_G.add_nodes_from(range(len(all_labels)))
    valid = (all_edges[:, 0] < len(all_labels)) & (all_edges[:, 1] < len(all_labels))
    full_G.add_edges_from(all_edges[valid].tolist())

    # Find nodes with at least 1 citation (in or out)
    nodes_with_edges = set()
    for u, v in full_G.edges():
        nodes_with_edges.add(u)
        nodes_with_edges.add(v)

    print(f"  Total papers: {len(all_labels)}")
    print(f"  Papers with citations: {len(nodes_with_edges)}")

    # Subsample
    random.seed(42)
    sample_nodes = sorted(random.sample(list(nodes_with_edges), min(sample_size, len(nodes_with_edges))))

    # Extract subgraph
    sub_G = full_G.subgraph(sample_nodes).copy()
    sub_labels = all_labels[sample_nodes]
    sub_features = all_features[sample_nodes]

    # Relabel to 0..N-1
    mapping = {old: new for new, old in enumerate(sub_G.nodes())}
    sub_G = nx.relabel_nodes(sub_G, mapping)

    # Remove isolates
    isolates = list(nx.isolates(sub_G))
    if isolates:
        sub_G.remove_nodes_from(isolates)
        keep = [i for i in range(len(sample_nodes)) if i not in set(isolates)]
        sub_labels = sub_labels[keep]
        sub_features = sub_features[keep]
        mapping2 = {old: new for new, old in enumerate(sub_G.nodes())}
        sub_G = nx.relabel_nodes(sub_G, mapping2)

    # Convert to undirected
    sub_G = sub_G.to_undirected()

    print(f"  Subsample: {sub_G.number_of_nodes()} nodes, {sub_G.number_of_edges()} edges")
    print(f"  Classes: {len(np.unique(sub_labels))}")
    print(f"  Density: {nx.density(sub_G):.6f}")
    print(f"  Components: {nx.number_connected_components(sub_G)}")

    return sub_G, sub_features, sub_labels


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


def ambiguity_analysis(graph, P, clusters, cluster_membership):
    """Run ambiguity analysis on the subsample."""
    n_nodes = P.shape[0]
    entropies = np.array([compute_membership_entropy(P[i]) for i in range(n_nodes)])
    margins = np.array([compute_membership_margin(P[i]) for i in range(n_nodes)])
    n_clusters_assigned = np.array([len(cluster_membership.get(i, [])) for i in range(n_nodes)])

    corr_ent_clusters, _ = spearmanr(entropies, n_clusters_assigned)
    corr_margin_clusters, _ = spearmanr(margins, n_clusters_assigned)

    return {
        'entropy_mean': entropies.mean(),
        'entropy_std': entropies.std(),
        'ambiguity_mean': margins.mean(),
        'ambiguity_std': margins.std(),
        'corr_ent_clusters': corr_ent_clusters,
        'corr_margin_clusters': corr_margin_clusters,
        'overlap_ratio': (n_clusters_assigned > 1).mean(),
    }


def main():
    ds_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    seeds = list(range(5))
    num_classes = 349
    sample_size = 50000

    # Load dataset
    print("=" * 70)
    print("  OGBN-MAG EVALUATION")
    print("=" * 70)
    graph, features, target = load_ogbn_mag_subsample(ds_root, sample_size)
    print(f"  Final: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Methods
    methods = [
        ("no_overlap",     "no_overlap",           {"clustering_overlap": False}),
        ("WMC_0.10",       "original_wmc",         {"clustering_overlap": True, "membership_closeness": 0.10}),
        ("WMC_0.30",       "original_wmc",         {"clustering_overlap": True, "membership_closeness": 0.30}),
        ("entropy_l050",   "entropy_adaptive_wmc",  {"clustering_overlap": True, "membership_closeness": 0.30, "adaptation_lambda": 0.5}),
        ("margin_l050",    "margin_adaptive_wmc",   {"clustering_overlap": True, "membership_closeness": 0.30, "adaptation_lambda": 0.5}),
    ]

    all_results = []
    ambiguity_results = []

    for method_label, strategy, extra_kw in methods:
        print(f"\n{'='*70}")
        print(f"  Method: {method_label}")
        print(f"{'='*70}")

        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            kw = dict(
                dataset_name='OGB_MAG', ds_root=ds_root, clustering_method='danmf',
                epochs=10, test_ratio=0.3, seed=seed, dropout=0.5,
                learning_rate=0.01, cluster_number=num_classes, num_trial=1,
                layers=[16, 16, 16], overlap_strategy=strategy, **extra_kw,
            )
            args = SimpleArgs(**kw)
            result = run_single(graph, features, target, args, strategy)
            result.update({"dataset": "OGB_MAG", "method": method_label, "seed": seed})
            all_results.append(result)
            print(f"F1={result['f1']:.4f} overlap={result['overlap_ratio']:.2f}x time={result['runtime']:.1f}s")

        # Ambiguity analysis for first seed
        if seeds:
            kw0 = dict(
                dataset_name='OGB_MAG', ds_root=ds_root, clustering_method='danmf',
                epochs=10, test_ratio=0.3, seed=seeds[0], dropout=0.5,
                learning_rate=0.01, cluster_number=num_classes, num_trial=1,
                layers=[16, 16, 16], overlap_strategy=strategy, **extra_kw,
            )
            args0 = SimpleArgs(**kw0)
            torch.manual_seed(seeds[0])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seeds[0])
            from ra_ocgcn_clustering import AdaptiveWMCClusteringMachine as RAO
            cm = RAO(args0, graph, features, target)
            cm.decompose()
            amb = ambiguity_analysis(graph, cm.membership_matrix, cm.clusters, cm.cluster_membership)
            amb['method'] = method_label
            ambiguity_results.append(amb)

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(results_dir, 'mag_raw.csv'), index=False)

    amb_df = pd.DataFrame(ambiguity_results)
    amb_df.to_csv(os.path.join(results_dir, 'mag_ambiguity.csv'), index=False)

    # Summary
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    summary = df.groupby('method').agg(
        f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
        overlap_mean=('overlap_ratio', 'mean'), runtime_mean=('runtime', 'mean'),
    ).round(4)
    print(summary.to_string())
    summary.to_csv(os.path.join(results_dir, 'mag_summary.csv'))

    print(f"\n{'='*70}")
    print("  AMBIGUITY ANALYSIS")
    print(f"{'='*70}")
    print(amb_df[['method', 'entropy_mean', 'ambiguity_mean', 'corr_ent_clusters', 'overlap_ratio']].round(4).to_string())

    print(f"\nResults saved to: {results_dir}/")


if __name__ == "__main__":
    main()
