#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
t-SNE visualisation coloured by DANMF cluster (not class label).

Core hypothesis test: high-entropy (ambiguous) nodes should lie on the
boundaries between DANMF clusters, while low-entropy nodes sit deep inside
a single cluster.

Usage:
    python generate_tsne_danmf_clusters.py --dataset CiteSeer
    python generate_tsne_danmf_clusters.py --dataset IMDB
    python generate_tsne_danmf_clusters.py --dataset CiteSeer IMDB
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_utils import SimpleArgs, fit_danmf_cached, load_dataset, NUM_LABELS
from ra_ocgcn_clustering import AdaptiveWMCClusteringMachine
from clustergcn import ClusterGCNTrainer, seed_everything
from overlap_selection.common import compute_membership_entropy


# ── Embedding extraction ─────────────────────────────────────────────

def extract_embeddings(model, clustering_machine, device):
    """Extract second-to-last-layer embeddings, averaged across clusters."""
    model.eval()
    all_emb, all_nid = [], []
    with torch.no_grad():
        for cluster in clustering_machine.clusters:
            edges = clustering_machine.sg_edges[cluster].to(device)
            feats = clustering_machine.sg_features[cluster].to(device)
            nodes = clustering_machine.sg_nodes[cluster]
            h = feats
            for i in range(len(model.layers) - 1):
                h = model.layers[i](h, edges)
                h = torch.nn.functional.relu(h)
                if i == len(model.layers) - 2:
                    emb = h.clone()
            nodes_np = nodes.cpu().numpy()
            emb_np = emb.cpu().numpy()
            for idx, nid in enumerate(nodes_np):
                all_emb.append(emb_np[idx])
                all_nid.append(nid)
    all_emb = np.array(all_emb)
    all_nid = np.array(all_nid)
    unique_nid = np.unique(all_nid)
    unique_emb = np.zeros((len(unique_nid), all_emb.shape[1]))
    for i, nid in enumerate(unique_nid):
        mask = all_nid == nid
        unique_emb[i] = np.mean(all_emb[mask], axis=0)
    return unique_emb, unique_nid


def compute_entropy(P, node_ids):
    ent = np.zeros(len(node_ids))
    for i, nid in enumerate(node_ids):
        ent[i] = compute_membership_entropy(P[int(nid)])
    return ent


def dominant_cluster(P, node_ids):
    """Return argmax cluster for each node."""
    return np.array([int(np.argmax(P[int(nid)])) for nid in node_ids])


def cluster_centroids_2d(coords_2d, labels):
    """Compute mean 2D position per cluster."""
    centroids = {}
    for c in np.unique(labels):
        m = labels == c
        centroids[c] = coords_2d[m].mean(axis=0)
    return centroids


def dist_to_centroid_2d(coords_2d, labels, centroids):
    """Euclidean distance of each node to its cluster centroid in 2D t-SNE space."""
    d = np.zeros(len(labels))
    for i, c in enumerate(labels):
        d[i] = np.linalg.norm(coords_2d[i] - centroids[c])
    return d


# ── Colour palette ───────────────────────────────────────────────────

def cluster_colours(n_clusters):
    """Distinct palette for up to ~15 clusters."""
    base = plt.cm.tab20(np.linspace(0, 1, 20))
    return [base[i % 20] for i in range(n_clusters)]


# ── Main ─────────────────────────────────────────────────────────────

def run(dataset_name, ds_root, output_dir, seed=42, epochs=20):
    print(f"\n{'='*60}")
    print(f"  DANMF-cluster t-SNE: {dataset_name}")
    print(f"{'='*60}")

    seed_everything(seed)
    graph, features, target = load_dataset(dataset_name, ds_root)
    print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    # Args matching the paper experiments
    n_cls = NUM_LABELS[dataset_name]
    args = SimpleArgs(
        dataset_name=dataset_name, ds_root=ds_root,
        clustering_method='danmf', clustering_overlap=True,
        membership_closeness=0.3, adaptation_lambda=0.5,
        overlap_strategy='original_wmc',
        epochs=epochs, test_ratio=0.3, seed=seed, dropout=0.5,
        learning_rate=0.01, cluster_number=n_cls,
        num_trial=1, layers=[16, 16, 16],
    )

    # DANMF
    print("  Fitting DANMF...")
    danmf_result = fit_danmf_cached(graph, args, seed)
    P = danmf_result['P']
    n_danmf = P.shape[1]
    print(f"  DANMF clusters: {n_danmf}, membership matrix: {P.shape}")

    # Train baseline GCN
    print("  Training baseline GCN (fixed WMC=0.30)...")
    cm = AdaptiveWMCClusteringMachine(args, graph, features, target)
    cm.decompose(danmf_result=danmf_result)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = ClusterGCNTrainer(args, cm)
    trainer.train()
    scores = trainer.test()
    print(f"  F1 micro={scores['micro']:.4f}  macro={scores['macro']:.4f}")

    # Extract embeddings + entropy + dominant cluster
    emb, nid = extract_embeddings(trainer.model, cm, device)
    ent = compute_entropy(P, nid)
    dc = dominant_cluster(P, nid)
    print(f"  Embeddings: {emb.shape}")

    # t-SNE
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(emb) - 1),
                learning_rate='auto', n_iter=1000, random_state=seed, init='pca')
    coords = tsne.fit_transform(emb)

    # Centroid distances (in t-SNE space)
    cents = cluster_centroids_2d(coords, dc)
    dists = dist_to_centroid_2d(coords, dc, cents)

    # Pearson correlation: entropy vs distance-to-centroid
    corr = np.corrcoef(ent, dists)[0, 1]

    # ── Stats ──
    mean_h = ent.mean()
    var_h = ent.var()
    frac_amb = (ent > 0.3).mean()
    print(f"\n  Mean entropy:      {mean_h:.4f}")
    print(f"  Entropy variance:  {var_h:.4f}")
    print(f"  Ambiguous (H>0.3): {frac_amb*100:.1f}%")
    print(f"  Pearson r(H, dist): {corr:.4f}")

    # ── Plot ──
    palette = cluster_colours(n_danmf)
    min_sz, max_sz = 6, 90
    sizes = min_sz + (max_sz - min_sz) * ent

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by size so large (high-entropy) nodes draw on top
    order = np.argsort(sizes)
    for c in range(n_danmf):
        m = (dc == c) & np.isin(np.arange(len(dc)), order)
        # draw small-entropy first (lower zorder), high-entropy later
        pass

    # Simpler: scatter per cluster, all at once, then overlay high-entropy
    for c in range(n_danmf):
        m = dc == c
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=[palette[c]], s=sizes[m], alpha=0.65,
                   edgecolors='gray', linewidths=0.25,
                   zorder=2)

    # Re-draw top-10% entropy nodes on top with thicker border
    top_threshold = np.percentile(ent, 90)
    top_mask = ent >= top_threshold
    for c in range(n_danmf):
        m = top_mask & (dc == c)
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1],
                       c=[palette[c]], s=sizes[m], alpha=0.9,
                       edgecolors='black', linewidths=0.6,
                       zorder=3)

    # Draw cluster centroids
    for c, cent in cents.items():
        ax.plot(cent[0], cent[1], 'x', color='black', markersize=10,
                markeredgewidth=2, zorder=4)

    # Cluster legend
    handles = [Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=palette[c], markersize=8,
                       label=f'Cluster {c}')
               for c in range(n_danmf)]
    # Entropy size legend
    handles.append(Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='lightgray', markersize=5,
                           label='Low entropy'))
    handles.append(Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='lightgray', markersize=12,
                           label='High entropy'))
    handles.append(Line2D([0], [0], marker='x', color='black',
                           markersize=8, markeredgewidth=2,
                           linestyle='None', label='Centroid'))

    ax.legend(handles=handles, loc='upper right', framealpha=0.9,
              fontsize=7.5, title='DANMF Cluster / Entropy', title_fontsize=8,
              ncol=2 if n_danmf > 8 else 1)

    ax.set_xlabel('t-SNE dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE dimension 2', fontsize=10)
    ax.set_title(
        f't-SNE of GCN Embeddings Coloured by DANMF Cluster -- {dataset_name}',
        fontsize=11, fontweight='bold')

    # Stats box
    stats = (f"Mean H: {mean_h:.3f}   Var(H): {var_h:.4f}\n"
             f"Ambiguous (H>0.3): {frac_amb*100:.1f}%\n"
             f"Pearson r(H, dist-to-centroid): {corr:+.3f}\n"
             f"Nodes: {len(nid)}   DANMF clusters: {n_danmf}\n"
             f"F1 micro: {scores['micro']:.4f}")
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=8,
            va='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.9))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir,
                            f'tsne_danmf_clusters_{dataset_name.lower()}.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    return pdf_path


def main():
    parser = argparse.ArgumentParser(
        description='t-SNE coloured by DANMF cluster (boundary hypothesis)')
    parser.add_argument('--dataset', nargs='+', default=['CiteSeer'])
    parser.add_argument('--ds-root', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds_root = args.ds_root or os.path.join(root, 'tmp')
    out_dir = args.output_dir or os.path.join(root, 'figures')

    for ds in args.dataset:
        run(ds, ds_root, out_dir, args.seed, args.epochs)
    print("\nDone.")


if __name__ == '__main__':
    main()
