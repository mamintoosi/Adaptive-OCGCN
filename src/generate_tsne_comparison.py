#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate comparative t-SNE visualizations: Baseline vs Adaptive-OCGCN.

For each dataset, trains two GCN models:
  1. Baseline: OCGCN with fixed WMC=0.30 (no adaptation)
  2. Proposed: Adaptive-OCGCN with Margin-Adaptive WMC (lambda=0.50)

Produces a side-by-side figure showing how the adaptive method
improves embedding quality for ambiguous (high-entropy) nodes.

Usage:
    python generate_tsne_comparison.py --dataset CiteSeer
    python generate_tsne_comparison.py --dataset IMDB
    python generate_tsne_comparison.py --dataset CiteSeer IMDB
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
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_utils import (
    SimpleArgs, fit_danmf_cached, load_dataset, NUM_LABELS,
)
from ra_ocgcn_clustering import AdaptiveWMCClusteringMachine
from clustergcn import ClusterGCNTrainer, seed_everything
from overlap_selection.common import compute_membership_entropy


# ── helpers ──────────────────────────────────────────────────────────

def extract_embeddings(model, clustering_machine, device):
    """
    Extract node embeddings from the second-to-last GCN layer.

    Averages embeddings for nodes appearing in multiple overlapping clusters.
    Returns (unique_embeddings, unique_node_ids, unique_labels).
    """
    model.eval()
    all_emb, all_nid, all_tgt = [], [], []

    with torch.no_grad():
        for cluster in clustering_machine.clusters:
            edges  = clustering_machine.sg_edges[cluster].to(device)
            feats  = clustering_machine.sg_features[cluster].to(device)
            target = clustering_machine.sg_targets[cluster].to(device).squeeze()
            nodes  = clustering_machine.sg_nodes[cluster]

            h = feats
            for i in range(len(model.layers) - 1):
                h = model.layers[i](h, edges)
                h = torch.nn.functional.relu(h)
                if i == len(model.layers) - 2:
                    emb = h.clone()

            nodes_np  = nodes.cpu().numpy()
            emb_np    = emb.cpu().numpy()
            target_np = target.cpu().numpy()
            for idx, nid in enumerate(nodes_np):
                all_emb.append(emb_np[idx])
                all_nid.append(nid)
                all_tgt.append(target_np[idx])

    all_emb  = np.array(all_emb)
    all_nid  = np.array(all_nid)
    all_tgt  = np.array(all_tgt)

    unique_nid, inverse = np.unique(all_nid, return_inverse=True)
    unique_emb  = np.zeros((len(unique_nid), all_emb.shape[1]))
    unique_tgt  = np.zeros(len(unique_nid), dtype=int)
    for i, nid in enumerate(unique_nid):
        mask = (all_nid == nid)
        unique_emb[i] = np.mean(all_emb[mask], axis=0)
        unique_tgt[i] = all_tgt[mask][0]

    return unique_emb, unique_nid, unique_tgt


def compute_entropy(P, node_ids):
    """Compute normalised membership entropy for selected nodes."""
    ent = np.zeros(len(node_ids))
    for i, nid in enumerate(node_ids):
        ent[i] = compute_membership_entropy(P[int(nid)])
    return ent


def train_one(dataset_name, ds_root, strategy, wmc, lam, epochs,
              danmf_result, seed):
    """
    Train a single GCN run with a given overlap strategy.

    Returns (model, clustering_machine, scores, device).
    """
    seed_everything(seed)

    graph, features, target = load_dataset(dataset_name, ds_root)

    args = SimpleArgs(
        dataset_name=dataset_name,
        ds_root=ds_root,
        clustering_method='danmf',
        clustering_overlap=(strategy != 'no_overlap'),
        membership_closeness=wmc,
        adaptation_lambda=lam,
        overlap_strategy=strategy,
        epochs=epochs,
        test_ratio=0.3,
        seed=seed,
        dropout=0.5,
        learning_rate=0.01,
        cluster_number=NUM_LABELS[dataset_name],
        num_trial=1,
        layers=[16, 16, 16],
    )

    cm = AdaptiveWMCClusteringMachine(args, graph, features, target)
    cm.decompose(danmf_result=danmf_result)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = ClusterGCNTrainer(args, cm)
    trainer.train()
    scores = trainer.test()

    return trainer.model, cm, scores, device


# ── plotting ─────────────────────────────────────────────────────────

def plot_single(ax, embeddings_2d, labels, entropies, class_names, colors,
                title, stats_text):
    """Draw one t-SNE subplot on the given axes."""
    min_sz, max_sz = 8, 80
    sizes = min_sz + (max_sz - min_sz) * entropies

    for cls in range(len(class_names)):
        m = labels == cls
        ax.scatter(embeddings_2d[m, 0], embeddings_2d[m, 1],
                   c=[colors[cls]], s=sizes[m], alpha=0.7,
                   edgecolors='gray', linewidths=0.3)

    # class legend (always on left panel)
    handles = [Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=colors[i], markersize=8,
                       label=class_names[i])
               for i in range(len(class_names))]
    handles.append(Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='gray', markersize=6,
                           label='Low H'))
    handles.append(Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='gray', markersize=14,
                           label='High H'))
    ax.legend(handles=handles, loc='upper right', framealpha=0.9,
              fontsize=7.5, title='Class / Entropy', title_fontsize=8)

    ax.set_xlabel('t-SNE dim 1', fontsize=9)
    ax.set_ylabel('t-SNE dim 2', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=7.5,
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))


def make_comparison_figure(dataset_name, results, class_names, colors,
                           output_dir):
    """
    Create a 1x2 side-by-side figure and save as PDF.

    results = {
        'baseline':  {emb_2d, labels, entropies, f1_micro, f1_macro, overlap},
        'adaptive':  {emb_2d, labels, entropies, f1_micro, f1_macro, overlap},
    }
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, key, label in [
        (ax1, 'baseline', '(a) Baseline - Fixed WMC = 0.30'),
        (ax2, 'adaptive', '(b) Proposed - Margin-Adaptive  (lambda = 0.50)'),
    ]:
        r = results[key]
        mean_h = np.mean(r['entropies'])
        ambig  = np.mean(r['entropies'] > 0.3) * 100
        stats  = (f"F1-micro: {r['f1_micro']:.4f}\n"
                  f"F1-macro: {r['f1_macro']:.4f}\n"
                  f"Mean H: {mean_h:.3f}\n"
                  f"Ambiguous (H>0.3): {ambig:.1f}%\n"
                  f"Overlap ratio: {r['overlap']:.2f}")
        plot_single(ax, r['emb_2d'], r['labels'], r['entropies'],
                    class_names, colors, label, stats)

    fig.suptitle(
        f't-SNE of GCN Embeddings - {dataset_name}: '
        f'Baseline vs Margin-Adaptive OCGCN',
        fontsize=12, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir,
                            f'tsne_comparison_{dataset_name.lower()}.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    return pdf_path


# ── main ─────────────────────────────────────────────────────────────

def run_comparison(dataset_name, ds_root, output_dir, seed=42, epochs=20):
    print(f"\n{'='*70}")
    print(f"  Comparative t-SNE: {dataset_name}")
    print(f"{'='*70}")

    seed_everything(seed)
    graph, features, target = load_dataset(dataset_name, ds_root)
    print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    # Shared DANMF decomposition (same as paper experiments)
    base_args = SimpleArgs(
        dataset_name=dataset_name, ds_root=ds_root,
        clustering_method='danmf', clustering_overlap=True,
        membership_closeness=0.3, adaptation_lambda=0.5,
        overlap_strategy='original_wmc',
        epochs=epochs, test_ratio=0.3, seed=seed, dropout=0.5,
        learning_rate=0.01, cluster_number=NUM_LABELS[dataset_name],
        num_trial=1, layers=[16, 16, 16],
    )
    print("  Fitting DANMF (shared)...")
    danmf_result = fit_danmf_cached(graph, base_args, seed)
    P = danmf_result['P']

    # ── Train baseline ──
    print("\n  [1/2] Training baseline (fixed WMC=0.30)...")
    model_b, cm_b, scores_b, device = train_one(
        dataset_name, ds_root,
        strategy='original_wmc', wmc=0.30, lam=0.5,
        epochs=epochs, danmf_result=danmf_result, seed=seed,
    )
    emb_b, nid_b, lab_b = extract_embeddings(model_b, cm_b, device)
    ent_b = compute_entropy(P, nid_b)
    print(f"    F1 micro={scores_b['micro']:.4f}  macro={scores_b['macro']:.4f}")

    # ── Train adaptive ──
    print("  [2/2] Training adaptive (Margin-Adaptive, lambda=0.50)...")
    model_a, cm_a, scores_a, device = train_one(
        dataset_name, ds_root,
        strategy='margin_adaptive_wmc', wmc=0.30, lam=0.5,
        epochs=epochs, danmf_result=danmf_result, seed=seed,
    )
    emb_a, nid_a, lab_a = extract_embeddings(model_a, cm_a, device)
    ent_a = compute_entropy(P, nid_a)
    print(f"    F1 micro={scores_a['micro']:.4f}  macro={scores_a['macro']:.4f}")

    # ── t-SNE (same random state for comparable layouts) ──
    print("\n  Running t-SNE...")
    tsne_params = dict(
        n_components=2, perplexity=min(30, len(emb_b) - 1),
        learning_rate='auto', n_iter=1000, random_state=seed, init='pca',
    )
    tsne_b = TSNE(**tsne_params).fit_transform(emb_b)
    # Re-init with same params but fresh object so both get independent fits
    tsne_a = TSNE(**tsne_params).fit_transform(emb_a)

    # ── Class names / colours ──
    n_cls = int(np.max(lab_b) + 1)
    if dataset_name == 'CiteSeer':
        class_names = ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI']
        colors = plt.cm.Set2(np.linspace(0, 1, 6))
    elif dataset_name == 'IMDB':
        class_names = ['Action', 'Comedy', 'Drama', 'Romance', 'Thriller']
        colors = plt.cm.Set1(np.linspace(0, 1, 5))
    else:
        class_names = [f'Class {i}' for i in range(n_cls)]
        colors = plt.cm.tab10(np.linspace(0, 1, n_cls))

    # ── Figure ──
    results = {
        'baseline': dict(emb_2d=tsne_b, labels=lab_b, entropies=ent_b,
                         f1_micro=scores_b['micro'], f1_macro=scores_b['macro'],
                         overlap=float(np.sum(cm_b.ClusterNodes)) / graph.number_of_nodes()),
        'adaptive': dict(emb_2d=tsne_a, labels=lab_a, entropies=ent_a,
                         f1_micro=scores_a['micro'], f1_macro=scores_a['macro'],
                         overlap=float(np.sum(cm_a.ClusterNodes)) / graph.number_of_nodes()),
    }

    pdf = make_comparison_figure(dataset_name, results, class_names, colors,
                                output_dir)

    # ── Per-class stats ──
    print(f"\n  Per-class entropy comparison:")
    print(f"  {'Class':<12} {'Baseline H':>11} {'Adaptive H':>11} {'Ambig(B)':>9} {'Ambig(A)':>9}")
    print(f"  {'-'*55}")
    for cls in range(len(class_names)):
        mb, ma = lab_b == cls, lab_a == cls
        print(f"  {class_names[cls]:<12} {np.mean(ent_b[mb]):>11.3f} "
              f"{np.mean(ent_a[ma]):>11.3f} "
              f"{np.mean(ent_b[mb]>0.3)*100:>8.1f}% "
              f"{np.mean(ent_a[ma]>0.3)*100:>8.1f}%")

    return pdf


def main():
    parser = argparse.ArgumentParser(
        description='Comparative t-SNE: Baseline vs Adaptive-OCGCN')
    parser.add_argument('--dataset', nargs='+', default=['CiteSeer'])
    parser.add_argument('--ds-root', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds_root   = args.ds_root   or os.path.join(project_root, 'tmp')
    out_dir   = args.output_dir or os.path.join(project_root, 'figures')

    for ds in args.dataset:
        run_comparison(ds, ds_root, out_dir, args.seed, args.epochs)

    print("\nDone.")


if __name__ == '__main__':
    main()
