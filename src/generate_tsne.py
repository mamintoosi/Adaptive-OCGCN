#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate t-SNE visualizations of GCN node embeddings for the Adaptive-OCGCN paper.

This script:
1. Loads a graph dataset (CiteSeer or IMDB)
2. Trains a ClusterGCN model using the same pipeline as the experiments
3. Extracts embeddings from the second-to-last layer of the GCN
4. Computes membership entropy using the same functions as the overlap selectors
5. Generates publication-ready t-SNE visualizations

Usage:
    python generate_tsne.py --dataset CiteSeer
    python generate_tsne.py --dataset IMDB
    python generate_tsne.py --dataset CiteSeer --dataset IMDB
"""

import argparse
import os
import sys
import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_utils import SimpleArgs, fit_danmf_cached, load_dataset, load_hetero_dataset
from clustering import ClusteringMachine, fit_danmf
from clustergcn import ClusterGCNTrainer, seed_everything
from layers import StackedGCN
from overlap_selection.common import compute_membership_entropy


def extract_embeddings(model, clustering_machine, device):
    """
    Extract node embeddings from the second-to-last layer of the GCN model.
    
    The GCN architecture is:
        GCNConv -> ReLU -> GCNConv -> ReLU -> GCNConv -> Dropout -> log_softmax
    
    We want embeddings from the layer before the final projection (before dropout).
    
    Returns:
        embeddings: numpy array of shape (N, hidden_dim) with all node embeddings
        node_indices: list mapping embedding indices to original node indices
    """
    model.eval()
    
    all_embeddings = []
    all_node_indices = []
    all_targets = []
    
    with torch.no_grad():
        for cluster in clustering_machine.clusters:
            edges = clustering_machine.sg_edges[cluster].to(device)
            features = clustering_machine.sg_features[cluster].to(device)
            target = clustering_machine.sg_targets[cluster].to(device).squeeze()
            nodes = clustering_machine.sg_nodes[cluster]
            
            # Manual forward pass to get intermediate embeddings
            h = features
            for i in range(len(model.layers) - 1):
                h = model.layers[i](h, edges)
                h = torch.nn.functional.relu(h)
                if i == len(model.layers) - 2:
                    # This is the second-to-last layer - capture embeddings here
                    embeddings = h.clone()
            
            # Map back to original node indices
            nodes_np = nodes.cpu().numpy()
            embeddings_np = embeddings.cpu().numpy()
            target_np = target.cpu().numpy()
            
            # For overlapping clusters, a node may appear multiple times.
            # We'll average embeddings for nodes that appear in multiple clusters.
            for idx, node_id in enumerate(nodes_np):
                all_embeddings.append(embeddings_np[idx])
                all_node_indices.append(node_id)
                all_targets.append(target_np[idx])
    
    all_embeddings = np.array(all_embeddings)
    all_node_indices = np.array(all_node_indices)
    all_targets = np.array(all_targets)
    
    # Handle duplicates: average embeddings for nodes appearing in multiple clusters
    unique_nodes, inverse = np.unique(all_node_indices, return_inverse=True)
    
    # Average embeddings for each unique node
    unique_embeddings = np.zeros((len(unique_nodes), all_embeddings.shape[1]))
    for i in range(len(unique_nodes)):
        mask = (all_node_indices == unique_nodes[i])
        unique_embeddings[i] = np.mean(all_embeddings[mask], axis=0)
    
    # Get targets for unique nodes (take first occurrence)
    unique_targets = np.zeros(len(unique_nodes), dtype=int)
    for i in range(len(unique_nodes)):
        mask = (all_node_indices == unique_nodes[i])
        unique_targets[i] = all_targets[mask][0]
    
    return unique_embeddings, unique_nodes, unique_targets


def compute_node_entropy(membership_matrix):
    """
    Compute normalized entropy for each node based on DANMF membership matrix.
    
    Args:
        membership_matrix: Normalized membership matrix P (N x K)
    
    Returns:
        entropies: numpy array of shape (N,) with normalized entropy values
    """
    entropies = np.zeros(membership_matrix.shape[0])
    for i in range(membership_matrix.shape[0]):
        entropies[i] = compute_membership_entropy(membership_matrix[i])
    return entropies


def run_tsne_visualization(dataset_name, ds_root, output_dir, seed=42):
    """
    Generate t-SNE visualization for a single dataset.
    
    Args:
        dataset_name: Name of the dataset ('CiteSeer' or 'IMDB')
        ds_root: Root directory containing the datasets
        output_dir: Directory to save the output figures
        seed: Random seed for reproducibility
    """
    print(f"\n{'='*70}")
    print(f"  Generating t-SNE visualization for {dataset_name}")
    print(f"{'='*70}")
    
    # Set random seeds
    seed_everything(seed)
    
    # Load dataset
    print(f"  Loading dataset...")
    graph, features, target = load_dataset(dataset_name, ds_root)
    print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    print(f"  Features: {features.shape[1]}, Classes: {int(np.max(target) + 1)}")
    
    # Get number of labels for DANMF
    num_labels = {
        'Cora': 7, 'CiteSeer': 6, 'PubMed': 3, 'WikiCS': 10,
        'ACM': 3, 'DBLP': 4, 'IMDB': 5,
    }
    
    # Create args for the experiment
    args = SimpleArgs(
        dataset_name=dataset_name,
        ds_root=ds_root,
        clustering_method='danmf',
        clustering_overlap=True,
        membership_closeness=0.3,
        adaptation_lambda=0.5,
        overlap_strategy='entropy_adaptive_wmc',
        epochs=20,  # More epochs for better embeddings
        test_ratio=0.3,
        seed=seed,
        dropout=0.5,
        learning_rate=0.01,
        cluster_number=num_labels[dataset_name],
        num_trial=1,
        layers=[16, 16, 16],
    )
    
    # Fit DANMF (cached)
    print(f"  Fitting DANMF clustering...")
    danmf_result = fit_danmf_cached(graph, args, seed)
    membership_matrix = danmf_result['P']
    print(f"  Membership matrix shape: {membership_matrix.shape}")
    
    # Create clustering machine and train model
    print(f"  Creating clustering machine...")
    cm = ClusteringMachine(args, graph, features, target)
    cm.decompose(danmf_result=danmf_result)
    
    print(f"  Training GCN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = ClusterGCNTrainer(args, cm)
    trainer.train()
    
    # Evaluate
    scores = trainer.test()
    print(f"  Test F1 - Micro: {scores['micro']:.4f}, Macro: {scores['macro']:.4f}")
    
    # Extract embeddings
    print(f"  Extracting embeddings from second-to-last layer...")
    embeddings, node_ids, labels = extract_embeddings(trainer.model, cm, device)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # Compute entropy for each node
    print(f"  Computing membership entropy...")
    # Reindex membership matrix to match node_ids
    P_reordered = membership_matrix[node_ids]
    entropies = compute_node_entropy(P_reordered)
    
    # t-SNE reduction
    print(f"  Running t-SNE (this may take a moment)...")
    tsne_params = {
        'n_components': 2,
        'perplexity': min(30, len(embeddings) - 1),
        'learning_rate': 'auto',
        'n_iter': 1000,
        'random_state': seed,
        'init': 'pca',
        'early_exaggeration': 12,
    }
    tsne = TSNE(**tsne_params)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create visualization
    print(f"  Creating visualization...")
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Class names and colors
    if dataset_name == 'CiteSeer':
        class_names = ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI']
        colors = plt.cm.Set2(np.linspace(0, 1, 6))
    elif dataset_name == 'IMDB':
        class_names = ['Action', 'Comedy', 'Drama', 'Romance', 'Thriller']
        colors = plt.cm.Set1(np.linspace(0, 1, 5))
    else:
        n_classes = int(np.max(labels) + 1)
        class_names = [f'Class {i}' for i in range(n_classes)]
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # Scale point sizes based on entropy (higher entropy = larger points)
    min_size = 10
    max_size = 100
    sizes = min_size + (max_size - min_size) * entropies
    
    # Plot each class
    for cls in range(len(class_names)):
        mask = (labels == cls)
        scatter = ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[cls]],
            s=sizes[mask],
            alpha=0.7,
            edgecolors='gray',
            linewidths=0.3,
            label=class_names[cls],
        )
    
    # Create legend for classes
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
               markersize=10, label=class_names[i])
        for i in range(len(class_names))
    ]
    
    # Add entropy legend
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                   markersize=8, label='Low entropy'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                   markersize=16, label='High entropy'))
    
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9,
              fontsize=9, title='Class / Entropy (size)')
    
    # Labels and title
    ax.set_xlabel('t-SNE dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE dimension 2', fontsize=11)
    ax.set_title(f't-SNE Visualization of GCN Embeddings — {dataset_name}\n'
                 f'(colored by class, sized by membership entropy)', fontsize=12)
    
    # Add statistics as text annotation
    mean_entropy = np.mean(entropies)
    # Compute overlap fraction: nodes with entropy > 0.3 are considered ambiguous
    overlap_threshold = 0.3
    overlap_fraction = np.mean(entropies > overlap_threshold)
    
    stats_text = (f'Mean entropy: {mean_entropy:.3f}\n'
                  f'Ambiguous nodes (H>0.3): {overlap_fraction:.1%}\n'
                  f'Nodes: {len(labels)}, t-SNE perplexity: {tsne_params["perplexity"]}')
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Compute per-class statistics
    print(f"\n  Per-class statistics:")
    print(f"  {'Class':<12} {'Count':>6} {'Mean H':>8} {'Ambiguous':>10}")
    print(f"  {'-'*40}")
    for cls in range(len(class_names)):
        mask = (labels == cls)
        cls_count = np.sum(mask)
        cls_mean_h = np.mean(entropies[mask])
        cls_ambiguous = np.mean(entropies[mask] > overlap_threshold)
        print(f"  {class_names[cls]:<12} {cls_count:>6} {cls_mean_h:>8.3f} {cls_ambiguous:>10.1%}")
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'tsne_{dataset_name.lower()}_embeddings.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Figure saved to: {output_path}")
    
    # Also save as PDF for vector format
    pdf_path = os.path.join(output_dir, f'tsne_{dataset_name.lower()}_embeddings.pdf')
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for cls in range(len(class_names)):
        mask = (labels == cls)
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[cls]],
            s=sizes[mask],
            alpha=0.7,
            edgecolors='gray',
            linewidths=0.3,
            label=class_names[cls],
        )
    
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9,
              fontsize=9, title='Class / Entropy (size)')
    ax.set_xlabel('t-SNE dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE dimension 2', fontsize=11)
    ax.set_title(f't-SNE Visualization of GCN Embeddings — {dataset_name}\n'
                 f'(colored by class, sized by membership entropy)', fontsize=12)
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  PDF saved to: {pdf_path}")
    
    return {
        'dataset': dataset_name,
        'embeddings': embeddings,
        'embeddings_2d': embeddings_2d,
        'labels': labels,
        'entropies': entropies,
        'node_ids': node_ids,
        'mean_entropy': mean_entropy,
        'overlap_fraction': overlap_fraction,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate t-SNE visualizations for Adaptive-OCGCN')
    parser.add_argument('--dataset', nargs='+', default=['CiteSeer'],
                       help='Dataset(s) to visualize (default: CiteSeer)')
    parser.add_argument('--ds-root', default=None,
                       help='Root directory for datasets (default: ../tmp)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for figures (default: ../figures)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    args = parser.parse_args()
    
    # Set default paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds_root = args.ds_root or os.path.join(project_root, 'tmp')
    output_dir = args.output_dir or os.path.join(project_root, 'figures')
    
    print(f"Project root: {project_root}")
    print(f"Dataset root: {ds_root}")
    print(f"Output directory: {output_dir}")
    print(f"Datasets: {args.dataset}")
    
    results = {}
    for dataset_name in args.dataset:
        result = run_tsne_visualization(dataset_name, ds_root, output_dir, args.seed)
        results[dataset_name] = result
    
    # Print summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for ds_name, res in results.items():
        print(f"\n  {ds_name}:")
        print(f"    Mean entropy: {res['mean_entropy']:.3f}")
        print(f"    Ambiguous nodes (H>0.3): {res['overlap_fraction']:.1%}")
        print(f"    Embeddings shape: {res['embeddings'].shape}")
        print(f"    2D embeddings shape: {res['embeddings_2d'].shape}")


if __name__ == '__main__':
    main()
