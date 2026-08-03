"""
Adaptive-WMC Overlapping Cluster-GCN Clustering Module.

Replaces global WMC threshold with node-specific thresholds based on
ambiguity in the DANMF membership matrix. No warm-up predictor needed.
"""
from typing import List

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from clustering import fit_danmf
from overlap_selection import create_selector


class AdaptiveWMCClusteringMachine:
    """
    Clustering machine with adaptive WMC overlap selection.

    Supports four modes via the overlap_strategy parameter:
    - original_wmc: global threshold (baseline)
    - entropy_adaptive_wmc: threshold adapts to membership entropy
    - margin_adaptive_wmc: threshold adapts to membership margin
    - hybrid_adaptive_wmc: threshold adapts to max(entropy, margin)
    """

    def __init__(self, args, graph, features, target):
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self._set_sizes()

        self.overlap_strategy = getattr(args, 'overlap_strategy', 'original_wmc')
        self.membership_closeness = getattr(args, 'membership_closeness', 0.3)
        self.adaptation_lambda = getattr(args, 'adaptation_lambda', 0.5)

    def _set_sizes(self) -> None:
        self.feature_count: int = self.features.shape[1]
        self.class_count: int = int(np.max(self.target) + 1)

    def decompose(self, danmf_result=None) -> None:
        """Decompose graph using DANMF clustering + adaptive overlap."""
        self.danmf_clustering(danmf_result=danmf_result)
        self.general_data_partitioning()
        self.transfer_edges_and_nodes()

    def danmf_clustering(self, danmf_result=None) -> None:
        """Perform DANMF clustering with selectable overlap strategy."""
        num_labels = {
            'CiteSeer': 6, 'Cora': 7, 'PubMed': 3, 'WikiCS': 10,
            'ACM': 3, 'DBLP': 4, 'IMDB': 5,
        }

        cluster_count = num_labels.get(self.args.dataset_name, self.args.cluster_number)

        if danmf_result is not None:
            P = danmf_result["P"]
            self.clusters: List[int] = list(danmf_result["clusters"])
            values_list = [danmf_result["hard_membership"][node] for node in sorted(self.graph.nodes())]
        else:
            result = fit_danmf(
                self.graph,
                cluster_count,
                seed=getattr(self.args, "seed", 42),
            )
            P = result["P"]
            self.clusters = list(result["clusters"])
            values_list = [result["hard_membership"][node] for node in sorted(self.graph.nodes())]

        self.membership_matrix: np.ndarray = P

        if not self.args.clustering_overlap:
            self.cluster_membership = {
                node: membership for node, membership in enumerate(values_list)
            }
        else:
            selector = create_selector(
                self.overlap_strategy,
                membership_closeness=self.membership_closeness,
                lam=self.adaptation_lambda,
            )
            near_clusters = selector.select_overlap(P, self.clusters)
            self.cluster_membership = {
                node: membership for node, membership in enumerate(near_clusters)
            }

    def general_data_partitioning(self) -> None:
        """Create data partitions for each cluster."""
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}
        self.ClusterNodes: List[int] = []

        for cluster in self.clusters:
            if self.args.clustering_overlap:
                subgraph = self.graph.subgraph([
                    node for node in sorted(self.graph.nodes())
                    if cluster in self.cluster_membership[node]
                ])
            else:
                subgraph = self.graph.subgraph([
                    node for node in sorted(self.graph.nodes())
                    if self.cluster_membership[node] == cluster
                ])

            self.ClusterNodes.append(len(subgraph.nodes()))
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}

            self.sg_edges[cluster] = [
                [mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()
            ] + [
                [mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()
            ]

            self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(
                list(mapper.values()), test_size=self.args.test_ratio,
            )
            self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
            self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])

            self.sg_features[cluster] = self.features[self.sg_nodes[cluster], :]
            self.sg_targets[cluster] = self.target[self.sg_nodes[cluster], :]

    def transfer_edges_and_nodes(self) -> None:
        """Convert data to PyTorch tensors."""
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster])
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])
