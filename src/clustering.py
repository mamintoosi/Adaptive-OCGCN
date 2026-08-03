import torch
import random
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from karateclub.community_detection.overlapping import DANMF


def fit_danmf(graph, cluster_count, seed=42, pre_iterations=500, iterations=200):
    """
    Fit a DANMF model on the graph and return a cached decomposition.

    :param graph: Networkx Graph.
    :param cluster_count: Number of latent communities.
    :param seed: Random seed (fully controls DANMF initialization).
    :return dict with:
        P                - normalized membership matrix (N x K)
        clusters         - list of valid cluster IDs
        hard_membership  - dict {node: cluster_id}
    """
    np.random.seed(seed)
    model = DANMF(
        layers=[32, 2 * cluster_count],
        pre_iterations=pre_iterations,
        iterations=iterations,
        seed=seed,
    )
    model.fit(graph)
    P = normalize(model._P, axis=1)
    memberships = model.get_memberships()
    values_list = [memberships[node] for node in sorted(graph.nodes())]
    clusters = list(set(values_list))
    hard_membership = {node: memberships[node] for node in sorted(graph.nodes())}
    return {"P": P, "clusters": clusters, "hard_membership": hard_membership}


class ClusteringMachine(object):
    """
    Clustering the graph, feature set and target.
    """
    def __init__(self, args, graph, features, target):
        """
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self._set_sizes()

    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        self.feature_count = self.features.shape[1] 
        self.class_count = np.max(self.target)+1

    def decompose(self, danmf_result=None):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        :param danmf_result: Optional cached DANMF output from fit_danmf.
        """
        if self.args.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            self.metis_clustering()
        elif self.args.clustering_method == "random":
            print("\nRandom graph clustering started.\n")
            self.random_clustering()
        elif self.args.clustering_method == "danmf":
            self.danmf_clustering(danmf_result=danmf_result)
        elif self.args.clustering_method == "graph":
            print("\ngraph clustering started.\n")
            self.graph_clustering()

        self.general_data_partitioning()
        self.transfer_edges_and_nodes()

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        """
        Clustering the graph with Metis. Requires the 'metis' package.
        """
        try:
            import metis
        except ImportError:
            raise ImportError(
                "The 'metis' package is not installed. "
                "Use clustering_method='danmf' (default) or install metis."
            )
        (st, parts) = metis.part_graph(self.graph, self.args.cluster_number)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def danmf_clustering(self, danmf_result=None):
        """
        Clustering the graph with DANMF.

        If a cached decomposition is supplied (from fit_danmf), it is reused
        so that all overlap strategies share the same community memberships
        for a given dataset/seed.
        """
        if danmf_result is not None:
            P = danmf_result["P"]
            self.clusters = list(danmf_result["clusters"])
            values_list = [danmf_result["hard_membership"][node] for node in sorted(self.graph.nodes())]
        else:
            cluster_count = getattr(self.args, "cluster_number", 10)
            result = fit_danmf(
                self.graph,
                cluster_count,
                seed=getattr(self.args, "seed", 42),
            )
            P = result["P"]
            self.clusters = list(result["clusters"])
            values_list = [result["hard_membership"][node] for node in sorted(self.graph.nodes())]

        if not self.args.clustering_overlap:
            near_clusters = values_list
        else:
            near_clusters = []
            for i in range(P.shape[0]):
                row = P[i]
                max_in_row = np.max(row)
                if max_in_row == 0:
                    cluster_indices = [0]
                else:
                    npw = np.where(row >= (max_in_row * self.args.membership_closeness))
                    tmp = npw[0].tolist()
                    cluster_indices = [x for x in tmp if x in self.clusters]
                near_clusters.append(cluster_indices)

        self.cluster_membership = {node: membership for node, membership in enumerate(near_clusters)}

    def graph_clustering(self):
        """
        Clustering the graph with spectral clustering (sklearn).
        """
        from sklearn.cluster import SpectralClustering
        adj = nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes()))
        clustering = SpectralClustering(
            n_clusters=self.args.cluster_number,
            affinity="precomputed",
            random_state=getattr(self.args, "seed", 42),
        )
        labels = clustering.fit_predict(adj)
        self.clusters = list(range(self.args.cluster_number))
        self.cluster_membership = {node: int(labels[i]) for i, node in enumerate(sorted(self.graph.nodes()))}

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}
        self.ClusterNodes = []
        for cluster in self.clusters:
            if self.args.clustering_overlap:
                subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if cluster in self.cluster_membership[node]])
            else:
                subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])

            self.ClusterNodes.append(len(subgraph.nodes()))
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()), test_size = self.args.test_ratio)
            self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
            self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])
            self.sg_features[cluster] = self.features[self.sg_nodes[cluster],:]
            self.sg_targets[cluster] = self.target[self.sg_nodes[cluster],:]

    def transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format.
        """
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster])
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])
