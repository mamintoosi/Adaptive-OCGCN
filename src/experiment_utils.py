"""
Shared helpers for experiment scripts.

Centralizes dataset loading, RNG seeding, DANMF caching and the single-run
pipeline so all experiment scripts behave identically:
  - every run is fully seeded (torch, numpy, python)
  - the DANMF decomposition is cached once per dataset with a fixed seed and
    reused across all overlap strategies and run seeds -> a clean paired
    design (matching the original OCGCN protocol)
  - both micro and macro F1 are reported
"""
import os
import random
import sys
import time

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clustering import ClusteringMachine, fit_danmf
from ra_ocgcn_clustering import AdaptiveWMCClusteringMachine
from clustergcn import ClusterGCNTrainer, seed_everything
from hetero_utils import HETERO_LOADERS
from utils import load_citation_dataset

NUM_LABELS = {
    'Cora': 7, 'CiteSeer': 6, 'PubMed': 3, 'WikiCS': 10,
    'ACM': 3, 'DBLP': 4, 'IMDB': 5,
}


class SimpleArgs:
    """Minimal argparse-free argument container."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def load_hetero_dataset(ds_name, ds_root):
    """Load a heterogeneous dataset and convert it to a homogeneous graph."""
    loader = HETERO_LOADERS[ds_name]
    graph, features, target = loader(ds_root)

    isolates = list(nx.isolates(graph))
    if isolates:
        graph.remove_nodes_from(isolates)
        features = np.delete(features, isolates, axis=0)
        non_isolates = [n for n in range(len(target)) if n not in set(isolates)]
        target = target[non_isolates]
    mapping = {old: new for new, old in enumerate(sorted(graph.nodes()))}
    graph = nx.relabel_nodes(graph, mapping)
    return graph, features, target


def load_dataset(ds_name, ds_root):
    """Load any supported dataset (citation or heterogeneous)."""
    if ds_name in ('ACM', 'DBLP', 'IMDB'):
        return load_hetero_dataset(ds_name, ds_root)
    return load_citation_dataset(ds_name, ds_root)


# Fixed clustering seed matching the original OCGCN protocol. The original
# code never passed a seed to DANMF, so karateclub's default (seed=42) produced
# an IDENTICAL decomposition for every run. We preserve that: a single cached
# decomposition per dataset is shared by all strategies and all run seeds,
# giving a clean paired design where only the train/test split and GCN
# initialization vary per seed.
_DANMF_CLUSTER_SEED = 42

_fit_cache = {}


def fit_danmf_cached(graph, args, seed=None):
    """
    Fit DANMF once per dataset with a FIXED clustering seed and cache it.

    All overlap strategies on all run seeds share the exact same community
    memberships (matching the original OCGCN experiments), so paired
    statistical tests only reflect split + initialization variance.
    """
    key = args.dataset_name
    if key not in _fit_cache:
        cluster_count = NUM_LABELS.get(args.dataset_name, args.cluster_number)
        _fit_cache[key] = fit_danmf(graph, cluster_count, seed=_DANMF_CLUSTER_SEED)
    return _fit_cache[key]


def run_single(graph, features, target, args, method, danmf_result=None):
    """
    Run one experiment and return F1 scores.

    :param method: "no_overlap", "original_wmc", "entropy_adaptive_wmc",
                   "margin_adaptive_wmc" or "hybrid_adaptive_wmc".
    :param danmf_result: Optional cached DANMF decomposition (from fit_danmf).
    """
    seed_everything(args.seed)

    start = time.time()

    if method == "no_overlap":
        cm = ClusteringMachine(args, graph, features, target)
    else:
        cm = AdaptiveWMCClusteringMachine(args, graph, features, target)

    cm.decompose(danmf_result=danmf_result)
    trainer = ClusterGCNTrainer(args, cm)
    trainer.train()
    scores = trainer.test()
    elapsed = time.time() - start

    avg_overlap = float(np.sum(cm.ClusterNodes)) / len(graph.nodes())

    return {
        "f1_micro": scores["micro"],
        "f1_macro": scores["macro"],
        "runtime": elapsed,
        "overlap_ratio": avg_overlap,
        "num_clusters": len(cm.clusters),
    }
