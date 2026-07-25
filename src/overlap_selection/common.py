"""
Common utilities and base interface for overlap selection strategies.

All strategies inherit from OverlapSelector and implement select_overlap().
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class OverlapSelector(ABC):
    """
    Abstract base class for overlap selection strategies.

    Subclasses implement select_overlap() to determine which clusters
    each node belongs to, based on the DANMF membership matrix.
    """

    @abstractmethod
    def select_overlap(
        self,
        P: np.ndarray,
        valid_clusters: List[int],
    ) -> List[List[int]]:
        """
        Select overlap nodes and assign clusters.

        Args:
            P: Normalized DANMF membership matrix (N x K), rows sum to 1.
            valid_clusters: List of valid cluster IDs from DANMF.

        Returns:
            List of length N, where entry i is the list of cluster IDs
            assigned to node i.
        """
        ...


def compute_membership_entropy(row: np.ndarray) -> float:
    """
    Compute normalized entropy of a membership probability vector.

    H(i) = -Σ P(i,k) * log(P(i,k))
    H_norm = H / log(K)  where K = number of non-zero entries

    Args:
        row: Normalized membership probabilities for one node (K,).

    Returns:
        Normalized entropy in [0, 1]. 0 = deterministic, 1 = uniform.
    """
    probs = row[row > 0]
    if len(probs) <= 1:
        return 0.0

    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))
    return entropy / max_entropy


def compute_membership_margin(row: np.ndarray) -> float:
    """
    Compute ambiguity from the margin between top-2 membership probabilities.

    margin = p1 - p2
    ambiguity = 1 - margin

    Args:
        row: Normalized membership probabilities for one node (K,).

    Returns:
        Ambiguity score in [0, 1]. 0 = confident (large margin),
        1 = maximally ambiguous (equal top-2).
    """
    sorted_probs = np.sort(row)[::-1]
    p1 = sorted_probs[0]
    p2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = p1 - p2
    return 1.0 - margin
