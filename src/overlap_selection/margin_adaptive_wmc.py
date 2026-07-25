"""
Margin-Adaptive WMC: threshold adapts to membership margin.

For each node i:
    A(i) = 1 - (p1 - p2)   where p1, p2 are top-2 probabilities
    WMC_i = WMC_base * (1 - λ * A(i))

Small margin (ambiguous) → lower threshold → more overlap.
Large margin (confident) → threshold stays near WMC_base.
"""
from typing import List

import numpy as np

from overlap_selection.common import OverlapSelector, compute_membership_margin


class MarginAdaptiveWMCSelector(OverlapSelector):
    """
    Margin-adaptive overlap selection.

    Nodes with small margin between top-2 membership probabilities
    (ambiguous cluster assignment) receive a lower WMC threshold.
    """

    def __init__(self, membership_closeness: float = 0.3, lam: float = 0.5):
        """
        Args:
            membership_closeness: Base WMC threshold in (0, 1].
            lam: Adaptation strength in [0, 1]. Higher = more adaptation.
        """
        self.membership_closeness = membership_closeness
        self.lam = lam

    def select_overlap(
        self,
        P: np.ndarray,
        valid_clusters: List[int],
    ) -> List[List[int]]:
        near_clusters = []
        for i in range(P.shape[0]):
            row = P[i]
            max_in_row = np.max(row)

            if max_in_row == 0:
                cluster_indices = [0]
            else:
                ambiguity = compute_membership_margin(row)
                wmc_i = self.membership_closeness * (1.0 - self.lam * ambiguity)
                wmc_i = max(wmc_i, 0.01)  # floor to avoid selecting everything

                npw = np.where(row >= max_in_row * wmc_i)
                tmp = npw[0].tolist()
                cluster_indices = [x for x in tmp if x in valid_clusters]

            near_clusters.append(cluster_indices)
        return near_clusters
