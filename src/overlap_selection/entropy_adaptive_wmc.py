"""
Entropy-Adaptive WMC: threshold adapts to membership entropy.

For each node i:
    H_norm(i) = normalized entropy of P(i,:)
    WMC_i = WMC_base * (1 - λ * H_norm(i))

High entropy (ambiguous membership) → lower threshold → more overlap.
Low entropy (confident membership) → threshold stays near WMC_base.
"""
from typing import List

import numpy as np

from overlap_selection.common import OverlapSelector, compute_membership_entropy


class EntropyAdaptiveWMCSelector(OverlapSelector):
    """
    Entropy-adaptive overlap selection.

    Nodes with high membership entropy (ambiguous cluster assignment)
    receive a lower WMC threshold, allowing them to join more clusters.
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
                h_norm = compute_membership_entropy(row)
                wmc_i = self.membership_closeness * (1.0 - self.lam * h_norm)
                wmc_i = max(wmc_i, 0.01)  # floor to avoid selecting everything

                npw = np.where(row >= max_in_row * wmc_i)
                tmp = npw[0].tolist()
                cluster_indices = [x for x in tmp if x in valid_clusters]

            near_clusters.append(cluster_indices)
        return near_clusters
