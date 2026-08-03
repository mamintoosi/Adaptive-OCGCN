"""
Hybrid-Adaptive WMC: threshold adapts to the stronger of two ambiguity signals.

For each node i:
    H_norm(i) = normalized entropy of P(i,:)
    A(i)      = 1 - (p1 - p2)          (margin ambiguity)
    a_i       = max(H_norm(i), A(i))
    WMC_i     = WMC_base * (1 - λ * a_i)

Using the maximum of the two complementary signals lets the selector react to
whichever type of ambiguity is dominant for each node: global distribution
spread (entropy) or top-two competition (margin).
"""
from typing import List

import numpy as np

from overlap_selection.common import (
    OverlapSelector,
    compute_membership_entropy,
    compute_membership_margin,
)


class HybridAdaptiveWMCSelector(OverlapSelector):
    """
    Hybrid entropy+margin adaptive overlap selection.

    Nodes with high ambiguity under either measure receive a lower WMC
    threshold, allowing them to join more clusters.
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
                entropy = compute_membership_entropy(row)
                margin = compute_membership_margin(row)
                ambiguity = max(entropy, margin)
                wmc_i = self.membership_closeness * (1.0 - self.lam * ambiguity)
                wmc_i = max(wmc_i, 0.01)  # floor to avoid selecting everything

                npw = np.where(row >= max_in_row * wmc_i)
                tmp = npw[0].tolist()
                cluster_indices = [x for x in tmp if x in valid_clusters]

            near_clusters.append(cluster_indices)
        return near_clusters
