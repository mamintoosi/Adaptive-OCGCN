"""
Original WMC: global threshold for all nodes.

Node i is assigned to cluster c if:
    P(i,c) >= WMC * max(P(i,*))

This is the baseline method from the original OCGCN paper.
"""
from typing import List

import numpy as np

from overlap_selection.common import OverlapSelector


class OriginalWMCSelector(OverlapSelector):
    """
    Original Overlapping Cluster-GCN overlap selection.

    Uses a single global threshold (WMC) for every node.
    """

    def __init__(self, membership_closeness: float = 0.3):
        """
        Args:
            membership_closeness: Global WMC threshold in (0, 1].
                                  Lower = more overlap.
        """
        self.membership_closeness = membership_closeness

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
                npw = np.where(row >= max_in_row * self.membership_closeness)
                tmp = npw[0].tolist()
                cluster_indices = [x for x in tmp if x in valid_clusters]

            near_clusters.append(cluster_indices)
        return near_clusters
