"""
Overlap selection strategies for Overlapping Cluster-GCN.

Provides three selectable modes for determining which nodes belong to
multiple clusters based on the DANMF membership matrix:

1. original_wmc — global threshold (baseline from the original paper)
2. entropy_adaptive_wmc — threshold adapts to membership entropy
3. margin_adaptive_wmc — threshold adapts to membership margin

All strategies operate on the normalized DANMF membership matrix P
and produce a list of cluster assignments per node.
"""
from overlap_selection.common import OverlapSelector, compute_membership_entropy, compute_membership_margin
from overlap_selection.original_wmc import OriginalWMCSelector
from overlap_selection.entropy_adaptive_wmc import EntropyAdaptiveWMCSelector
from overlap_selection.margin_adaptive_wmc import MarginAdaptiveWMCSelector

SELECTORS = {
    "original_wmc": OriginalWMCSelector,
    "entropy_adaptive_wmc": EntropyAdaptiveWMCSelector,
    "margin_adaptive_wmc": MarginAdaptiveWMCSelector,
}


def create_selector(strategy: str, **kwargs) -> OverlapSelector:
    """
    Factory function to create an overlap selector.

    Args:
        strategy: One of "original_wmc", "entropy_adaptive_wmc", "margin_adaptive_wmc"
        **kwargs: Strategy-specific parameters (membership_closeness, lam)

    Returns:
        An OverlapSelector instance

    Raises:
        ValueError: If strategy is not recognized
    """
    if strategy not in SELECTORS:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(SELECTORS.keys())}")

    import inspect
    sig = inspect.signature(SELECTORS[strategy].__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return SELECTORS[strategy](**filtered_kwargs)
