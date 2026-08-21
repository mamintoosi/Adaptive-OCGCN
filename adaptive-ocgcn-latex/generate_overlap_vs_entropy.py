#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate improved overlap_vs_entropy bar chart (Table 4 data).

Output: figures/overlap_vs_entropy.pdf and .png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Data from Table 4 (Stratified Ambiguity Analysis) ────────────────
# Each dataset: list of (label, overlap_ratio) in display order (left to right)
DATA = {
    'Cora':     [('Low', 0.000), ('Medium', 0.076), ('High', 0.761)],
    'CiteSeer': [('Low+Med', 0.000), ('High', 0.411)],
    'PubMed':   [('Low', 0.000), ('Medium', 0.000), ('High', 0.689)],
    'ACM':      [('Low', 0.000), ('Medium', 0.000), ('High', 0.718)],
    'DBLP':     [('Low+Med', 0.006), ('High', 0.620)],
    'IMDB':     [('Low', 0.170), ('Medium', 0.739), ('High', 0.953)],
}

# Display order: left to right on the grouped bar chart
DATASET_ORDER = ['Cora', 'CiteSeer', 'PubMed', 'ACM', 'DBLP', 'IMDB']

# Colour palette: light -> medium -> dark blue gradient
COLORS = {
    'Low':     '#A8D8EA',   # light blue
    'Low+Med': '#A8D8EA',   # same light blue for merged tercile
    'Medium':  '#5B9BD5',   # medium blue
    'High':    '#1B4F72',   # dark blue
}

EDGE_COLOR = '#333333'
MIN_BAR_HEIGHT = 0.015  # minimum visible height for zero bars


def main():
    # ── Layout constants ──
    n_datasets = len(DATASET_ORDER)
    max_bars = max(len(DATA[ds]) for ds in DATASET_ORDER)  # 3
    group_width = 0.75          # total width per dataset group
    bar_width = group_width / max_bars  # individual bar width
    group_gap = 0.35            # gap between dataset groups
    total_width = n_datasets * group_width + (n_datasets - 1) * group_gap

    fig, ax = plt.subplots(figsize=(7.2, 3.6))  # single-column width (~7 in)

    # ── Draw bars ──
    x_positions = []
    x_labels = []
    x_centers = []
    legend_handles = {}

    for ds_idx, ds_name in enumerate(DATASET_ORDER):
        bars = DATA[ds_name]
        n = len(bars)
        # Centre the bar group
        group_start = ds_idx * (group_width + group_gap)
        # For 2-bar datasets, shift bars slightly right to align with 3-bar layout
        offset = (max_bars - n) * bar_width / 2

        for bar_idx, (label, ratio) in enumerate(bars):
            x = group_start + offset + bar_idx * bar_width + bar_width / 2
            height = max(ratio, MIN_BAR_HEIGHT) if ratio == 0 else ratio
            color = COLORS[label]

            bar = ax.bar(
                x, height, width=bar_width * 0.88,
                color=color, edgecolor=EDGE_COLOR, linewidth=0.6,
                zorder=3,
            )

            # Store legend handle (only once per label type)
            if label not in legend_handles:
                legend_handles[label] = bar

            # Exact value label on top of bar (skip zero)
            if ratio > 0:
                ax.text(x, ratio + 0.02, f'{ratio:.3f}',
                        ha='center', va='bottom', fontsize=6.5,
                        fontweight='medium', color=EDGE_COLOR)

        x_centers.append(group_start + group_width / 2)
        x_labels.append(ds_name)

    # ── Axes ──
    ax.set_xticks(x_centers)
    ax.set_xticklabels(x_labels, fontsize=9, fontweight='medium')
    ax.set_xlim(-group_gap / 2,
                n_datasets * (group_width + group_gap) - group_gap / 2)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    ax.set_ylabel('Overlap Ratio', fontsize=10, fontweight='medium')
    ax.tick_params(axis='y', labelsize=8.5)

    # ── Grid ──
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # ── Legend ──
    # Order: Low, Low+Med, Medium, High  (but only draw what exists)
    legend_order = ['Low', 'Medium', 'High']
    handles = [legend_handles[k] for k in legend_order if k in legend_handles]
    labels  = [k for k in legend_order if k in legend_handles]
    ax.legend(handles, labels, loc='upper left', fontsize=8,
              framealpha=0.9, edgecolor='#cccccc', title='Entropy Tercile',
              title_fontsize=8.5)

    # ── Spine cleanup ──
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # ── Save ──
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(out_dir, exist_ok=True)

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'overlap_vs_entropy.{ext}')
        try:
            fig.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
            print(f'Saved: {path}')
        except PermissionError:
            tmp = os.path.join(out_dir, f'overlap_vs_entropy_new.{ext}')
            fig.savefig(tmp, dpi=600, bbox_inches='tight', facecolor='white')
            print(f'Saved (rename manually): {tmp}')

    plt.close(fig)
    print('Done.')


if __name__ == '__main__':
    main()
