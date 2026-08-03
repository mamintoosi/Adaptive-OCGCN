"""Generate publication-quality plots for the Adaptive-OCGCN paper.

All figures are derived from the experiment result CSVs (results/*.csv), so the
figures always match the numbers reported in the paper tables.

Run from the repo root:
    python adaptive-ocgcn-latex/generate_plots.py
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, 'results')
OUTPUT_DIR = os.path.join(HERE, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'ACM', 'DBLP', 'IMDB']


def load_fixed_summary():
    df = pd.read_csv(os.path.join(RESULTS, 'fixed_wmc_summary.csv'))
    return df


def get_micro(label, df, ds):
    row = df[(df['label'] == label) & (df['dataset'] == ds)]
    if len(row) == 0:
        return np.nan
    return row['f1_micro_mean'].iloc[0]


def get_overlap(label, df, ds):
    row = df[(df['label'] == label) & (df['dataset'] == ds)]
    if len(row) == 0:
        return np.nan
    return row['overlap_mean'].iloc[0]


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Main results grouped bar chart (micro F1, 20 seeds)
# ═══════════════════════════════════════════════════════════════════════════
def fig_main_results(df):
    labels = ['no_overlap', 'WMC_0.30', 'entropy_l050', 'margin_l050', 'hybrid_l050']
    names = ['No Overlap', 'OCGCN (WMC=0.30)', 'Entropy ($\\lambda$=0.5)',
             'Margin ($\\lambda$=0.5)', 'Hybrid ($\\lambda$=0.5)']
    colors = ['#b0b0b0', '#7faadb', '#e06666', '#93c47d', '#c9a0dc']

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(DATASETS))
    w = 0.16
    bars_all = []
    for j, (lab, name, color) in enumerate(zip(labels, names, colors)):
        vals = [get_micro(lab, df, ds) for ds in DATASETS]
        offset = (j - 2) * w
        bars = ax.bar(x + offset, vals, w, label=name, color=color,
                      edgecolor='black', linewidth=0.5)
        bars_all.append(bars)

    # Mark the best per dataset (only over the 5 shown methods)
    for i in range(len(DATASETS)):
        vals = [get_micro(lab, df, DATASETS[i]) for lab in labels]
        best_idx = int(np.nanargmax(vals))
        bars_all[best_idx][i].set_edgecolor('red')
        bars_all[best_idx][i].set_linewidth(2.0)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Micro F1')
    ax.set_title('Main Results: Micro F1 Across Overlap Selection Strategies (20 seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0.35, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'main_results_comparison.png'))
    plt.close()
    print('Saved: main_results_comparison.png')


# ═══════════════════════════════════════════════════════════════════════════
# Figures 2-4: Lambda sensitivity (entropy / margin / hybrid)
# ═══════════════════════════════════════════════════════════════════════════
def fig_lambda_sensitivity(strategy, fname, suptitle):
    ab = pd.read_csv(os.path.join(RESULTS, 'ablation_summary.csv'))
    configs = {
        'entropy': ['A2_entropy_l025', 'A3_entropy_l050', 'A4_entropy_l075'],
        'margin': ['A5_margin_l025', 'A6_margin_l050', 'A7_margin_l075'],
        'hybrid': ['A8_hybrid_l025', 'A9_hybrid_l050', 'A10_hybrid_l075'],
    }[strategy]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    lambda_vals = [0.25, 0.50, 0.75]
    colors_lam = ['#4a86c8', '#e06666', '#93c47d']

    for idx, (ds, ax) in enumerate(zip(DATASETS, axes.flat)):
        vals = []
        for cfg in configs:
            row = ab[(ab['dataset'] == ds) & (ab['config'] == cfg)]
            vals.append(row['f1_micro_mean'].iloc[0] if len(row) else np.nan)
        bars = ax.bar([str(l) for l in lambda_vals], vals, color=colors_lam,
                      edgecolor='black', linewidth=0.5)
        best_i = int(np.nanargmax(vals))
        bars[best_i].set_edgecolor('red')
        bars[best_i].set_linewidth(2.0)
        ax.set_title(ds, fontweight='bold')
        ax.set_ylabel('Micro F1')
        ax.set_xlabel('$\\lambda$')
        lo = min(vals) - 0.02
        hi = max(vals) + 0.01
        ax.set_ylim(lo, hi)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (hi - lo) * 0.01,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname))
    plt.close()
    print(f'Saved: {fname}')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Fair comparison at matched overlap ratios
# ═══════════════════════════════════════════════════════════════════════════
def fig_fair_comparison():
    fair = pd.read_csv(os.path.join(RESULTS, 'fair_comparison.csv'))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DATASETS))
    w = 0.32

    # Best adaptive per dataset (by f1_diff)
    best_rows = fair.loc[fair.groupby('dataset')['f1_diff'].idxmax()]
    best_rows = best_rows.set_index('dataset').reindex(DATASETS)

    bars_a = ax.bar(x - w / 2, best_rows['adapt_f1'], w, label='Best Adaptive',
                    color='#e06666', edgecolor='black', linewidth=0.5)
    bars_f = ax.bar(x + w / 2, best_rows['fixed_f1'], w, label='Closest Fixed WMC',
                    color='#7faadb', edgecolor='black', linewidth=0.5)

    for i in range(len(DATASETS)):
        diff = best_rows['f1_diff'].iloc[i]
        y_top = max(best_rows['adapt_f1'].iloc[i], best_rows['fixed_f1'].iloc[i]) + 0.005
        color = '#2ca02c' if diff > 0 else '#d62728'
        ax.annotate(f'{diff:+.3f}', xy=(x[i], y_top), ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=color)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Micro F1')
    ax.set_title('Fair Comparison: Best Adaptive vs. Closest Fixed WMC (Matched Overlap)')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.legend()
    ax.set_ylim(0.30, 0.95)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fair_comparison.png'))
    plt.close()
    print('Saved: fair_comparison.png')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Overlap efficiency comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig_overlap_efficiency():
    eff = pd.read_csv(os.path.join(RESULTS, 'overlap_efficiency.csv'))
    labels = ['WMC_0.10', 'WMC_0.20', 'WMC_0.30', 'entropy_l050', 'margin_l050', 'hybrid_l050']
    names = ['WMC=0.10', 'WMC=0.20', 'WMC=0.30',
             'Entropy ($\\lambda$=0.5)', 'Margin ($\\lambda$=0.5)', 'Hybrid ($\\lambda$=0.5)']
    colors = ['#b0b0b0', '#a0a0a0', '#909090', '#e06666', '#93c47d', '#c9a0dc']

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(DATASETS))
    w = 0.13

    for j, (lab, name, color) in enumerate(zip(labels, names, colors)):
        vals = []
        for ds in DATASETS:
            row = eff[(eff['label'] == lab) & (eff['dataset'] == ds)]
            vals.append(row['efficiency'].iloc[0] if len(row) else np.nan)
        offset = (j - 2.5) * w
        ax.bar(x + offset, vals, w, label=name, color=color,
               edgecolor='black', linewidth=0.5)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('F1 Gain per Unit Overlap Increase')
    ax.set_title('Overlap Efficiency: F1 Gain per Unit Overlap Increase (micro F1)')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'overlap_efficiency.png'))
    plt.close()
    print('Saved: overlap_efficiency.png')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: Statistical significance heatmap (Margin-Adaptive vs fixed WMC)
# ═══════════════════════════════════════════════════════════════════════════
def fig_significance_heatmap(df):
    fixed_wmcs = [0.10, 0.20, 0.30, 0.40, 0.50]
    labels = ['WMC_0.10', 'WMC_0.20', 'WMC_0.30', 'WMC_0.40', 'WMC_0.50']

    diff_matrix = np.zeros((len(DATASETS), len(fixed_wmcs)))
    for j, ds in enumerate(DATASETS):
        adapt = get_micro('margin_l050', df, ds)
        for i, lab in enumerate(labels):
            diff_matrix[j, i] = adapt - get_micro(lab, df, ds)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(diff_matrix, cmap='RdYlGn', vmin=-0.08, vmax=0.08, aspect='auto')
    ax.set_xticks(range(len(fixed_wmcs)))
    ax.set_xticklabels([f'WMC={w}' for w in fixed_wmcs])
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASETS)
    ax.set_title('Micro-F1 Difference: Margin-Adaptive vs. Fixed WMC')
    ax.set_xlabel('Fixed WMC Baseline')

    for i in range(len(fixed_wmcs)):
        for j in range(len(DATASETS)):
            v = diff_matrix[j, i]
            color = 'white' if abs(v) > 0.04 else 'black'
            ax.text(i, j, f'{v:+.3f}', ha='center', va='center', fontsize=9,
                    color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('$\\Delta$F1 (Adaptive $-$ Fixed)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'significance_heatmap.png'))
    plt.close()
    print('Saved: significance_heatmap.png')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8: Entropy gain vs entropy variance (scatter)
# ═══════════════════════════════════════════════════════════════════════════
def fig_gain_vs_variance(df):
    ds_comp = pd.read_csv(os.path.join(RESULTS, 'dataset_comparison.csv'))
    ds_comp = ds_comp.set_index('dataset').reindex(DATASETS)

    entropy_var = ds_comp['entropy_var'].values
    # Gain: best adaptive (micro) vs default WMC=0.30
    gains = []
    for ds in DATASETS:
        adapt = max(get_micro('margin_l050', df, ds), get_micro('hybrid_l050', df, ds),
                    get_micro('entropy_l050', df, ds))
        gains.append((adapt - get_micro('WMC_0.30', df, ds)) * 100)
    gains = np.array(gains)

    scatter_colors = ['#4a86c8', '#7faadb', '#93c47d', '#e06666', '#f9cb9c', '#b4a7d6']
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, ds in enumerate(DATASETS):
        ax.scatter(entropy_var[i], gains[i], s=120, c=scatter_colors[i],
                   edgecolors='black', linewidth=0.8, zorder=5)
        ax.annotate(ds, (entropy_var[i], gains[i]), textcoords='offset points',
                    xytext=(8, 5), fontsize=10, fontweight='bold')

    z = np.polyfit(entropy_var, gains, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(entropy_var) * 0.8, max(entropy_var) * 1.1, 100)
    ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.5, linewidth=1,
            label=f'Trend (slope={z[0]:.1f})')

    ax.axhline(y=0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Entropy Variance ($\\sigma^2$)')
    ax.set_ylabel('Micro-F1 Gain vs. WMC=0.30 (%)')
    ax.set_title('Performance Gain vs. Membership Entropy Variance')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gain_vs_variance.png'))
    plt.close()
    print('Saved: gain_vs_variance.png')


def copy_ambiguity_plots():
    """Copy the fresh ambiguity plots from results/plots into the paper figures dir."""
    plot_dir = os.path.join(RESULTS, 'plots')
    for fname in ['entropy_vs_clusters.png', 'overlap_vs_entropy.png',
                  'entropy_histograms.png', 'ambiguity_histograms.png',
                  'dataset_comparison.png', 'ambiguity_vs_overlap.png']:
        src = os.path.join(plot_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUTPUT_DIR, fname))
            print(f'Copied: {fname}')


def main():
    df = load_fixed_summary()
    fig_main_results(df)
    fig_lambda_sensitivity('entropy', 'entropy_lambda_sensitivity.png',
                           'Entropy-Adaptive WMC: Sensitivity to $\\lambda$')
    fig_lambda_sensitivity('margin', 'margin_lambda_sensitivity.png',
                           'Margin-Adaptive WMC: Sensitivity to $\\lambda$')
    fig_lambda_sensitivity('hybrid', 'hybrid_lambda_sensitivity.png',
                           'Hybrid-Adaptive WMC: Sensitivity to $\\lambda$')
    fig_fair_comparison()
    fig_overlap_efficiency()
    fig_significance_heatmap(df)
    fig_gain_vs_variance(df)
    copy_ambiguity_plots()
    print('\nAll plots generated successfully!')


if __name__ == '__main__':
    main()
