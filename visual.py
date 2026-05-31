#!/usr/bin/env python
# coding: utf-8
"""Visualization"""

import os
import matplotlib.pyplot as plt


def plot_cpi_curve(df_res, norm_scores, peak_report, metrics=None,
                   colors=None, alpha=0.2, figsize=(12, 6), time_col='Pseudotime',
                   save_path=None, fmt=None):
    """
    Plot the Bio-CPD CPI (Cliff Point Index) curve with multiple peak annotations.
    
    Parameters
    ----------
    df_res : pd.DataFrame
        DataFrame containing columns ``t`` and ``combined_score``.
    norm_scores : np.ndarray
        Normalized scores for each metric.
    peak_report : pd.DataFrame
        DataFrame containing detected peaks with columns `Rank`, `t`, and `Confidence`.
    metrics : list
        List of metric names.
    colors : list
        Colors for the original metric curves.
    alpha : float
        Transparency of the original metric curves.
    figsize : tuple
        Figure size.
    time_col : str
        Label for the x-axis (typically pseudotime or trajectory coordinate).
"""
    if metrics is None:
        metrics = ['bhatt', 'wass']
    if colors is None:
        colors = ['#3498db', '#e67e22']

    plt.figure(figsize=figsize)

    # 1. Plot the unified comprehensive curve
    plt.plot(df_res['t'], df_res['combined_score'], color='black', lw=3,
             label='Unified CPI (Bio-CPD)')

    # 2. Plot background curves of individual raw metrics
    for i, m in enumerate(metrics):
        plt.plot(df_res['t'], norm_scores[:, i], color=colors[i], alpha=alpha,
                 ls=':', label=f'Raw {m}')

    # 3. Annotate all detected cliff points
    for _, row in peak_report.iterrows():
        color = 'red' if row['Rank'] == 1 else '#f39c12'
        plt.axvline(x=row['t'], color=color, ls='--', lw=2, alpha=0.7)
        plt.annotate(
            f"Rank {int(row['Rank'])}\nt={row['t']:.2f}",
            xy=(row['t'], row['Confidence']),
            xytext=(0, 10), textcoords='offset points',
            ha='center', va='bottom', color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.8)
        )

    plt.title('Bio-CPD Multi-Peak Detection', fontsize=14)
    plt.xlabel(time_col)
    plt.ylabel('Cliff Point Index')
    plt.legend(loc='upper right', ncol=2, fontsize=8)
    plt.grid(alpha=0.15)
    plt.tight_layout()
    if save_path:
        ext = fmt or os.path.splitext(save_path)[1].lstrip('.') or 'pdf'
        plt.savefig(save_path, dpi=300, format=ext, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
