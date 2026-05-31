#!/usr/bin/env python
# coding: utf-8
"""Sliding Window Scan and Peak Detection"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .metrics import (
    calc_bhattacharyya_distance,
    calc_wasserstein,
    get_entropy_weights,
)


def sliding_window_scan(
    features,
    time_vec,
    step_ratio=0.02,
    safe_margin_ratio=0.02,
    min_cells_ratio=0.025,
    random_state=42,
):
    """Slide a window along pseudotime and compute per-window metrics.

    Parameters
    ----------
    features : np.ndarray
        Expression or feature matrix of shape (n_cells, n_features).
    time_vec : np.ndarray
        Pseudotime vector.
    step_ratio : float
        Step size as a fraction of total pseudotime span.
    safe_margin_ratio : float
        Safety margin at trajectory boundaries.
    min_cells_ratio : float
        Minimum cells per window (fraction of total, floored at 10).

    Returns
    -------
    pd.DataFrame
        Columns: ``t``, ``bhatt``, ``wass``.
    """
    total_cells = len(time_vec)
    total_span = time_vec.max() - time_vec.min()

    step_size = total_span * step_ratio
    SAFE_MARGIN = total_span * safe_margin_ratio
    MIN_CELLS = max(10, int(total_cells * min_cells_ratio))

    results = []
    for t in np.arange(time_vec.min() + SAFE_MARGIN,
                       time_vec.max() - SAFE_MARGIN, step_size):
        mask_a = time_vec < t
        mask_b = time_vec >= t
        if mask_a.sum() < MIN_CELLS or mask_b.sum() < MIN_CELLS:
            continue
        idx_a = np.where(mask_a)[0][np.argsort(time_vec[mask_a])][-MIN_CELLS:]
        idx_b = np.where(mask_b)[0][np.argsort(time_vec[mask_b])][:MIN_CELLS]
        X_a, X_b = features[idx_a], features[idx_b]

        try:
            d_b = calc_bhattacharyya_distance(X_a, X_b)
            d_w = calc_wasserstein(X_a, X_b)
            if not np.isnan(d_b):
                results.append({'t': t, 'bhatt': d_b, 'wass': d_w})
        except Exception:
            continue

    return pd.DataFrame(results)


def compute_combined_score(df_res):
    """Combine BD + WD into a single CPI curve via entropy weighting.

    Returns
    -------
    df_res : pd.DataFrame
        DataFrame with added ``combined_score`` column.
    norm_scores : np.ndarray
    weights : np.ndarray
    """
    metrics = ['bhatt', 'wass']
    norm_scores, weights = get_entropy_weights(df_res[metrics].values)
    df_res['combined_score'] = np.dot(norm_scores, weights)
    return df_res, norm_scores, weights


def find_tipping_peaks(df_res, prominence=0.15, distance=3):
    """Detect peaks on the CPI curve.

    Returns
    -------
    pd.DataFrame
        Columns: ``Rank``, ``t``, ``Confidence``.
    """
    peaks, _ = find_peaks(df_res['combined_score'],
                          prominence=prominence, distance=distance)
    peak_report = pd.DataFrame({
        't': df_res.loc[peaks, 't'].values,
        'Confidence': df_res.loc[peaks, 'combined_score'].values,
    }).sort_values('Confidence', ascending=False)
    peak_report['Rank'] = range(1, len(peak_report) + 1)
    return peak_report
