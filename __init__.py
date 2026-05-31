#!/usr/bin/env python
# coding: utf-8
"""
Bio-CPD: Biological Cliff Point Detection in single-cell trajectories
in single-cell trajectories.

Exposed core function: ``bio_cpd_pipeline``
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .utils import set_seed
from .data import load_and_validate
from .analysis import sliding_window_scan, compute_combined_score, find_tipping_peaks
from .extract_cell import extract_tipping_cells
from .visual import plot_cpi_curve
from .auroc import evaluate_peaks as _evaluate_peaks


def bio_cpd_pipeline(
    adata,
    time_col='Pseudotime',
    step_ratio=0.02,
    safe_margin_ratio=0.02,
    min_cells_ratio=0.025,
    prominence=0.15,
    distance=3,
    extract_ratio=0.04,
    feature_type='raw',
    n_components=50,
    auroc_n_null=200,
    auroc_delta_ratio=0.10,
    save_dir='./',
    random_seed=42,
    do_plot=True,
    save_plot=None,
    colors=None,
    alpha=0.2,
    figsize=(12, 6),
):
    """Detect cliff points along a single-cell pseudotime trajectory.

    Scans a sliding window, computes a combined Cliff Point Index (CPI)
    from Bhattacharyya distance and Wasserstein distance with entropy-based
    adaptive weighting, detects peaks, and optionally validates each peak
    via AUROC against a null distribution.

    Parameters
    ----------
    adata : AnnData
        Single-cell expression data. Raw expression is used directly.
    time_col : str
        Column in ``adata.obs`` with continuous trajectory values.
    step_ratio : float
        Window step size as fraction of total pseudotime span.
    safe_margin_ratio : float
        Safety margin at trajectory boundaries (fraction of total span).
    min_cells_ratio : float
        Minimum cells per window as fraction of total (floored at 10).
    prominence : float
        Minimum peak prominence for detection.
    distance : int
        Minimum distance between peaks (in scanning points).
    extract_ratio : float
        Cell extraction window radius as fraction of pseudotime span.
    auroc_n_null : int
        Number of null permutations for AUROC. Set to 0 to skip.
    auroc_delta_ratio : float
        AUROC local-neighborhood half-width as fraction of pseudotime span.
    feature_type : str
        Feature type: 'raw' for raw expression, 'pca' for PCA-reduced features.
        Default is 'raw'.
    n_components : int
        Number of PCA components when feature_type='pca'. Default is 50.
    save_dir : str
        Output directory for barcode CSVs and results JSON.
    random_seed : int
        Random seed for reproducibility.
    do_plot : bool
        Generate CPI curve plot.
    save_plot : str or None
        Save plot to this path as PDF.
    colors : list
        Colors for individual metric curves.
    alpha : float
        Transparency of individual metric curves.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    peak_report : pd.DataFrame
        Columns: ``Rank``, ``t``, ``Confidence``.
        If ``auroc_n_null > 0``, also includes ``auroc``, ``p_value``,
        ``grade``, and ``overall_auroc``.
        Returns None if no valid cliff points are found.
    """
    set_seed(random_seed)

    # 1. Data
    time_vec = load_and_validate(adata, time_col=time_col)
    total_span = time_vec.max() - time_vec.min()

    # 2. Feature extraction
    X_data = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    if feature_type == 'raw':
        print("Using raw expression features (no dimensionality reduction).")
        features = X_data
    elif feature_type == 'pca':
        print(f"Running PCA (n_components={n_components})...")
        pca = PCA(n_components=n_components, random_state=random_seed)
        features = pca.fit_transform(X_data)
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        raise ValueError(f"Unknown feature_type '{feature_type}'. "
                         f"Expected 'raw' or 'pca'.")

    # 2. Sliding window scan
    print("Scanning trajectory and calculating metrics...")
    df_res = sliding_window_scan(
        features, time_vec,
        step_ratio=step_ratio,
        safe_margin_ratio=safe_margin_ratio,
        min_cells_ratio=min_cells_ratio,
        random_state=random_seed,
    )

    if len(df_res) <= 5:
        print("Too few valid scanning points to determine cliff points.")
        return None

    # 3. Combined CPI
    df_res, norm_scores, weights = compute_combined_score(df_res)

    # 4. Peak detection
    peak_report = find_tipping_peaks(df_res, prominence=prominence,
                                     distance=distance)

    # Print report
    print("\n" + "=" * 45)
    print("Bio-CPD Cliff Point Detection Report")
    print("=" * 45)
    print(f"Adaptive weights: Bhatt={weights[0]:.2f} | Wass={weights[1]:.2f}")
    print("-" * 45)
    if not peak_report.empty:
        print(peak_report[['Rank', 't', 'Confidence']].to_string(index=False))
    else:
        print("No significant cliff points detected.")
    print("=" * 45)

    if peak_report.empty:
        return None

    # 5. AUROC validation (built-in)
    if auroc_n_null > 0:
        print("\nComputing AUROC for each cliff point "
              f"(delta={auroc_delta_ratio*100:.0f}% span, "
              f"n_null={auroc_n_null})...")
        delta = total_span * auroc_delta_ratio
        expr_matrix = np.asarray(features, dtype=np.float64)
        pseudotime_arr = np.asarray(time_vec, dtype=np.float64)

        peak_report = _evaluate_peaks(
            peak_report, expr_matrix, pseudotime_arr, delta,
            n_null=auroc_n_null, random_seed=random_seed,
        )

        overall = peak_report['overall_auroc'].iloc[0]
        print(f"  Overall AUROC: {overall:.4f}")
        cols = ['Rank', 't', 'Confidence', 'auroc', 'p_value', 'grade']
        print(peak_report[cols].to_string(index=False))

        # Save AUROC results JSON
        os.makedirs(save_dir, exist_ok=True)
        auroc_list = []
        for _, row in peak_report.iterrows():
            auroc_list.append({
                'rank': int(row['Rank']),
                'pseudotime': float(row['t']),
                'confidence': float(row['Confidence']),
                'auroc': float(row['auroc']) if not pd.isna(row['auroc']) else None,
                'p_value': float(row['p_value']) if not pd.isna(row['p_value']) else None,
                'grade': row['grade'],
            })
        result_json = {
            'method_id': 'BioCPD',
            'n_cliff_points': len(auroc_list),
            'overall_auroc': float(overall),
            'auroc_results': auroc_list,
        }
        json_path = os.path.join(save_dir, 'BioCPD_results.json')
        with open(json_path, 'w') as f:
            json.dump(result_json, f, indent=2, default=str)
        print(f"  Results saved to {json_path}")

    # 6. Plot
    if do_plot:
        plot_colors = colors if colors is not None else ['#3498db', '#e67e22']
        plot_cpi_curve(
            df_res, norm_scores, peak_report,
            metrics=['bhatt', 'wass'],
            colors=plot_colors, alpha=alpha, figsize=figsize,
            time_col=time_col, save_path=save_plot,
        )

    # 7. Extract tipping cells
    extract_tipping_cells(
        adata, peak_report,
        time_col=time_col, total_span=total_span,
        extract_ratio=extract_ratio, save_dir=save_dir,
    )

    return peak_report
