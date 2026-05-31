#!/usr/bin/env python
# coding: utf-8
"""Key cell extraction and preservation"""

import os
import pandas as pd


def extract_tipping_cells(adata, peak_report, time_col='Pseudotime', total_span=None,
                          extract_ratio=0.04, save_dir='./'):
    """
    Extract cells around detected peaks and save as CSV files.
    
    Parameters
    ----------
    adata : AnnData
        Single-cell data object.
    peak_report : pd.DataFrame
        DataFrame containing detected peaks (with `t` column).
    time_col : str
        Name of the pseudotime or trajectory column.
    total_span : float
        Total pseudotime span. If None, it will be automatically computed.
    extract_ratio : float
        Fraction of the total pseudotime span used as the extraction window radius.
    save_dir : str
        Directory path to save the CSV files.
     """
    if peak_report.empty:
        print("No valid peaks detected; export cannot be performed.")
        return

    if total_span is None:
        total_span = adata.obs[time_col].max() - adata.obs[time_col].min()

    os.makedirs(save_dir, exist_ok=True)

    peak_times = sorted(peak_report['t'].values)
    window_radius = total_span * extract_ratio
    used_cells = set()

    for i, t_peak in enumerate(peak_times, start=1):
        t_min = t_peak - window_radius
        t_max = t_peak + window_radius

        tipping_cells = adata.obs[
            (adata.obs[time_col] >= t_min) &
            (adata.obs[time_col] <= t_max) &
            (~adata.obs.index.isin(used_cells))
        ].index.tolist()

        used_cells.update(tipping_cells)

        tipping_name = f"tipping_peak_{i}"
        df_export = pd.DataFrame(tipping_cells, columns=["Barcode"])
        df_export['Tipping_Name'] = tipping_name
        df_export['Tipping_Peak'] = t_peak

        save_path = os.path.join(save_dir, f"{tipping_name}_barcodes.csv")
        df_export.to_csv(save_path, index=False)
        print(f"Peak {i} pseudotime: {t_peak:.3f}, number of extracted cells: {len(tipping_cells)} -> saved to {save_path}")
