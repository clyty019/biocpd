#!/usr/bin/env python
# coding: utf-8
"""Data loading and preprocessing"""

import pandas as pd


def load_and_validate(adata, time_col='Pseudotime'):
    """
    Verify whether the adata object contains the time_col column and return the time vector.

    Parameters
    ----
    adata : AnnData
        The single-cell data object.
    time_col : str
        The column name for pseudotime or continuous trajectory (in adata.obs).

    Returns
    ----
    time_vec : np.ndarray
        The time vector.
     """
    if time_col not in adata.obs:
        raise ValueError(f"The column name '{time_col}' was not found in adata.obs. Please check the metadata.")
    return adata.obs[time_col].values
