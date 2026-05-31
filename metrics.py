#!/usr/bin/env python
# coding: utf-8
"""Distance computation and entropy weight integration"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance


def calc_bhattacharyya_distance(X_a, X_b, eps=1e-4):
    """Bhattacharyya distance calculation"""
    dim = X_a.shape[1]
    mu1, cov1 = np.mean(X_a, axis=0), np.cov(X_a, rowvar=False) + np.eye(dim) * eps
    mu2, cov2 = np.mean(X_b, axis=0), np.cov(X_b, rowvar=False) + np.eye(dim) * eps
    sig_pool = (cov1 + cov2) / 2.0
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    _, logdet_pool = np.linalg.slogdet(sig_pool)
    diff = mu1 - mu2
    try:
        inv_sig_pool = np.linalg.inv(sig_pool)
        term1 = 0.125 * np.dot(diff.T, np.dot(inv_sig_pool, diff))
        term2 = 0.5 * logdet_pool - 0.25 * logdet1 - 0.25 * logdet2
        bd = term1 + term2
        return bd if bd > 0 else 0.0
    except:
        return np.nan


def calc_wasserstein(X_a, X_b):
    """Wasserstein distance calculation"""
    return np.mean([wasserstein_distance(X_a[:, i], X_b[:, i]) for i in range(X_a.shape[1])])



def get_entropy_weights(scores_matrix):
    """
    The entropy weight method calculates the weight of each indicator.

    Parameters
    ----
    scores_matrix : np.ndarray
        A score matrix with the shape (n_windows, n_metrics).

    Returns
    ----
    norm_scores : np.ndarray
        The normalized score matrix.
    weights : np.ndarray
        The weight vector for each indicator.
     """
    scaler = MinMaxScaler()
    norm_scores = scaler.fit_transform(scores_matrix)
    X = norm_scores + 1e-6
    p = X / X.sum(axis=0)
    n_windows = X.shape[0]
    if n_windows <= 1:
        return norm_scores, np.array([0.5, 0.5])
    E = -np.sum(p * np.log(p), axis=0) / np.log(n_windows)
    D = 1 - E
    weights = D / D.sum()
    return norm_scores, weights
