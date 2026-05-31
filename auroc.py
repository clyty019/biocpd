#!/usr/bin/env python
# coding: utf-8
"""Built-in AUROC validation for detected cliff points."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score


def compute_auroc(expr_matrix, pseudotime, cliff_t, delta, n_null=200,
                  random_seed=42):
    """Compute AUROC and null-distribution p-value for a single cliff point.

    Parameters
    ----------
    expr_matrix : np.ndarray
        Expression matrix (n_cells, n_genes).
    pseudotime : np.ndarray
        Pseudotime vector.
    cliff_t : float
        Candidate cliff point pseudotime.
    delta : float
        Half-width of the local neighborhood.
    n_null : int
        Number of null permutations.
    random_seed : int
        Random seed.

    Returns
    -------
    dict or None
    """
    rng = np.random.RandomState(random_seed)

    pre_mask = (pseudotime >= cliff_t - delta) & (pseudotime <= cliff_t)
    post_mask = (pseudotime > cliff_t) & (pseudotime <= cliff_t + delta)
    n_pre, n_post = pre_mask.sum(), post_mask.sum()
    if n_pre < 10 or n_post < 10:
        return None

    X_real = np.vstack([expr_matrix[pre_mask], expr_matrix[post_mask]])
    y_real = np.hstack([np.zeros(n_pre), np.ones(n_post)])

    clf = LogisticRegression(penalty='l2', C=1.0, max_iter=500,
                             random_state=random_seed, solver='lbfgs')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    scores = cross_val_score(clf, X_real, y_real, cv=cv, scoring='roc_auc')
    auroc = float(scores.mean())

    # Null distribution
    null_aurocs = []
    vmin, vmax = pseudotime.min() + delta, pseudotime.max() - delta
    for _ in range(n_null):
        rand_t = rng.uniform(vmin, vmax)
        if abs(rand_t - cliff_t) < 4 * delta:
            continue
        pre_m = (pseudotime >= rand_t - delta) & (pseudotime <= rand_t)
        post_m = (pseudotime > rand_t) & (pseudotime <= rand_t + delta)
        if pre_m.sum() < 10 or post_m.sum() < 10:
            continue
        X_null = np.vstack([expr_matrix[pre_m], expr_matrix[post_m]])
        y_null = np.hstack([np.zeros(pre_m.sum()), np.ones(post_m.sum())])
        n_tot = len(y_null)
        n_train, n_test = int(n_tot * 0.8), n_tot - int(n_tot * 0.8)
        if n_test < 2:
            continue
        idx = np.arange(n_tot); rng.shuffle(idx)
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        clf_null = LogisticRegression(penalty='l2', C=1.0, max_iter=500,
                                      random_state=random_seed, solver='lbfgs')
        clf_null.fit(X_null[train_idx], y_null[train_idx])
        y_pred = clf_null.predict_proba(X_null[test_idx])[:, 1]
        null_aurocs.append(float(roc_auc_score(y_null[test_idx], y_pred)))

    null_aurocs = np.array(null_aurocs)
    p_value = float((null_aurocs >= auroc).mean()) if len(null_aurocs) > 0 else 1.0

    grade = 'noise'
    if auroc >= 0.8:   grade = 'strong'
    elif auroc >= 0.7: grade = 'moderate'
    elif auroc >= 0.6: grade = 'weak'

    return {
        'pseudotime': float(cliff_t),
        'n_pre': int(n_pre),
        'n_post': int(n_post),
        'auroc': auroc,
        'null_mean': float(null_aurocs.mean()) if len(null_aurocs) > 0 else 0.5,
        'null_sd': float(null_aurocs.std()) if len(null_aurocs) > 0 else 0.0,
        'p_value': p_value,
        'grade': grade,
    }


def evaluate_peaks(peak_report, expr_matrix, pseudotime, delta,
                   n_null=200, random_seed=42):
    """Compute AUROC for every peak in the report.

    Parameters
    ----------
    peak_report : pd.DataFrame
        Detected peaks with column ``t``.
    expr_matrix : np.ndarray
    pseudotime : np.ndarray
    delta : float
    n_null : int
    random_seed : int

    Returns
    -------
    pd.DataFrame
        Peak report with AUROC columns appended.
    """
    if peak_report is None or peak_report.empty:
        return peak_report

    auroc_results = []
    for _, row in peak_report.iterrows():
        r = compute_auroc(expr_matrix, pseudotime, float(row['t']),
                          delta, n_null=n_null, random_seed=random_seed)
        if r is not None:
            auroc_results.append(r)

    if not auroc_results:
        peak_report['auroc'] = np.nan
        peak_report['p_value'] = np.nan
        peak_report['grade'] = 'N/A'
        peak_report['overall_auroc'] = 'N/A'
        return peak_report

    df_auroc = pd.DataFrame(auroc_results)
    df_auroc = df_auroc.rename(columns={'pseudotime': 't'})

    # Merge on t (closest match)
    peak_report = peak_report.merge(
        df_auroc[['t', 'auroc', 'p_value', 'grade']], on='t', how='left')

    overall = float(df_auroc['auroc'].mean())
    peak_report['overall_auroc'] = overall

    return peak_report
