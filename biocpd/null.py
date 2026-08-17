#!/usr/bin/env python
# coding: utf-8
"""Permutation null for cliff-point significance (two-layer design).

Peak-level permutation test that assigns an empirical p-value to every
detected cliff point while preserving the *complete* detection procedure:

* **Detection layer (unchanged)** — for each permutation of the pseudotime
  vector, the entire detection pipeline is re-run: sliding-window scan,
  per-curve min-max normalization of the two distances, entropy-based
  weighting, and peak detection with ``scipy.signal.find_peaks``.
* **Statistic layer (fixed reference)** — every peak, real or null, is scored
  with a *fixed-reference* Cliff Point Index: the Bhattacharyya and Wasserstein
  distances are min-max normalized with the real data's min/max and weighted by
  the real data's entropy weights. Per-curve normalization would map every
  curve's maximum to ~1.0 and give no permutation power, so a fixed reference
  is required to make peak statistics comparable across permutations.

For a real peak ``j`` at position ``t_j`` with statistic ``S_j``, each of the
``B`` permutations contributes one null statistic ``T_bj``: the strongest null
peak statistic whose position falls in the local window ``[t_j - d, t_j + d]``,
where ``d = distance * step_ratio * span`` is the half-width a detected peak
occupies given ``find_peaks(distance=distance)``; if no null peak falls in the
window, ``T_bj = 0``. The empirical p-value is

    p_j = (1 + #{b : T_bj >= S_j}) / (B + 1),

and Benjamini-Hochberg FDR is applied across the dataset's peaks;
``q < 0.05`` is labelled ``confirmed``, otherwise ``exploratory``.

Definitions referenced by reviewers (all fixed scan/peak parameters):

* ``prominence``          minimum peak prominence (``scipy.signal.find_peaks``)
* ``distance``            minimum inter-peak separation in scanning points;
                          physical half-width of a peak = distance * step_ratio * span
* ``safe_margin_ratio``   boundary exclusion: scan restricted to
                          [t_min + safe_margin_ratio*span, t_max - safe_margin_ratio*span]
* ``step_ratio``          window step as a fraction of the pseudotime span
* ``min_cells_ratio``     minimum cells per window (fraction of total, floored at 10)
* span ratio              local-window half-width ``d`` relative to the span
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from multiprocessing import get_context

from .analysis import sliding_window_scan
from .metrics import get_entropy_weights

# Scratch state for fork workers (set before Pool creation, read-only after;
# copy-on-write avoids pickling large expression matrices).
_FEATURES = None
_PSEUDOTIME = None
_SCAN_PARAMS = None
_REAL_SCALE = None
_REAL_WEIGHTS = None

REPORT_COLUMNS = ['rank', 't', 'cpi', 'abs_bd', 'p_nominal', 'p_adj', 'status']


def _fixed_cpi(bhatt, wass, scale, weights):
    """Fixed-reference CPI: BD/WD min-max normalized by ``scale``, entropy-weighted."""
    bd_min, bd_max, wd_min, wd_max = scale
    nb = (bhatt - bd_min) / (bd_max - bd_min)
    nw = (wass - wd_min) / (wd_max - wd_min)
    return weights[0] * nb + weights[1] * nw


def _scan(features, pseudotime, step_ratio, safe_margin_ratio,
          min_cells_ratio, random_state):
    df = sliding_window_scan(features, pseudotime, step_ratio=step_ratio,
                             safe_margin_ratio=safe_margin_ratio,
                             min_cells_ratio=min_cells_ratio,
                             random_state=random_state)
    if df is None or df.empty:
        return np.array([]), np.array([]), np.array([])
    return df['bhatt'].values, df['wass'].values, df['t'].values


def _detect_peaks(bhatt, wass, prominence, distance):
    """Detection-layer peak indices on the per-curve normalized CPI curve.

    Returns indices sorted by descending CPI (rank 1 = strongest).
    """
    norm, w = get_entropy_weights(np.column_stack([bhatt, wass]))
    cpi = norm @ w
    idx, _ = find_peaks(cpi, prominence=prominence, distance=distance)
    return idx[np.argsort(cpi[idx])[::-1]]


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR (monotone-adjusted)."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    out = np.empty(n)
    out[order] = q
    return out


def _empty_report():
    return pd.DataFrame(columns=REPORT_COLUMNS)


def _run_permutation(seed):
    """Complete detection on one shuffled pseudotime -> null peaks (t, statistic)."""
    pt_sh = np.random.RandomState(seed).permutation(_PSEUDOTIME.copy())
    bhatt, wass, t = _scan(_FEATURES, pt_sh,
                           step_ratio=_SCAN_PARAMS['step_ratio'],
                           safe_margin_ratio=_SCAN_PARAMS['safe_margin_ratio'],
                           min_cells_ratio=_SCAN_PARAMS['min_cells_ratio'],
                           random_state=_SCAN_PARAMS['random_state'])
    if len(t) == 0:
        return np.array([]), np.array([])
    idx = _detect_peaks(bhatt, wass, _SCAN_PARAMS['prominence'],
                        _SCAN_PARAMS['distance'])
    S = _fixed_cpi(bhatt[idx], wass[idx], _REAL_SCALE, _REAL_WEIGHTS)
    return t[idx], S


def _run_permutations(seeds, n_procs):
    if n_procs <= 1 or not hasattr(os, 'fork'):
        return [_run_permutation(s) for s in seeds]
    ctx = get_context('fork')
    with ctx.Pool(min(n_procs, len(seeds))) as pool:
        return pool.map(_run_permutation, seeds)


def permutation_peak_pvalues(
    features,
    pseudotime,
    prominence=0.15,
    distance=3,
    step_ratio=0.02,
    safe_margin_ratio=0.02,
    min_cells_ratio=0.025,
    n_permutations=100,
    random_state=42,
    n_procs=1,
):
    """Empirical peak-level permutation p-values for detected cliff points.

    Detects cliff points on the real pseudotime with the standard pipeline
    (sliding-window scan, per-curve normalization, entropy weights,
    ``find_peaks``), then tests each one against the permutation null defined in
    the module docstring. This is the significance framework of the revised
    manuscript; it supersedes the random-position AUROC null of
    :func:`biocpd.auroc.compute_auroc` (which over-states significance for
    shallow peaks).

    Parameters
    ----------
    features : np.ndarray
        Expression or feature matrix, shape (n_cells, n_genes).
    pseudotime : np.ndarray
        Pseudotime vector, shape (n_cells,).
    prominence : float
        Minimum peak prominence (default 0.15).
    distance : int
        Minimum inter-peak separation in scanning points (default 3; physical
        half-width = distance * step_ratio * span).
    step_ratio : float
        Window step as a fraction of pseudotime span (default 0.02).
    safe_margin_ratio : float
        Boundary-exclusion safety margin (default 0.02).
    min_cells_ratio : float
        Minimum cells per window as a fraction of total, floored at 10
        (default 0.025).
    n_permutations : int
        Number of pseudotime permutations B (default 100). Set to 0 to return
        the real peaks without p-values.
    random_state : int
        Random seed.
    n_procs : int
        Number of parallel worker processes (fork pool on POSIX; 1 = sequential).
        Expression matrices are shared copy-on-write, so this does not re-copy
        the data per worker.

    Returns
    -------
    pd.DataFrame
        One row per real peak, columns:
        ``rank``, ``t``, ``cpi`` (fixed-reference CPI statistic S_j),
        ``abs_bd`` (absolute Bhattacharyya distance at the peak),
        ``p_nominal`` (permutation p_j), ``p_adj`` (BH-FDR q),
        ``status`` ('confirmed' if q < 0.05 else 'exploratory').
    """
    global _FEATURES, _PSEUDOTIME, _SCAN_PARAMS, _REAL_SCALE, _REAL_WEIGHTS

    features = np.asarray(features)
    pseudotime = np.asarray(pseudotime, dtype=np.float64)

    bhatt, wass, t = _scan(features, pseudotime, step_ratio=step_ratio,
                           safe_margin_ratio=safe_margin_ratio,
                           min_cells_ratio=min_cells_ratio,
                           random_state=random_state)
    if len(t) == 0:
        return _empty_report()

    # Real-data fixed reference (statistic layer) and detection-layer peaks
    scale = (bhatt.min(), bhatt.max(), wass.min(), wass.max())
    _, weights = get_entropy_weights(np.column_stack([bhatt, wass]))
    idx = _detect_peaks(bhatt, wass, prominence, distance)
    if len(idx) == 0:
        return _empty_report()

    real_t = t[idx]
    S_j = _fixed_cpi(bhatt[idx], wass[idx], scale, weights)

    if n_permutations <= 0:
        return pd.DataFrame({
            'rank': np.arange(1, len(idx) + 1),
            't': real_t,
            'cpi': S_j,
            'abs_bd': bhatt[idx],
            'p_nominal': np.nan,
            'p_adj': np.nan,
            'status': 'n/a',
        })[REPORT_COLUMNS]

    seeds = [int(random_state) + 1000 + i for i in range(n_permutations)]
    _FEATURES = features
    _PSEUDOTIME = pseudotime
    _SCAN_PARAMS = dict(step_ratio=step_ratio, safe_margin_ratio=safe_margin_ratio,
                        min_cells_ratio=min_cells_ratio, random_state=random_state,
                        prominence=prominence, distance=distance)
    _REAL_SCALE = scale
    _REAL_WEIGHTS = weights
    null = _run_permutations(seeds, n_procs)

    span = float(pseudotime.max() - pseudotime.min())
    d = distance * step_ratio * span  # local-window half-width

    pvals = []
    for tj, Sj in zip(real_t, S_j):
        cnt = 0
        for tn, Sn in null:
            inwin = np.abs(tn - tj) <= d
            Tbj = float(Sn[inwin].max()) if inwin.any() else 0.0
            if Tbj >= Sj:
                cnt += 1
        pvals.append((1 + cnt) / (n_permutations + 1))
    pvals = np.array(pvals)
    padj = _bh_fdr(pvals)

    return pd.DataFrame({
        'rank': np.arange(1, len(idx) + 1),
        't': real_t,
        'cpi': S_j,
        'abs_bd': bhatt[idx],
        'p_nominal': pvals,
        'p_adj': padj,
        'status': np.where(padj < 0.05, 'confirmed', 'exploratory'),
    })[REPORT_COLUMNS]
