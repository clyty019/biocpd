# Bio-CPD

**Biological SwifT-identification of Pseudotime Cliff points in single-cell trajectories**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Bio-CPD detects **cliff point-associated transcriptional populations** (transcriptional change points) along continuous single-cell trajectories such as pseudotime, developmental time, or aging clocks. It scans a sliding window across the trajectory, computes a combined Cliff Point Index (CPI) from Bhattacharyya distance and Wasserstein distance with entropy-based adaptive weighting, identifies peaks, and validates each candidate via logistic-regression AUROC against null distributions — all in a single function call.

---

## Key Features

- **Single function**: `bio_cpd_pipeline()` runs the complete workflow — scan, detect, validate, export
- **No preprocessing required**: works directly on raw expression matrices — no PCA, dimensionality reduction, or smoothing needed
- **Built-in AUROC validation**: every detected cliff point is statistically validated against null-position classifiers (200 permutations by default)
- **Permutation null**: peak-level empirical p-values from the complete detection re-run on shuffled pseudotime — the significance framework of the revised manuscript
- **Grouped AUROC**: donor/sample-level validation with GroupKFold (no cell leakage between train and test)
- **Adaptive metric weighting**: entropy-based weights automatically balance Bhattacharyya and Wasserstein distances per dataset
- **Cell extraction**: automatically exports cell barcodes within each cliff point window for downstream analysis (RCTD, DEG, etc.)
- **Publication-ready plots**: CPI curve with multi-peak annotation exported as high-resolution PDF
- **Zero deep-learning dependencies**: pure NumPy/SciPy/scikit-learn stack

## Algorithm Overview

```
Pseudotime + Expression Matrix
        │
        ▼
  Sliding Window Scan (adaptive window sizing)
        │
        ▼
  Per-window: Bhattacharyya distance + Wasserstein distance
        │
        ▼
  Entropy-weighted CPI curve
        │
        ▼
  Peak detection → Ranked Cliff Points
        │
        ▼
  AUROC validation (5-fold CV + null permutations)  ← built-in
        │
        ▼
  Permutation null (optional) ← re-run full detection on shuffled
        │                        pseudotime → peak-level p-values + BH-FDR
        ▼
  Cell barcode extraction + CPI plot
```

## Getting Started

Clone the repository and use the package directly from the repo directory:

```bash
git clone https://github.com/clyty019/biocpd.git
cd biocpd
```

Then `import biocpd` in Python.

### Dependencies

- Python &ge; 3.9
- anndata, scanpy
- numpy, pandas, scipy, scikit-learn
- matplotlib

## Quick Start

```python
import biocpd
import anndata
import pandas as pd

# Load data
adata = anndata.read_h5ad("expression.h5ad")
pseudo = pd.read_csv("pseudotime.csv", index_col=0)
adata.obs["Pseudotime"] = pseudo["Pseudotime"]

# Run Bio-CPD
peak_report = biocpd.bio_cpd_pipeline(
    adata,
    time_col="Pseudotime",
    save_dir="./results",
    save_plot="cliff_points.pdf",
)

print(peak_report)
#    Rank         t      CPI   auroc     p_value   grade  overall_auroc
# 0     1  6.961698    0.936849  0.9568  0.00e+00   strong         0.8790
# 1     2  3.586329    0.900728  0.8752  4.59e-01   strong         0.8790
# 2     3  8.227461    0.798741  0.7793  5.59e-01  moderate        0.8790
```

### Peak-level permutation significance

The revised manuscript assigns significance with a **permutation null**: for
each detected peak, the *complete* detection pipeline is re-run on shuffled
pseudotime, and the peak's empirical p-value counts how often a null peak in its
local window is at least as strong. This supersedes the random-position AUROC
null for significance claims.

```python
import numpy as np
from biocpd.null import permutation_peak_pvalues

X = np.asarray(adata.X.toarray())
pseudo = adata.obs["Pseudotime"].values.astype(float)

sig = permutation_peak_pvalues(
    X, pseudo,
    n_permutations=100,   # B permutations
    n_procs=8,            # parallelize with a fork pool (shares X copy-on-write)
)

print(sig)
#    rank        t       cpi   abs_bd  p_nominal   p_adj     status
# 0     1  6.9617  0.936849  12.311    0.0099    0.0099   confirmed
# 1     2  3.5863  0.900728   8.141    0.0099    0.0099   confirmed
# 2     3  8.2275  0.798741   6.981    0.0495    0.0495   confirmed
```

Parameters that define the null (all fixed defaults match the manuscript):
`prominence=0.15`, `distance=3` (min inter-peak separation in scanning points;
physical half-width = `distance * step_ratio * span`), `safe_margin_ratio=0.02`
(boundary exclusion), `step_ratio=0.02`, `min_cells_ratio=0.025`. The local
window used for null matching is `[t_j - d, t_j + d]` with
`d = distance * step_ratio * span`.

## API Reference

### `bio_cpd_pipeline(adata, ...)`

Main entry point. Returns a `pd.DataFrame` of detected cliff points with AUROC validation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adata` | `AnnData` | required | Single-cell expression data |
| `time_col` | `str` | `"Pseudotime"` | Column in `adata.obs` with continuous trajectory values |
| `step_ratio` | `float` | `0.02` | Window step as fraction of total pseudotime span |
| `safe_margin_ratio` | `float` | `0.02` | Safety margin at trajectory boundaries |
| `min_cells_ratio` | `float` | `0.025` | Minimum cells per window (fraction of total, floored at 10) |
| `prominence` | `float` | `0.15` | Minimum peak prominence |
| `distance` | `int` | `3` | Minimum distance between peaks (in scanning points) |
| `extract_ratio` | `float` | `0.04` | Cell extraction window radius (fraction of span) |
| `auroc_n_null` | `int` | `200` | Null permutations for AUROC. Set to 0 to skip |
| `auroc_delta_ratio` | `float` | `0.10` | AUROC neighborhood half-width (fraction of span) |
| `save_dir` | `str` | `"./"` | Output directory |
| `random_seed` | `int` | `42` | Random seed |
| `do_plot` | `bool` | `True` | Generate CPI curve plot |
| `save_plot` | `str` | `None` | Save plot to this path as PDF |
| `colors` | `list` | `None` | Colors for individual metric curves |
| `alpha` | `float` | `0.2` | Transparency of individual metric curves |
| `figsize` | `tuple` | `(12, 6)` | Figure size in inches |

### Return value

`pd.DataFrame` with columns:

| Column | Description |
|---|---|
| `Rank` | Peak rank (1 = highest CPI) |
| `t` | Pseudotime position of the cliff point |
| `CPI` | Entropy-weighted Cliff Point Index at the peak |
| `auroc` | Logistic-regression AUROC (5-fold CV) |
| `p_value` | Fraction of null AUROCs &ge; observed AUROC |
| `grade` | `strong` (&ge;0.80), `moderate` (&ge;0.70), `weak` (&ge;0.60), `noise` (<0.60) |
| `overall_auroc` | Mean AUROC across all detected peaks |

### Low-level API

- `sliding_window_scan(features, time_vec, ...)` — returns per-window metric DataFrame
- `compute_combined_score(df_res)` — entropy-weighted CPI curve
- `find_cliff_peaks(df_res, ...)` — peak detection
- `calc_bhattacharyya_distance(X_a, X_b)` — Bhattacharyya distance
- `calc_wasserstein(X_a, X_b)` — mean Wasserstein distance
- `get_entropy_weights(scores_matrix)` — entropy-based adaptive weights
- `compute_auroc(expr_matrix, pseudotime, cliff_t, delta, ...)` — single-peak AUROC validation (random-position null)
- `compute_auroc_grouped(expr_matrix, pseudotime, cliff_t, delta, groups, ...)` — donor/sample-grouped AUROC (GroupKFold, no leakage)
- `evaluate_peaks(peak_report, expr_matrix, pseudotime, delta, ...)` — batch AUROC evaluation
- `permutation_peak_pvalues(features, pseudotime, ...)` — peak-level permutation p-values + BH-FDR

## Output Files

After running `bio_cpd_pipeline`, the `save_dir` will contain:

| File | Description |
|---|---|
| `cliff_point_N_barcodes.csv` | Cell barcodes within each cliff point window |
| `BioCPD_results.json` | Full results including per-peak AUROC, p-values, grades |
| `cliff_points.pdf` (if `save_plot` set) | CPI curve with annotated peaks |

## Recommended Parameters

| Parameter | Default | Rationale |
|---|---|---|
| `prominence` | `0.15` | Good balance between sensitivity and false-positive control |
| `distance` | `3` | Allows closely-spaced genuine cliff points |
| `min_cells_ratio` | `0.025` | Adaptive window sizing preserves local structure |
| `auroc_n_null` | `200` | Sufficient for stable p-value estimates |
| `auroc_delta_ratio` | `0.10` | 10% of pseudotime span captures local neighborhood well |

## Citation

If you use Bio-CPD in your research, please cite:

```
@software{biocpd2025,
  title     = {Bio-CPD: Biological SwifT-identification of Pseudotime Cliff points},
  author    = {Bio-CPD Team},
  year      = {2025},
  url       = {https://github.com/clyty019/biocpd},
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
