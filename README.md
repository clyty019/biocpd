# Bio-CPD

**A Distributional Divergence-based Framework**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Bio-CPD detects **cliff points** (tipping points / critical transitions) along continuous single-cell trajectories such as pseudotime, developmental time, or aging. It scans a sliding window across the trajectory, computes a combined Cliff Point Index (CPI) from Bhattacharyya distance and Wasserstein distance with entropy-based adaptive weighting, identifies peaks, and validates each candidate via logistic-regression AUROC against null distributions — all in a single function call.

---

## Key Features

- **Single function**: `bio_cpd_pipeline()` runs the complete workflow — scan, detect, validate, export
- **Built-in AUROC validation**: every detected cliff point is statistically validated against null-position classifiers (200 permutations by default)
- **Adaptive metric weighting**: entropy-based weights automatically balance Bhattacharyya and Wasserstein distances per dataset
- **Cell extraction**: automatically exports cell barcodes within each tipping peak window for downstream analysis (RCTD, DEG, etc.)
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
  Cell barcode extraction + CPI plot
```

## Installation

```bash
git clone https://github.com/clyty019/biocpd.git
cd biocpd
pip install -e .
```

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
#    Rank         t  Confidence   auroc     p_value   grade  overall_auroc
# 0     1  6.961698    0.936849  0.9568  0.00e+00   strong         0.8790
# 1     2  3.586329    0.900728  0.8752  4.59e-01   strong         0.8790
# 2     3  8.227461    0.798741  0.7793  5.59e-01  moderate        0.8790
```

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
| `Rank` | Peak rank (1 = highest CPI confidence) |
| `t` | Pseudotime position of the cliff point |
| `Confidence` | CPI value at the peak |
| `auroc` | Logistic-regression AUROC (5-fold CV) |
| `p_value` | Fraction of null AUROCs &ge; observed AUROC |
| `grade` | `strong` (&ge;0.80), `moderate` (&ge;0.70), `weak` (&ge;0.60), `noise` (<0.60) |
| `overall_auroc` | Mean AUROC across all detected peaks |

### Low-level API

- `sliding_window_scan(features, time_vec, ...)` — returns per-window metric DataFrame
- `compute_combined_score(df_res)` — entropy-weighted CPI curve
- `find_tipping_peaks(df_res, ...)` — peak detection
- `calc_bhattacharyya_distance(X_a, X_b)` — Bhattacharyya distance
- `calc_wasserstein(X_a, X_b)` — mean Wasserstein distance
- `get_entropy_weights(scores_matrix)` — entropy-based adaptive weights
- `compute_auroc(expr_matrix, pseudotime, cliff_t, delta, ...)` — single-peak AUROC validation
- `evaluate_peaks(peak_report, expr_matrix, pseudotime, delta, ...)` — batch AUROC evaluation

## Output Files

After running `bio_cpd_pipeline`, the `save_dir` will contain:

| File | Description |
|---|---|
| `tipping_peak_N_barcodes.csv` | Cell barcodes within each tipping peak window |
| `BioCPD_results.json` | Full results including per-peak AUROC, p-values, grades |
| `cliff_points.pdf` (if `save_plot` set) | CPI curve with annotated peaks |

## Recommended Parameters

Based on extensive ablation studies across 6 datasets (simulated, periodontitis, DFU, gastric cancer, colorectal cancer, aging):

| Parameter | Default | Rationale |
|---|---|---|
| `prominence` | `0.15` | Good balance between sensitivity and false-positive control |
| `distance` | `3` | Allows closely-spaced genuine tipping points |
| `min_cells_ratio` | `0.025` | Adaptive window sizing preserves local structure |
| `auroc_n_null` | `200` | Sufficient for stable p-value estimates |
| `auroc_delta_ratio` | `0.10` | 10% of pseudotime span captures local neighborhood well |


## License

MIT License. See [LICENSE](LICENSE) for details.
