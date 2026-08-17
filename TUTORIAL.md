# Bio-CPD Tutorial

This tutorial walks through the complete Bio-CPD workflow: cliff point detection with built-in AUROC validation and permutation-based significance testing.

---

## Table of Contents

1. [Setup and Data Preparation](#1-setup-and-data-preparation)
2. [Running Bio-CPD](#2-running-bio-cpd)
3. [Interpreting Built-in AUROC Results](#3-interpreting-built-in-auroc-results)
4. [Peak-level Permutation Significance](#4-peak-level-permutation-significance)
5. [Interpreting Results](#5-interpreting-results)
6. [Complete Example Script](#6-complete-example-script)

---

## 1. Setup and Data Preparation

### Installation

Clone the repository and `import biocpd` directly from the repo directory.

### Required imports

```python
import numpy as np
import pandas as pd
import anndata as ad
import biocpd

import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
```

### Load data

Your data must consist of:

1. **Expression matrix**: an AnnData (`.h5ad`) object with cells &times; genes
2. **Pseudotime metadata**: a CSV file with cell IDs as the index and a `Pseudotime` column

```python
adata = ad.read_h5ad("scRNAepith.h5ad")
pseudo_df = pd.read_csv("pseudotime_meta.csv", index_col=0)
pseudotime = pseudo_df["Pseudotime"].values.astype(np.float64)

print(f"Cells: {adata.shape[0]}, Genes: {adata.shape[1]}")
print(f"Pseudotime range: [{pseudotime.min():.3f}, {pseudotime.max():.3f}]")
```

---

## 2. Running Bio-CPD

```python
adata.obs["Pseudotime"] = pseudotime

peak_report = biocpd.bio_cpd_pipeline(
    adata,
    time_col="Pseudotime",
    prominence=0.15,
    distance=3,
    step_ratio=0.02,
    safe_margin_ratio=0.02,
    min_cells_ratio=0.025,
    auroc_n_null=200,            # built-in AUROC validation
    auroc_delta_ratio=0.10,
    random_seed=SEED,
    do_plot=True,
    save_plot="BioCPD_cliff.pdf",
    save_dir="./results",
)
```

**Expected output:**

```
=============================================
Bio-CPD Cliff Point Detection Report
=============================================
Adaptive weights: Bhatt=0.44 | Wass=0.56
---------------------------------------------
 Rank         t      CPI
    1 17.701793    0.826668
    2  9.620540    0.599050
    3  3.078573    0.200562
=============================================

Computing AUROC for each cliff point (delta=10% span, n_null=200)...
  Overall AUROC: 0.8421
 Rank         t      CPI   auroc     p_value     grade
    1 17.701793    0.826668  0.9875  1.01e-02     strong
    2  9.620540    0.599050  0.8033  1.00e+00     strong
    3  3.078573    0.200562  0.7355  5.66e-01   moderate
  Results saved to ./results/BioCPD_results.json
```

The entropy weights (`Bhatt=0.44 | Wass=0.56`) are automatically adapted per dataset. AUROC is computed for every detected peak using logistic regression with 5-fold stratified cross-validation against 200 random null positions.

---

## 3. Interpreting Built-in AUROC Results

The returned `peak_report` DataFrame contains everything you need:

```python
print(peak_report.columns)
# Index(['Rank', 't', 'CPI', 'auroc', 'p_value', 'grade', 'overall_auroc'])

for _, row in peak_report.iterrows():
    print(f"Rank {int(row['Rank'])}: t={row['t']:.4f}, "
          f"AUROC={row['auroc']:.4f}, p={row['p_value']:.2e}, "
          f"grade={row['grade']}")

print(f"\nOverall AUROC: {peak_report['overall_auroc'].iloc[0]:.4f}")
```

The `BioCPD_results.json` file contains the same information in machine-readable format:

```json
{
  "method_id": "BioCPD",
  "n_cliff_points": 3,
  "overall_auroc": 0.8421,
  "auroc_results": [
    {
      "rank": 1,
      "pseudotime": 17.702,
      "cpi": 0.827,
      "auroc": 0.9875,
      "p_value": 0.0101,
      "grade": "strong"
    },
    ...
  ]
}
```

## 4. Peak-level Permutation Significance

The built-in AUROC p-value is a *random-position* null: it asks how often a
classifier at a random trajectory position achieves the same separation. The
revised manuscript instead assigns significance with a **peak-level permutation
null** (`biocpd.null.permutation_peak_pvalues`): for each detected peak, the
*complete* detection pipeline is re-run on shuffled pseudotime, and the peak's
empirical p-value counts how often a null peak within its local window is at
least as strong. This is the recommended significance test for publication.

```python
import numpy as np
from biocpd.null import permutation_peak_pvalues

X = np.asarray(adata.X.toarray())
pseudo = adata.obs["Pseudotime"].values.astype(float)

sig = permutation_peak_pvalues(
    X, pseudo,
    n_permutations=100,   # B permutations
    n_procs=8,            # fork pool (X is shared copy-on-write)
)
print(sig)
```

The returned `DataFrame` has one row per detected peak:

| Column | Description |
|---|---|
| `rank` | Peak rank (1 = strongest CPI) |
| `t` | Pseudotime position of the cliff point |
| `cpi` | Fixed-reference CPI statistic `S_j` used for testing |
| `abs_bd` | Absolute Bhattacharyya distance at the peak |
| `p_nominal` | Empirical permutation p-value `(1 + #{T_bj >= S_j}) / (B + 1)` |
| `p_adj` | Benjamini-Hochberg FDR across the dataset's peaks |
| `status` | `confirmed` if `p_adj < 0.05`, else `exploratory` |

All detection parameters (prominence, distance, safe margin, step, min cells)
are shared with the main pipeline and fixed to the manuscript defaults, so the
peak set is identical to `bio_cpd_pipeline()`.

---

## 5. Interpreting Results

### CP Span Ratio

A critical quality metric is the **CP Span Ratio**:

```
span_ratio = (max_cp_t - min_cp_t) / total_pseudotime_span
```

- **High span ratio (&ge;50%)**: CPs are distributed across the trajectory &mdash; biologically meaningful
- **Low span ratio (<10%)**: CPs cluster at one end &mdash; likely noise or boundary artifacts

### AUROC Grade Interpretation

| Grade | AUROC Range | Interpretation |
|---|---|---|
| strong | &ge; 0.80 | Clear separation between pre- and post-CP cell states |
| moderate | 0.70 – 0.80 | Detectable separation, moderate effect size |
| weak | 0.60 – 0.70 | Marginal separation, interpret with caution |
| noise | < 0.60 | No meaningful separation at this position |

### p-value Interpretation

The p-value represents the fraction of null-position AUROCs that meet or exceed the observed AUROC. Use it together with AUROC grade and CP Span Ratio — **never rely on p-values alone**.

---

## 6. Complete Example Script

A minimal but complete pipeline:

```python
#!/usr/bin/env python3
"""Minimal Bio-CPD example."""
import numpy as np, pandas as pd
import anndata as ad
import biocpd

# 1. Load
adata = ad.read_h5ad("expression.h5ad")
pseudo = pd.read_csv("pseudotime.csv", index_col=0)
adata.obs["Pseudotime"] = pseudo["Pseudotime"]

# 2. Run
peaks = biocpd.bio_cpd_pipeline(
    adata,
    time_col="Pseudotime",
    prominence=0.15,
    distance=3,
    auroc_n_null=200,
    random_seed=42,
    do_plot=True,
    save_plot="cliff_points.pdf",
    save_dir="./results",
)

# 3. Inspect
if peaks is not None and not peaks.empty:
    print(f"\nDetected {len(peaks)} cliff points:")
    print(peaks[["Rank", "t", "CPI", "auroc", "grade"]].to_string(index=False))
    print(f"\nOverall AUROC: {peaks['overall_auroc'].iloc[0]:.4f}")
    print(f"Barcodes saved to ./results/cliff_point_*_barcodes.csv")
    print(f"Full results saved to ./results/BioCPD_results.json")
else:
    print("No cliff points detected.")
```

---

## References

- Bhattacharyya, A. (1943). On a measure of divergence between two statistical populations. *Bull. Calcutta Math. Soc.*
