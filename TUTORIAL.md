# Bio-CPD Tutorial

This tutorial walks through the complete Bio-CPD workflow.
---

## Table of Contents

1. [Setup and Data Preparation](#1-setup-and-data-preparation)
2. [Running Bio-CPD](#2-running-bio-cpd)
3. [Interpreting Built-in AUROC Results](#3-interpreting-built-in-auroc-results)
4. [Interpreting Results](#6-interpreting-results)
5. [Complete Example Script](#7-complete-example-script)

---

## 1. Setup and Data Preparation

### Installation

```bash
pip install biocpd
```

### Required imports

```python
import os, json
import numpy as np
import pandas as pd
import anndata as ad
import biocpd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks

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
adata = ad.read_h5ad("scRNA.h5ad")
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
 Rank         t  Confidence
    1 17.701793    0.826668
    2  9.620540    0.599050
    3  3.078573    0.200562
=============================================

Computing AUROC for each cliff point (delta=10% span, n_null=200)...
  Overall AUROC: 0.8421
 Rank         t  Confidence   auroc     p_value     grade
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
# Index(['Rank', 't', 'Confidence', 'auroc', 'p_value', 'grade', 'overall_auroc'])

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
      "confidence": 0.827,
      "auroc": 0.9875,
      "p_value": 0.0101,
      "grade": "strong"
    },
    ...
  ]
}
```

---

## 4. Interpreting Results

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

The p-value represents the fraction of null-position AUROCs that meet or exceed the observed AUROC. Use it together with AUROC grade and CP Span Ratio.

---

## 5. Complete Example Script

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
    print(peaks[["Rank", "t", "Confidence", "auroc", "grade"]].to_string(index=False))
    print(f"\nOverall AUROC: {peaks['overall_auroc'].iloc[0]:.4f}")
    print(f"Barcodes saved to ./results/tipping_peak_*_barcodes.csv")
    print(f"Full results saved to ./results/BioCPD_results.json")
else:
    print("No cliff points detected.")
```
