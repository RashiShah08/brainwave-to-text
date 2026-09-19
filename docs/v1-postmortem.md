# v1 post-mortem

The original scripts were removed from the working tree once v2 had replaced
every one of them. They were never runnable (every path was hardcoded to a
directory that no longer existed) and nothing imported them. They remain in git
history; `git log --diff-filter=D --stat -- legacy/` finds the commit that
removed them, and `git show <commit>^:legacy/<file>` prints any one of them.

What is kept here is the reasoning, because the defects in v1 are the reason v2
is shaped the way it is.

## Why they were replaced

| File | Problem |
|---|---|
| `extract_features_to_csv.py` | Assigned label 0 to resting baseline runs, to left-fist trials, **and** to both-fists trials; label 1 to right-fist **and** both-feet. The two classes had no coherent meaning. Also zeroed ICA components 0–1 by index (FastICA component order is arbitrary), upsampled 160→256 Hz for no benefit, used only the first 1 s of each 4.1 s trial, and never recorded subject identity. |
| `clean_data.py` | Dropped `file_source`, destroying the only column that carried subject identity — which is what made subject-wise validation impossible downstream. Its per-file 3σ "outlier" filter flagged a row if **any** of 576 features deviated, discarding 48% of the data (32,967 → 17,147 rows). |
| `train_dataset.py` | Plain stratified random split with no subject grouping, so adjacent epochs from one recording landed on both sides of the split. |
| `test_model.py` | Evaluated on the full `cleaned_features.csv`, ~80% of which was the training set. This produced the widely-quoted 94.3% figure. |
| `app.py` | Reimplemented feature extraction by hand, and had drifted from the training version (no ICA, fixed first-second window). Ran with `debug=True`, had no upload size limit, and displayed the classes as "Left"/"Right", which was wrong for most of class 0. |
| `preprocess_data.py` | Dead code. Hand-rolled `.edf.event` parser; wrote `X_all.npy`/`y_all.npy`, which nothing read and which were never generated. |
| `check_model.py` | Dead code. Loaded `models/shallowconvnet_best.h5`, which does not exist. |

## The numbers

v1's saved confusion matrices were unambiguous. On the genuinely held-out 20%:
**72.07%** accuracy against a **67.93%** majority-class baseline, with **28.5%**
recall on class 1. On the full dataset: 94.30%.

Subtracting the two shows 19 errors across 13,717 training rows (99.86%) versus
958 across 3,430 held-out rows (72.07%) — a 28-point generalisation gap.

## Superseded data files

v1's outputs — `processed_data/` (~700 MB of feature CSVs) and `models/` (a
79 MB pickle and the two confusion-matrix figures quoted above) — were never
committed and have since been deleted. Nothing in v2 read them. The figures
above are the record of what they contained.
