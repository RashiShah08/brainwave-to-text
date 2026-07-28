# Changelog

Notable changes to this project. Versions follow [semantic versioning](https://semver.org/).

Accuracy figures are stated with the protocol that produced them, because a
motor-imagery number without its protocol is meaningless.

## [3.0.0] — 2026-07-28

Adds neural decoders, per-user calibration, continuous decoding and a second
dataset. Several of the headline additions produced **negative** results; those
are documented rather than dropped.

### Added

- **Neural decoders** — EEGNet, ShallowConvNet and EEG-Conformer (`bwt.deep`),
  wrapped as scikit-learn classifiers so they reuse the existing
  subject-grouped cross-validation, artifact format and serving-parity tests
  with no parallel code path. EEGNet reaches **0.628 ± 0.027** cross-subject
  against `csp_lda`'s 0.611. PyTorch is an optional dependency.
- **Per-user calibration** (`bwt.calibration`, `bwt calibrate`) — fine-tuning a
  population model on a subject's own trials, with calibration curves that
  always evaluate on trials the adapted model never saw.
- **Continuous decoding** (`bwt.streaming`, `bwt stream`, `/live`) — sliding
  windows into a sequential-probability-ratio accumulator that commits only when
  the posterior crosses a threshold, plus a browser UI fed by an NDJSON
  streaming endpoint.
- **Second dataset** — BCI Competition IV-2a via MOABB, behind a dataset
  registry (`bwt.data.datasets`, `bwt datasets`). Session-holdout evaluation
  (`bwt.evaluation.session_holdout`) reproduces the competition protocol:
  **0.903** two-class and **0.809** four-class (κ 0.745) on subject A01.
- **Explainability** (`bwt.explain`, `bwt explain`) — ERD curves, CSP scalp
  topographies and a lateralisation index confirming the decoder tracks
  contralateral sensorimotor rhythm.
- **Resumable benchmarking** (`bwt.benchmarking`) — checkpoints every fold, so
  an interrupted run loses nothing and `--time-budget` stops cleanly at a fold
  boundary.
- `mi_move_vs_rest` task for asynchronous gating — the only task permitted to
  merge effectors, pinned by a test to exactly that one entry.
- MIT `LICENSE`, GitHub Actions CI, ruff configuration, `Makefile`,
  `.dockerignore`.

### Measured, and negative

- **Evidence accumulation is far weaker than an idealised simulation implies.**
  On real out-of-fold probabilities, 0.611 single-trial rises to **0.707** at a
  0.99 commit threshold, costing 12.8 trials per decision, with 28% of sequences
  never committing. A simulation assuming independent draws predicts 0.99+. Real
  errors are correlated within a subject, so for someone the model cannot decode,
  more evidence yields a *confidently wrong* answer.
- **Per-user calibration did not help on EEGMMIDB.** Refitting from scratch is
  worse than the population model at every budget (0.50–0.57 vs 0.625);
  fine-tuning EEGNet gained about a point with σ = 0.15 over 3 subjects. Likely
  this corpus's ~45-trials-per-subject ceiling rather than a failure of the
  method — BCI IV-2a supplies 288 per session — but it is what was measured.

### Changed

- `load_bundle` now takes a `dataset` argument; cache keys gained a dataset
  prefix, so pre-3.0 cache files are ignored and can be deleted.
- `RiemannianRecenter` uses the log-Euclidean mean. The affine-invariant
  `mean_riemann` failed to converge on batches of a few thousand 64×64 matrices,
  costing eleven minutes per fold for no measurable gain.
- Deep training keeps the training set resident in VRAM and indexes batches
  directly; a `DataLoader` cost 4× the wall time on networks this small.
- Eleven `zip()` calls now pass `strict=True` where the sequences must be the
  same length, turning silent truncation into an error.

### Fixed

- `summary_table` reported the number of *measurements* under the key
  `n_subjects`; both are now reported separately.
- ERD was computed against the first window of the decoding epoch, which is
  already inside the imagery period. `bwt explain` now re-loads over
  `ERD_WINDOW` to obtain a true pre-cue baseline, and warns if asked to compute
  ERD without one.

## [2.0.0] — 2026-07-25

Complete rewrite. v1 reported 94.3% accuracy; that came from evaluating on its
own training rows. Its honest held-out score was **72.07%** against a 67.93%
majority-class baseline, with 28.5% recall on class 1.

### Fixed

- **Label semantics.** `T1`/`T2` denote different movements in different runs
  (left/right fist in runs 3, 4, 7, 8, 11, 12; both fists/both feet in 5, 6, 9,
  10, 13, 14). v1 merged those families and folded resting baseline into class 0,
  producing a class meaning "right fist OR both feet". Tasks are now explicit
  data structures, and tests assert no task can merge hand with foot, or
  executed with imagined, contrasts.
- **Evaluation leakage.** v1 used a random split over epochs from the same
  recordings. Evaluation is now subject-grouped, with `assert_no_subject_leakage`
  on every fold and both within- and cross-subject protocols reported with
  confidence intervals.
- **Train/serve drift.** v1's web app reimplemented feature extraction and had
  diverged from training. All signal processing now lives inside the fitted
  pipeline, verified by parity tests on real recordings.
- **Features.** v1 used per-channel band power with no spatial filtering, a blind
  FastICA component-zeroing step, and a per-file 3σ filter that discarded 48% of
  the data. Replaced with CSP / FBCSP / Riemannian tangent-space pipelines.
- **Security.** `debug=True` removed, upload size capped, uploads no longer
  written to a client-controlled path, internal errors no longer leaked.

### Added

- Versioned model-card artifacts that refuse to load without documentation of
  their output classes.
- Subjects S088, S089, S092 and S100 excluded (128 Hz acquisition or corrupt
  annotations), verified against the EDF headers.
- A mental-command speller with information-transfer-rate metrics, a CLI, a
  hardened Flask service, pinned requirements and a Dockerfile.

## [1.0.0] — superseded

Original implementation, archived under `legacy/` with a written post-mortem.
Not runnable: every path was hardcoded to a directory that does not exist.
