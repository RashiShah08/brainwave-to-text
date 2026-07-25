# Brainwave-to-Text

An EEG motor-imagery decoder and mental-command speller, built on the PhysioNet
EEG Motor Movement/Imagery Database (EEGMMIDB).

The system classifies **imagined movements** from 64-channel scalp EEG and uses
the resulting sequence of discrete commands to drive a spelling interface that
produces text.

> **What this is not.** It does not read words, inner speech, or intent from the
> brain. No scalp-EEG system does. What a motor-imagery BCI can deliver is a few
> discrete commands per trial; text comes from using those commands to navigate a
> character-selection interface, which is how real assistive spellers work. The
> throughput numbers below are stated in those terms.

---

## Results

Task `mi_left_right` — imagined left fist vs imagined right fist, runs 4/8/12,
105 subjects, 4,722 trials, 64 channels @ 160 Hz, 0.5–3.5 s post-cue.
Chance = 50.0%.

| Pipeline | Within-subject | Cross-subject | Cohen κ (cross) |
|---|---|---|---|
| `csp_lda` *(default)* | 0.606 ± 0.174 | **0.611 ± 0.023** | 0.222 |
| `riemann_ts_aligned` | **0.631 ± 0.161** | 0.580 ± 0.007 | 0.161 |
| `riemann_ts` | 0.626 ± 0.155 | 0.578 ± 0.015 | 0.156 |
| `fb_riemann_ts` | 0.593 ± 0.137 | not measured¹ | — |
| `fbcsp_lda` | 0.559 ± 0.113 | 0.563 ± 0.010 | 0.126 |
| `bandpower_rf` | 0.538 ± 0.106 | 0.536 ± 0.009 | 0.072 |

Regenerate with `bwt benchmark --task mi_left_right`; the full record including
per-fold and per-subject scores is written to `reports/benchmark.json`.

¹ `fb_riemann_ts` cross-subject was not completed. Its four-band tangent-space
representation is 8,320 features wide, and fitting it on ~3,800 trials per fold
did not finish inside the time budget for this run. Fill the cell with
`bwt benchmark --pipelines fb_riemann_ts --protocols cross_subject`. Given it is
the weakest of the three Riemannian variants within-subject, it is not a
candidate for the default.

`csp_lda` is the default because it is the best **cross-subject** performer, and
cross-subject is the regime a shipped artifact actually operates in — it is
fitted on the training subjects and then applied to someone it has never seen.
The Riemannian pipelines are better **within-subject** and are the right choice
when calibrating on the end user's own recordings; select one with
`bwt train --pipeline riemann_ts_aligned`.

`bandpower_rf` is worth reading as a diagnosis of the previous version. It uses
per-channel log band power with **no spatial filtering** — which is essentially
v1's feature design — and reaches 0.538. The gap from there to 0.611 is what
spatial filtering buys, and it is why v1's approach could not have worked no
matter how the classifier was tuned.

**Read the two columns separately — they answer different questions.**

- **Within-subject** is what a user gets once the system is calibrated on their
  own recordings, which is how a BCI is actually deployed. Each subject is
  cross-validated against themselves and the reported figure is the mean over
  105 subjects. The ±0.155 spread matters more than the mean: individual
  subjects range from ~0.36 to ~0.98. A substantial minority of people cannot
  drive a motor-imagery BCI at all, which is a well-documented property of the
  paradigm and not a bug in this implementation.
- **Cross-subject** is the zero-calibration number: whole subjects are held out,
  so nothing about the test person is seen during training. Its tight ±0.015
  spread reflects that it is averaging over many subjects at once.

These figures are consistent with the published literature for this dataset and
contrast. **If you see a motor-imagery accuracy near 95% on EEGMMIDB, something
is leaking** — most often a random train/test split over epochs from the same
recording, or the inclusion of executed-movement runs, whose EMG contamination
is trivially separable and is not brain-computer interfacing.

### What that means for spelling

At 62.6% two-class accuracy over a 3-second trial, the Wolpaw information
transfer rate is roughly 1.3 bits/minute. The 32-symbol binary-tree speller
needs 5 correct decisions per character, so the probability of landing on the
intended character is 0.626⁵ ≈ 0.10. This is an honest research demonstrator,
not a usable communication device. It becomes one only for the subset of
subjects at the top of that distribution — at 0.90 accuracy the same speller
reaches roughly 0.9 characters/minute. `bwt predict` and the `/api/v1/model`
endpoint both report these numbers for the loaded model.

---

## Quick start

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
pip install -e .

bwt info                          # what data is present, what the runs mean
bwt train --task mi_left_right    # epoch, cross-validate, save an artifact
bwt models                        # list artifacts with their measured accuracy
bwt serve                         # http://127.0.0.1:5000
```

The dataset is expected at `raw_data/S001 … S109`. Point elsewhere with
`BWT_RAW_DATA=/path/to/eegmmidb`.

---

## The protocol, stated explicitly

The single largest defect in the previous version of this project was a
misreading of the annotations. **`T1` and `T2` mean different movements in
different runs:**

| Runs | Execution | `T1` | `T2` |
|---|---|---|---|
| 3, 7, 11 | executed | left fist | right fist |
| **4, 8, 12** | **imagined** | **left fist** | **right fist** |
| 5, 9, 13 | executed | both fists | both feet |
| **6, 10, 14** | **imagined** | **both fists** | **both feet** |
| 1, 2 | — | baseline: eyes open / eyes closed, no trials | |

`T0` is the cued rest period within task runs.

Pooling the two run families gives a class that means "right fist **or** both
feet" and another that means "left fist **or** both fists" — physiologically
meaningless. Pooling executed with imagined runs mixes real muscle activity into
a task that is supposed to measure imagery. `bwt.data.physionet` encodes all of
this as data, and `tests/test_physionet.py` asserts that no task definition can
merge hand with foot contrasts or executed with imagined runs.

### Excluded subjects

`S088`, `S089`, `S092`, `S100` are excluded from every analysis — 105 subjects
remain. Verified directly against the EDF headers in this repository, not taken
on trust:

- **S088, S092, S100** — recorded at 128 Hz with 5.125 s trials, a different
  acquisition configuration from the other 106 subjects' 160 Hz / 4.1 s.
- **S089** — baseline runs carry a single 60 s `T1` annotation instead of a rest
  marker, and R03 is 181 s with 22 rest events. Its annotation timing cannot be
  trusted.

---

## Architecture

```
src/bwt/
  paths.py          repository-relative paths; every location overridable by env var
  config.py         defaults -> configs/default.yaml -> BWT_* env -> CLI flags
  data/
    physionet.py    run/annotation protocol, task definitions, exclusions
    epochs.py       EDF -> labelled epoch tensor, with subject IDs attached
  pipelines.py      model registry: filtering + spatial filters + classifier
  evaluation.py     subject-aware CV protocols, permutation test, leakage guards
  artifacts.py      versioned persistence with a model card
  decoding.py       mental-command -> text speller, ITR metrics
  serving/          Flask app + predictor
  cli.py            bwt entry point
```

### One pipeline, used for both training and serving

Every model in the registry accepts the same input the epoch loader produces — a
`(trials, channels, times)` array in microvolts at 160 Hz — and performs **all**
of its own filtering and spatial projection internally. The fitted pipeline is
the entire artifact.

This is deliberate. The previous version had a separate hand-written feature
extractor in the web app that had silently drifted from the training one, so the
served model computed different features from the ones it was fitted on.
`tests/test_serving.py::TestTrainServeParity` asserts byte-identical predictions
between the two paths and greps the serving module for signal-processing calls
that should not be there.

### Available pipelines

| Name | Method |
|---|---|
| `csp_lda` | Common Spatial Patterns on 8–30 Hz → shrinkage LDA. The reference method. |
| `fbcsp_lda` | Filter-Bank CSP over eight sub-bands → mutual-information selection → LDA (Ang et al., 2008). |
| `riemann_ts` | Covariances → Riemannian tangent space → logistic regression (Barachant et al., 2012). |
| `riemann_ts_aligned` | As above, with per-recording recentering for cross-subject transfer (Zanini et al., 2018). Transductive — see below. |
| `fb_riemann_ts` | Tangent-space features per sub-band, concatenated. |
| `bandpower_rf` | Per-channel log band power → random forest. No spatial filtering; a deliberately weak interpretable floor. |

`riemann_ts_aligned` uses statistics of the batch it is transforming, so
predictions within one recording are not independent of each other. Artifacts
using it set `requires_batch_recentering` in the model card, and the service
warns when it is handed too few epochs to recenter with.

---

## Evaluation, and how it was wrong before

`bwt.evaluation` provides three protocols, and every cross-subject split passes
through `assert_no_subject_leakage`, which raises if any subject appears on both
sides of a fold. It is cheap, and it converts the most damaging silent bug in
this domain into a crash.

- `within_subject_cv` — stratified k-fold inside each subject; reports the
  distribution over subjects, not a single pooled number.
- `cross_subject_cv` — `GroupKFold` on subject ID.
- `permutation_test` — shuffles labels **within** each subject, preserving group
  structure, and reports `(hits + 1) / (n + 1)` so the p-value is never zero.

### What the old numbers were

The previous pipeline reported **94.3%**. That came from `test_model.py`
evaluating on the full `cleaned_features.csv`, roughly 80% of which had trained
the model. Its two saved confusion matrices differ by exactly the training rows:

| | Trials | Errors | Accuracy |
|---|---|---|---|
| Training portion | 13,717 | 19 | 99.86% |
| Genuinely held out | 3,430 | 958 | **72.07%** |

Against a 67.93% majority-class baseline, with 28.5% recall on class 1. And even
that 72% was inflated, because a plain random split puts adjacent windows from
the same recording on both sides. Details in `legacy/README.md`.

---

## Command reference

```bash
bwt info                                    # dataset inventory and run protocol
bwt prepare --task mi_four_class            # epoch and cache
bwt train   --task mi_left_right --pipeline riemann_ts
bwt evaluate --pipeline csp_lda --permutations 100
bwt benchmark --task mi_left_right          # compare every pipeline
bwt predict raw_data/S001/S001R04.edf --json
bwt models                                  # artifacts + measured accuracy
bwt serve --port 8080
```

Useful flags: `--subjects 1,2,5-9`, `--tmin/--tmax`, `--cv-splits`, `--n-jobs`,
`--no-cache`, `--log-level DEBUG`.

### Tasks

| Task | Classes | Runs |
|---|---|---|
| `mi_left_right` | left_fist, right_fist | 4, 8, 12 |
| `mi_fists_feet` | both_fists, both_feet | 6, 10, 14 |
| `mi_four_class` | all four imagined movements | 4, 6, 8, 10, 12, 14 |
| `mi_left_right_rest` | left_fist, right_fist, rest | 4, 8, 12 |
| `me_left_right` | left_fist, right_fist (**executed**) | 3, 7, 11 |

`me_left_right` exists as an upper reference only. It scores higher because real
movement leaks EMG into the EEG; it is not a BCI result and must never be
reported as one.

---

## HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Liveness; reports the loaded model and its classes |
| `GET /api/v1/model` | Full model card, measured accuracy, input contract |
| `POST /api/v1/predict` | Multipart `file=<recording.edf>` → JSON predictions |
| `GET /` , `POST /predict` | Browser UI |

```bash
curl -F file=@raw_data/S001/S001R04.edf http://127.0.0.1:5000/api/v1/predict
```

The response carries per-trial labels with probabilities, the majority class,
the spelled text, and any warnings — for example when a recording has no cue
annotations and had to be split into fixed windows, which makes each prediction
an unaligned guess rather than a decoded trial.

### Input contract

Enforced, never coerced. A recording that does not match is rejected with a 400
rather than silently resampled or reordered, because either would invalidate the
spatial filters the model learned.

- EDF/EDF+, exactly 160 Hz, the 64-channel 10–10 montage (order is normalised).
- Long enough to yield one epoch of the model's window.
- Uploads capped at 64 MB, written to a private temp file that never derives
  from the client-supplied filename, and removed in a `finally` block.
- `debug` is never enabled — the Werkzeug debugger is a remote code execution
  primitive, and the previous version shipped it.

---

## Development

```bash
pip install -r requirements-dev.txt
pytest                    # 164 tests
pytest -m "not slow"      # skip the ones needing the dataset or EDF export
pytest --cov=bwt
```

The unit suite is hermetic: it generates synthetic epochs with a real, learnable
class difference and writes synthetic EDF files, so it runs on a checkout with
no dataset present.

### Configuration

`configs/default.yaml` mirrors every code default. Override any field with an
environment variable — `BWT_SERVE_PORT=8080`, `BWT_DATA_TASK=mi_four_class`,
`BWT_LOG_FORMAT=json` for one JSON object per log line.

### Model artifacts

Each is a directory containing `pipeline.joblib` and `model_card.json`. The card
records the task and what its classes mean, the exact input contract, the
subjects trained on and excluded, the measured accuracy under every protocol,
and the library versions used. Loading refuses a mismatched schema version and
warns on library drift. A model whose card is missing is refused outright —
serving a classifier whose output classes are undocumented is how the previous
version came to display "Left"/"Right" for a model that meant nothing of the
kind.

---

## Limitations

- **Accuracy is modest and highly variable between people.** See the results
  section. This is the paradigm, not the implementation.
- **The speller is a demonstrator.** Compounding per-trial errors over 5
  decisions per character makes it impractical at the mean accuracy; it is
  usable only for high performers.
- **Cross-subject transfer is weak.** Real deployments calibrate per user.
- **One dataset, one recording rig.** Nothing here has been validated against
  consumer headsets or a different montage; the input contract will reject them,
  which is the intended behaviour rather than a silent wrong answer.
- **No online/streaming mode.** Inference is file-at-a-time.

---

## References

- Schalk et al. (2004). BCI2000: A General-Purpose Brain-Computer Interface System. *IEEE TBME* 51(6).
- Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23).
- Ang et al. (2008). Filter Bank Common Spatial Pattern (FBCSP) in BCI. *IJCNN*.
- Barachant et al. (2012). Multiclass Brain-Computer Interface Classification by Riemannian Geometry. *IEEE TBME* 59(4).
- Zanini et al. (2018). Transfer Learning: A Riemannian Geometry Framework. *IEEE TBME* 65(5).
- Wolpaw et al. (2002). Brain-computer interfaces for communication and control. *Clin. Neurophysiol.* 113(6).

## License

MIT. Research and educational use. **Not a medical device.**
