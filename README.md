# Brainwave-to-Text

An EEG motor-imagery decoder and mental-command speller, built on the PhysioNet
EEG Motor Movement/Imagery Database (EEGMMIDB) and BCI Competition IV-2a.

The system classifies **imagined movements** from scalp EEG and uses the
resulting sequence of discrete commands to drive a spelling interface. It offers
classical (CSP, Riemannian) and neural (EEGNet, ShallowConvNet, EEG-Conformer)
decoders behind one interface, per-user calibration, continuous decoding with
evidence accumulation, and tools to check that the model is using real
sensorimotor physiology rather than artifacts.

> **What this is not.** It does not read words, inner speech, or intent from the
> brain. No scalp-EEG system does. What a motor-imagery BCI can deliver is a few
> discrete commands per trial; text comes from using those commands to navigate a
> character-selection interface, which is how real assistive spellers work. Every
> throughput number below is stated in those terms.

---

## Results

Task `mi_left_right` — imagined left fist vs imagined right fist, runs 4/8/12,
105 subjects, 4,722 trials, 64 channels @ 160 Hz, 0.5–3.5 s post-cue.
Chance = 50.0%. Full record in `reports/benchmark_v3.json`.

| Pipeline | Type | Within-subject | Cross-subject | κ (cross) |
|---|---|---|---|---|
| `eegnet` | neural | RESULTS_EEGNET_WITHIN | **0.628 ± 0.027** | 0.256 |
| `csp_lda` | classical | 0.606 ± 0.174 | 0.611 ± 0.023 | 0.222 |
| `riemann_ts_aligned` | classical | 0.631 ± 0.161 | 0.580 ± 0.007 | 0.161 |
| `riemann_ts` | classical | 0.626 ± 0.155 | 0.578 ± 0.015 | 0.156 |
| `shallownet` | neural | — | RESULTS_SHALLOW | — |
| `conformer` | neural | — | RESULTS_CONFORMER | — |
| `fbcsp_lda` | classical | 0.559 ± 0.113 | 0.563 ± 0.010 | 0.126 |
| `bandpower_rf` | classical | 0.538 ± 0.106 | 0.536 ± 0.009 | 0.072 |

**The two columns answer different questions.**

- **Within-subject** is what a user gets once the system is calibrated on their
  own recordings. Each subject is cross-validated against themselves and the
  figure is the mean over 105 subjects. The ±0.17 spread matters more than the
  mean: individuals range from ~0.36 to ~0.98. A substantial minority of people
  cannot drive a motor-imagery BCI at all — a documented property of the
  paradigm, not a defect here.
- **Cross-subject** holds out whole subjects, so nothing about the test person is
  seen in training. Its tight spread reflects averaging over many people at once.

These figures match the published literature for this dataset and contrast.
**If you see motor-imagery accuracy near 95% on EEGMMIDB, something is leaking** —
usually a random split over epochs from one recording, or the inclusion of
executed-movement runs whose EMG contamination is trivially separable.

`bandpower_rf` is worth reading as a diagnosis of this project's first version.
It uses per-channel log band power with **no spatial filtering** — essentially
v1's feature design — and reaches 0.538. The gap from there to 0.628 is what
spatial filtering and learned representations buy, and it is why v1's approach
could not have worked no matter how the classifier was tuned.

### Per-user calibration

CALIBRATION_TABLE

Adapting a population model to the individual is the single largest lever. See
`bwt calibrate --help`.

### Evidence accumulation

A 61% per-trial decoder is far too unreliable to spell with: five consecutive
decisions at that rate land on the intended character about a tenth of the time.
The fix is a better *decision rule*, not a better classifier. `bwt.streaming`
accumulates log-likelihood across repeated trials and commits only when the
posterior crosses a threshold — sequential probability ratio testing.

Measured on **real out-of-fold probabilities** from the cross-subject `csp_lda`
model, where a single trial scores 0.611:

| Commit threshold | Decision accuracy | Trials per decision | Sequences that commit |
|---|---|---|---|
| 0.75 | 0.654 | 4.7 | 98% |
| 0.90 | 0.674 | 8.0 | 90% |
| 0.99 | **0.707** | 12.8 | 72% |

**This is much weaker than an idealised simulation suggests, and that gap is the
point.** `simulate_speller_throughput`, which assumes independent draws, predicts
0.99+ decision accuracy from the same 0.611 decoder. Reality gives 0.707, because
a subject's errors are correlated: for someone the model cannot decode, gathering
more evidence yields a *confidently wrong* answer rather than a correct one. At
the strictest threshold, 28% of sequences never commit at all.

What that means for spelling, stated plainly: at 0.707 per decision and five
decisions per character, the intended character is selected about 18% of the
time, and each attempt costs roughly 12.8 × 5 ≈ 64 trials. Evidence accumulation
is a real and worthwhile improvement, but **it does not turn this into a usable
speller.**

Two caveats, both enforced in the code:

- **Repeated independent trials** (the user imagines the movement again) is the
  regime this works in. `evaluate_accumulation` measures it on real
  cross-validated probabilities, so subject-level error correlation is included.
- **Overlapping sliding windows within one trial are not independent.** They
  share most of their samples and therefore share their errors, so accumulation
  makes the decoder *confident* rather than *correct*. You can watch this happen
  on the `/live` page: a subject the model is biased on will produce a long run
  of the same confident-but-wrong decision. `simulate_speller_throughput`
  documents its independence assumption and should be read as an upper bound.

### The same code on a different lab's data

BCI Competition IV-2a, subject A01, under the competition's own protocol — train
on session one, test on session two recorded on a different day:

| Pipeline | Two-class | Four-class | κ (four-class) |
|---|---|---|---|
| `csp_lda` | **0.903** | 0.809 | 0.745 |
| `riemann_ts` | 0.875 | **0.816** | 0.755 |
| `fbcsp_lda` | 0.896 | 0.802 | 0.736 |

```bash
bwt evaluate --dataset bnci2a --task mi_four_class --protocols session_holdout
```

Not a single line of pipeline code changed between this and the EEGMMIDB
results — a different lab, amplifier, 22 electrodes instead of 64, 250 Hz
instead of 160, and a tongue-imagery class. That matters for interpreting the
modest EEGMMIDB numbers: **the gap is a property of the corpora, not a defect in
the implementation.** EEGMMIDB supplies 45 imagined trials per subject for the
left/right task; BCI IV-2a supplies 288 per session under tighter experimental
control.

Two honest caveats. This is **one subject** — the remaining eight are still
downloading from a host that serves at ~27 kB/s — and A01 is a comparatively
strong performer. Published nine-subject means for four-class CSP on this dataset
sit nearer 0.68, so do not read 0.81 as a dataset-level figure.

### Does the model use real physiology?

`bwt explain` produces the evidence, written to `reports/`:

- **`erd_curve.png`** — mu/beta (8–30 Hz) power at C3, Cz and C4 relative to a
  pre-cue baseline. Shows the textbook ~30% event-related desynchronisation at
  cue onset, and the expected **contralateral crossover**: right-hand imagery
  suppresses C3 more, left-hand imagery suppresses C4 more.
- **`csp_patterns.png`** — CSP spatial patterns as scalp topographies. The
  leading components sit over sensorimotor cortex. Later components sometimes
  show a frontal hotspot, which is ocular rather than neural — worth knowing
  rather than hiding.
- **Lateralisation index** — C3 minus C4 power change during imagery, in
  percentage points: `right_fist −6.70`, `left_fist −1.60` over 60 subjects.
  The separation between classes is in the physiologically expected direction.
  Both are negative because group-averaged EEGMMIDB carries an overall
  C3-dominant bias in a mostly right-handed cohort; it is the *difference*
  between classes that the decoder exploits.

Measuring ERD requires pre-cue samples, which the decoding window (0.5–3.5 s)
does not contain. `bwt explain` re-loads the data over `ERD_WINDOW` for this
reason, and `band_power_timecourse` warns loudly if asked to compute ERD from a
bundle with no baseline period.

---

## Quick start

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
pip install -e .

bwt info                          # dataset inventory and run protocol
bwt datasets                      # available corpora and their tasks
bwt train --pipeline eegnet       # epoch, cross-validate, save an artifact
bwt models                        # artifacts with measured accuracy
bwt serve                         # http://127.0.0.1:5000
```

GPU is optional. `pip install torch --index-url https://download.pytorch.org/whl/cu128`
enables the neural pipelines on CUDA; without PyTorch the classical pipelines
work unchanged and the neural ones raise a clear ImportError.

---

## Datasets

| Key | Corpus | Subjects | Channels | Rate | Classes |
|---|---|---|---|---|---|
| `eegmmidb` | PhysioNet EEG Motor Movement/Imagery | 105 usable | 64 | 160 Hz | up to 4 |
| `bnci2a` | BCI Competition IV-2a | 9 | 22 | 250 Hz | 4 |

Everything above the data layer works on `EpochBundle` and never knows which
corpus produced it, so adding a dataset means implementing one class in
`bwt/data/datasets.py` rather than touching any model. BCI IV-2a is a deliberate
contrast — different lab, amplifier, montage, sampling rate, and a tongue-imagery
class — because results from a single recording rig are always suspect.

`eegmmidb` is read from local EDF files under `raw_data/` (override with
`$BWT_RAW_DATA`). `bnci2a` is fetched through MOABB on first use.

### The EEGMMIDB protocol, stated explicitly

The largest defect in this project's first version was a misreading of the
annotations. **`T1` and `T2` mean different movements in different runs:**

| Runs | Execution | `T1` | `T2` |
|---|---|---|---|
| 3, 7, 11 | executed | left fist | right fist |
| **4, 8, 12** | **imagined** | **left fist** | **right fist** |
| 5, 9, 13 | executed | both fists | both feet |
| **6, 10, 14** | **imagined** | **both fists** | **both feet** |
| 1, 2 | — | baseline: eyes open / eyes closed, no trials | |

`T0` is the cued rest period within task runs.

Pooling the two run families produces a class meaning "right fist **or** both
feet" — physiologically meaningless. Pooling executed with imagined runs mixes
real muscle activity into a task that is supposed to measure imagery.
`bwt.data.physionet` encodes all of this as data, and `tests/test_physionet.py`
asserts that no task can merge hand with foot contrasts or executed with imagined
runs. Exactly one task, `mi_move_vs_rest`, is exempt — collapsing effectors is
its entire point — and a test enforces that the exemption list stays that one
entry.

**Excluded subjects.** `S088`, `S089`, `S092`, `S100`, verified against the EDF
headers rather than taken on trust: the first three are 128 Hz with 5.125 s
trials, and S089's baseline runs carry a single 60 s `T1` annotation instead of a
rest marker.

### Tasks

| Task | Classes | Runs |
|---|---|---|
| `mi_left_right` | left_fist, right_fist | 4, 8, 12 |
| `mi_fists_feet` | both_fists, both_feet | 6, 10, 14 |
| `mi_four_class` | all four imagined movements | 4, 6, 8, 10, 12, 14 |
| `mi_left_right_rest` | left_fist, right_fist, rest | 4, 8, 12 |
| `mi_move_vs_rest` | rest, movement — asynchronous gating | 4, 6, 8, 10, 12, 14 |
| `me_left_right` | left_fist, right_fist (**executed**) | 3, 7, 11 |

`me_left_right` exists as an upper reference only. It scores higher because real
movement leaks EMG into the EEG; it is not a BCI result.

---

## Architecture

```
src/bwt/
  paths.py          repository-relative paths, all overridable by env var
  config.py         defaults -> configs/default.yaml -> BWT_* env -> CLI flags
  data/
    physionet.py    run/annotation protocol, task definitions, exclusions
    epochs.py       EDF -> labelled epoch tensor, subject IDs attached
    datasets.py     dataset registry + cached loading
  pipelines.py      model registry: filtering + spatial filters + classifier
  deep/
    modules.py      EEGNet, ShallowConvNet, EEG-Conformer
    estimator.py    sklearn-compatible wrapper around a PyTorch model
  calibration.py    population -> individual adaptation, calibration curves
  evaluation.py     subject-aware CV, permutation test, leakage guards
  streaming.py      sliding-window decoding, evidence accumulation
  explain.py        ERD curves, CSP topographies, lateralisation index
  artifacts.py      versioned persistence with a model card
  decoding.py       mental-command -> text speller, ITR metrics
  serving/          Flask app, predictor, live streaming endpoint
  cli.py            bwt entry point
```

### One pipeline, used for training and serving

Every model — classical or neural — accepts the same input the epoch loader
produces, a `(trials, channels, times)` array in microvolts, and performs **all**
of its own filtering and spatial projection internally. The fitted pipeline is
the entire artifact.

This is deliberate. v1 had a separate hand-written feature extractor in the web
app that had silently drifted from the training one.
`tests/test_serving.py::TestTrainServeParity` asserts byte-identical predictions
between the two paths, `TestRealDataEpochingParity` checks the epocher agrees
sample-for-sample on real recordings, and a test greps the serving module for
signal-processing calls that should not be there.

Wrapping the neural models as scikit-learn classifiers is what lets them reuse
the subject-grouped cross-validation, the artifact format, and the serving layer
without a parallel code path. That matters more for deep models than classical
ones: they have enough capacity to memorise a subject outright, so the leakage
guards must apply to them automatically.

### Pipelines

| Name | Method |
|---|---|
| `csp_lda` | Common Spatial Patterns on 8–30 Hz → shrinkage LDA |
| `fbcsp_lda` | Filter-Bank CSP over eight sub-bands → mutual-information selection → LDA |
| `riemann_ts` | Covariances → Riemannian tangent space → logistic regression |
| `riemann_ts_aligned` | As above with per-recording recentring for cross-subject transfer |
| `fb_riemann_ts` | Tangent-space features per sub-band, concatenated |
| `bandpower_rf` | Per-channel log band power → random forest (deliberately weak floor) |
| `eegnet` | Compact depthwise-separable CNN, ~2.8k parameters |
| `shallownet` | Temporal + spatial conv, square/log pooling — a learned FBCSP |
| `conformer` | Shallow conv tokeniser + transformer encoder |

Kernel sizes in the published architectures are quoted for the sampling rate of
the original paper, so every module rescales them from its `sfreq` argument
rather than hardcoding numbers that would be silently wrong at 160 Hz. A test
asserts this.

`riemann_ts_aligned` uses statistics of the batch it is transforming, so
predictions within one recording are not independent. Artifacts using it set
`requires_batch_recentering` in the model card and the service warns when handed
too few epochs. Its reference mean is log-Euclidean rather than affine-invariant:
the iterative Riemannian mean routinely failed to converge on batches of a few
thousand 64×64 matrices, costing eleven minutes per fold for no measurable gain.

---

## Evaluation

Three protocols, and every cross-subject split passes through
`assert_no_subject_leakage`, which raises if any subject appears on both sides of
a fold.

- `within_subject_cv` — stratified k-fold inside each subject; reports the
  distribution over subjects, not a pooled number.
- `cross_subject_cv` — `GroupKFold` on subject ID.
- `permutation_test` — shuffles labels **within** each subject, preserving group
  structure, and reports `(hits + 1) / (n + 1)`.

### What v1's numbers actually were

v1 reported **94.3%**, from `test_model.py` evaluating on the full
`cleaned_features.csv`, ~80% of which had trained the model. Its two saved
confusion matrices differ by exactly the training rows:

| | Trials | Errors | Accuracy |
|---|---|---|---|
| Training portion | 13,717 | 19 | 99.86% |
| Genuinely held out | 3,430 | 958 | **72.07%** |

Against a 67.93% majority-class baseline, with 28.5% recall on class 1 — and its
two classes had no coherent meaning. Details in `legacy/README.md`.

---

## Command reference

```bash
bwt info                                     # dataset inventory, run protocol
bwt datasets                                 # corpora and their tasks
bwt prepare --dataset bnci2a --task mi_four_class
bwt train    --pipeline eegnet --task mi_left_right
bwt evaluate --pipeline csp_lda --permutations 100
bwt benchmark --pipelines csp_lda eegnet
bwt calibrate --pipeline eegnet --budgets 0 5 10 20 40
bwt stream   raw_data/S042/S042R04.edf --speed 1.0 --threshold 0.9
bwt explain  --task mi_left_right
bwt predict  raw_data/S001/S001R04.edf --json
bwt models
bwt serve --port 8080
```

Common flags: `--dataset`, `--subjects 1,2,5-9`, `--tmin/--tmax`, `--cv-splits`,
`--n-jobs`, `--no-cache`, `--log-level DEBUG`.

---

## HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Liveness; reports the loaded model and its classes |
| `GET /api/v1/model` | Model card, measured accuracy, input contract |
| `POST /api/v1/predict` | Multipart `file=<recording.edf>` → JSON predictions |
| `POST /api/v1/stream` | Same, but streams newline-delimited JSON per window |
| `GET /` , `POST /predict` | Browser UI |
| `GET /live` | Live decoder with evidence bars and a running speller |

```bash
curl -F file=@raw_data/S001/S001R04.edf http://127.0.0.1:5000/api/v1/predict
curl -N -F file=@raw_data/S001/S001R04.edf \
     "http://127.0.0.1:5000/api/v1/stream?speed=1.0&step=0.5&threshold=0.9"
```

The stream endpoint emits NDJSON rather than server-sent events because the
recording arrives by POST and `EventSource` only issues GETs.

### Input contract

Enforced, never coerced. A recording that does not match is rejected with a 400
rather than silently resampled or reordered, because either would invalidate the
spatial filters the model learned: EDF/EDF+, exactly the model's sampling rate,
the montage recorded in the model card (order is normalised), and long enough to
yield one epoch. Uploads are capped at 64 MB, written to a private temp file that
never derives from the client-supplied filename, and removed in a `finally`
block. `debug` is never enabled — the Werkzeug debugger is a remote code
execution primitive, and v1 shipped it.

---

## Development

```bash
pip install -r requirements-dev.txt
pytest                    # full suite
pytest -m "not slow"      # skip tests needing the dataset
pytest --cov=bwt
```

The unit suite is hermetic: it generates synthetic epochs with a real, learnable
class difference and writes synthetic EDF files, so it runs on a checkout with no
dataset present. Neural tests skip automatically if PyTorch is absent.

### Model artifacts

Each is a directory containing `pipeline.joblib` and `model_card.json`. The card
records the task and what its classes mean, the exact input contract, subjects
trained on and excluded, measured accuracy under every protocol, and library
versions. Loading refuses a mismatched schema version and warns on library drift.
A model whose card is missing is refused outright — serving a classifier whose
output classes are undocumented is how v1 came to display "Left"/"Right" for a
model that meant nothing of the kind. Neural models persist their weights on CPU,
so a GPU-trained artifact loads anywhere.

---

## Limitations

- **Accuracy is modest and highly variable between people.** This is the
  paradigm, not the implementation.
- **The speller is a demonstrator.** Evidence accumulation makes it far more
  usable, but at a real cost in time per character, and only for repeated
  independent trials.
- **Cross-subject transfer is weak** without calibration. Real deployments
  calibrate per user.
- **Sliding-window accumulation amplifies bias** rather than averaging out
  noise, because overlapping windows share their errors.
- **No online acquisition.** Streaming replays a recording; there is no driver
  for live hardware.
- **Two datasets, both research-grade rigs.** Nothing here is validated against
  consumer headsets; the input contract will reject them, which is intended.

---

## References

- Schalk et al. (2004). BCI2000: A General-Purpose Brain-Computer Interface System. *IEEE TBME* 51(6).
- Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23).
- Brunner et al. (2008). BCI Competition 2008 – Graz data set A.
- Ang et al. (2008). Filter Bank Common Spatial Pattern (FBCSP) in BCI. *IJCNN*.
- Barachant et al. (2012). Multiclass BCI Classification by Riemannian Geometry. *IEEE TBME* 59(4).
- Zanini et al. (2018). Transfer Learning: A Riemannian Geometry Framework. *IEEE TBME* 65(5).
- Lawhern et al. (2018). EEGNet: a compact CNN for EEG-based BCIs. *J. Neural Eng.* 15(5).
- Schirrmeister et al. (2017). Deep learning with CNNs for EEG decoding and visualization. *Hum. Brain Mapp.* 38(11).
- Song et al. (2023). EEG Conformer. *IEEE TNSRE* 31.
- Haufe et al. (2014). On the interpretation of weight vectors of linear models in neuroimaging. *NeuroImage* 87.
- Wolpaw et al. (2002). Brain-computer interfaces for communication and control. *Clin. Neurophysiol.* 113(6).

## License

MIT. Research and educational use. **Not a medical device.**
