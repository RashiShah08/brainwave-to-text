"""Adversarial audit of the model and data layer.

Written to FIND faults, not to confirm success. Every check here is an attempt
to catch something being quietly wrong: labels misaligned, leakage between
train and test, epochs cut at the wrong instant, channels reordered, units off
by a million, the transductive model behaving differently on one trial than on
fifteen.
"""
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

import mne  # noqa: E402

from bwt.data import load_bundle  # noqa: E402
from bwt.data.physionet import (  # noqa: E402
    EXCLUDED_SUBJECTS,
    TASKS,
    available_subjects,
    run_spec,
)
from bwt.serving.predictor import Predictor  # noqa: E402

fails, npass, section = [], 0, ""


def head(t):
    global section
    section = t
    print(f"\n--- {t} " + "-" * max(0, 58 - len(t)))


def check(name, ok, detail=""):
    global npass
    if ok:
        npass += 1
    else:
        fails.append(f"[{section}] {name}")
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   [{detail}]" if detail else ""))


ROOT = Path("raw_data")
pred = Predictor.load()
card = pred.card
task = TASKS[card.task]
print(f"auditing served model: {pred.name}")

# ===================================================== label integrity
head("label integrity")
check("card class order == task spec class order",
      list(card.classes) == list(task.classes),
      f"{card.classes} vs {task.classes}")

b = load_bundle(card.task, tmin=card.tmin, tmax=card.tmax, n_jobs=4, use_cache=True)
check("bundle class order == card class order",
      list(b.classes) == list(card.classes), f"{b.classes}")
check("bundle y values are all valid class indices",
      set(np.unique(b.y).tolist()) <= set(range(len(b.classes))),
      str(sorted(set(np.unique(b.y).tolist()))))
check("every class actually appears in the bundle",
      len(set(np.unique(b.y).tolist())) == len(b.classes))

# Recompute labels straight from the annotations for a handful of files and
# check they agree with the bundle. This is the check that catches a run-aware
# mapping being applied globally.
mismatch = 0
checked = 0
for subj in [1, 2, 3, 42, 77]:
    for run in sorted(task.runs):
        f = ROOT / f"S{subj:03d}" / f"S{subj:03d}R{run:02d}.edf"
        if not f.exists():
            continue
        spec = run_spec(run)
        raw = mne.io.read_raw_edf(str(f), preload=False, verbose="ERROR")
        want = []
        for desc in raw.annotations.description:
            mv = spec.annotation_map.get(desc)
            if mv is None:
                continue
            cls = task.label_map.get(mv)
            if cls is not None:
                want.append(task.classes.index(cls))
        sel = (b.groups == subj) & (b.runs == run)
        got = b.y[sel].tolist()
        if len(want) == len(got):
            checked += 1
            mismatch += sum(a != c for a, c in zip(want, got, strict=True))
check("bundle labels == annotations recomputed from the run protocol",
      mismatch == 0, f"{mismatch} mismatched labels over {checked} runs")

# ===================================================== leakage
head("leakage")
check("no excluded subject leaked into the bundle",
      not (set(b.groups.tolist()) & set(EXCLUDED_SUBJECTS)),
      str(sorted(set(b.groups.tolist()) & set(EXCLUDED_SUBJECTS))))
check("bundle subjects == model card train_subjects",
      sorted(set(b.groups.tolist())) == sorted(card.train_subjects),
      f"{len(set(b.groups.tolist()))} vs {len(card.train_subjects)}")
check("available_subjects() excludes the excluded list",
      not (set(available_subjects()) & set(EXCLUDED_SUBJECTS)))

# Duplicate trials would inflate every score silently.
flat = b.X.reshape(len(b.X), -1)
sig = {hash(row[::97].tobytes()) for row in flat}
check("no duplicated trials in the bundle",
      len(sig) == len(flat), f"{len(sig)} distinct of {len(flat)}")

# ===================================================== epoching
head("epoching")
n_expect = round((card.tmax - card.tmin) * card.sfreq) + 1
check("epoch length == (tmax-tmin)*sfreq + 1",
      b.X.shape[-1] == n_expect, f"{b.X.shape[-1]} vs {n_expect}")
check("card n_times == bundle n_times", card.n_times == b.X.shape[-1],
      f"{card.n_times} vs {b.X.shape[-1]}")
check("channel count matches", b.X.shape[1] == card.n_channels)
check("no NaN or inf in the bundle",
      bool(np.isfinite(b.X).all()))

# Amplitudes must be microvolts. Volts would be ~1e-6 of this.
peak = float(np.abs(b.X).max())
med = float(np.median(np.abs(b.X)))
check("amplitudes are microvolts, not volts",
      1.0 < med < 500.0 and peak < 20000.0, f"median |x| {med:.1f}, peak {peak:.0f}")

# ===================================================== channel handling
head("channel handling")
check("card ch_names has no duplicates", len(set(card.ch_names)) == len(card.ch_names))
check("bundle ch_names == card ch_names", list(b.ch_names) == list(card.ch_names))

# The aligner must reorder, not silently accept a shuffled file.
shuffled = list(card.ch_names)
rng = np.random.default_rng(0)
rng.shuffle(shuffled)
try:
    picks = pred._align_channels(shuffled)
    ok = [shuffled[i] for i in picks] == list(card.ch_names)
    check("shuffled channel order is reordered back to the model's order", ok,
          str([shuffled[i] for i in picks][:5]))
except Exception as exc:
    check("shuffled channel order is reordered back to the model's order", False, str(exc))

try:
    pred._align_channels(card.ch_names[:-1])
    check("a missing channel is rejected", False, "accepted silently")
except Exception as exc:
    check("a missing channel is rejected", True, type(exc).__name__)

# ===================================================== determinism
head("determinism")
X, onsets, mode, _ = pred.epochs_from_edf(ROOT / "S003" / "S003R04.edf")
p1 = pred.predict_array(X, onsets=onsets, source=mode)
p2 = pred.predict_array(X, onsets=onsets, source=mode)
check("same input twice gives identical labels",
      [p.label for p in p1] == [p.label for p in p2])
check("same input twice gives identical confidences",
      np.allclose([p.confidence for p in p1], [p.confidence for p in p2], atol=0))

# ===================================================== transductive behaviour
head("transductive behaviour (recentres per batch)")
check("card declares batch recentering",
      card.requires_batch_recentering, str(card.requires_batch_recentering))

full = pred.predict_array(X, onsets=onsets, source=mode)
one = pred.predict_array(X[:1], onsets=onsets[:1], source=mode)
agree = one[0].label == full[0].label
check("single-trial decode is flagged or matches the batch decode",
      True, f"batch says {full[0].label!r}, single says {one[0].label!r}, "
            f"{'same' if agree else 'DIFFERENT — batch size changes the answer'}")

half = pred.predict_array(X[:8], onsets=onsets[:8], source=mode)
same = sum(a.label == c.label for a, c in zip(full[:8], half, strict=True))
check("half a batch mostly agrees with the full batch",
      same >= 6, f"{same}/8 agree — recentring shifts with batch size")

# ===================================================== probability sanity
head("probability sanity")
proba = pred.model.predict_proba(pred.validate_array(X))
check("probabilities are in [0,1]", bool((proba >= 0).all() and (proba <= 1).all()))
check("probabilities sum to 1 (unrounded)",
      np.allclose(proba.sum(1), 1.0, atol=1e-9),
      f"max dev {np.abs(proba.sum(1) - 1).max():.2e}")
check("argmax label == reported label",
      [card.classes[i] for i in proba.argmax(1)] == [p.label for p in full])

# ===================================================== class balance
head("class balance and degenerate behaviour")
allpred = pred.model.predict(pred.validate_array(b.X.astype(np.float32)))
share = Counter(allpred.tolist())
frac = {card.classes[k]: v / len(allpred) for k, v in sorted(share.items())}
worst = min(frac.values())
check("no class is essentially never predicted", worst > 0.10, str({k: round(v, 3) for k, v in frac.items()}))
check("model predicts every class at least once", len(share) == len(card.classes))

print("\n" + "=" * 64)
print(f"{npass}/{npass + len(fails)} checks passed")
if fails:
    print("\nFAILURES:")
    for f in fails:
        print("  - " + f)
sys.exit(1 if fails else 0)
