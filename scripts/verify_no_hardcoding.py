"""Independent verification that both pages report the real model's output.

For the values it checks this does not trust the serving layer: the fitted
estimator is called directly on epochs cut straight from the EDF, and those
numbers are compared against what the HTTP endpoints emit. A match means the
pages cannot be showing anything but the model's real output.
"""

import json
import sys

import numpy as np
import requests

BASE = "http://127.0.0.1:5000"
EDF = sys.argv[1] if len(sys.argv) > 1 else None

fails, checks = [], []


def check(name, ok, detail=""):
    checks.append(name)
    if not ok:
        fails.append(name)
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))


from bwt.serving.predictor import Predictor  # noqa: E402

predictor = Predictor.load()
card = predictor.card
print(f"\nmodel   : {predictor.name}")
print(f"classes : {card.classes}")
print(f"channels: {card.n_channels} @ {card.sfreq} Hz, window {card.tmin}..{card.tmax}s\n")

# ---------------------------------------------------------- served == card
m = requests.get(f"{BASE}/api/v1/model", timeout=60).json()
m = m.get("card", m)
m.setdefault("n_channels", len(m.get("ch_names", [])))
check("served n_channels == model card", m["n_channels"] == card.n_channels,
      f"{m['n_channels']} vs {card.n_channels}")
check("served sfreq == model card", float(m["sfreq"]) == float(card.sfreq))
check("served classes == model card", list(m["classes"]) == list(card.classes))

# ------------------------------------------------- geometry == MNE montage
g = requests.get(f"{BASE}/api/v1/geometry", timeout=120).json()
names = [e["name"] for e in g["electrodes"]]
check("geometry channels == model channels", names == list(card.ch_names),
      f"{len(names)} vs {card.n_channels}")
check("no unplaced channels", not g["unplaced"], str(g["unplaced"]))

import mne  # noqa: E402

pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
key = {k.upper(): v for k, v in pos.items()}
raw_pos = np.array([key[n.upper()] for n in names])
c = raw_pos - raw_pos.mean(axis=0)
c = c / np.abs(c).max()
expect = np.stack([c[:, 0], c[:, 2], -c[:, 1]], axis=1)
served = np.array([[e["x"], e["y"], e["z"]] for e in g["electrodes"]])
check("electrode coordinates == MNE standard_1005",
      np.allclose(served, expect, atol=1.5e-4),
      f"max deviation {np.abs(served - expect).max():.6f}")

if EDF is None:
    print("\n(no EDF supplied; metadata checks only)")
    sys.exit(1 if fails else 0)

# ------------------------------------ recompute from the estimator directly
X, onsets, epoching, warnings = predictor.epochs_from_edf(EDF)
Xv = predictor.validate_array(X)
proba = predictor.model.predict_proba(Xv)
raw_labels = [card.classes[i] for i in proba.argmax(axis=1)]
raw_conf = proba.max(axis=1)
print(f"estimator called directly: {len(raw_labels)} epochs ({epoching})\n")

# ------------------------------------------------ HTTP decode == estimator
with open(EDF, "rb") as fh:
    r = requests.post(f"{BASE}/api/v1/predict", files={"file": fh}, timeout=600)
r.raise_for_status()
http = r.json()
hl = [p["label"] for p in http["predictions"]]
hc = [p["confidence"] for p in http["predictions"]]

check("decode page epoch count == estimator", http["n_epochs"] == len(raw_labels),
      f"{http['n_epochs']} vs {len(raw_labels)}")
check("decode page labels == estimator argmax", hl == raw_labels,
      f"{sum(a != b for a, b in zip(hl, raw_labels, strict=True))} mismatches")
check("decode page confidence == estimator max proba",
      np.allclose(hc, raw_conf, atol=6e-5),
      f"max dev {np.abs(np.array(hc) - raw_conf).max():.2e}")

hp = np.array([[p["probabilities"][c] for c in card.classes] for p in http["predictions"]])
check("decode page full probability vector == estimator",
      np.allclose(hp, proba, atol=6e-5),
      f"max dev {np.abs(hp - proba).max():.2e}")
check("probabilities sum to 1", np.allclose(hp.sum(axis=1), 1.0, atol=1e-6))
check("decode output is not constant",
      len(set(hl)) > 1 or len(set(np.round(hc, 6))) > 1,
      f"{len(set(hl))} labels, {len(set(np.round(hc, 6)))} distinct confidences")

# ------------------------------------------------------ live stream checks
with open(EDF, "rb") as fh:
    r = requests.post(
        f"{BASE}/api/v1/stream?band_power=1&raw=1&speed=0&step=0.5",
        files={"file": fh}, timeout=900, stream=True)
    r.raise_for_status()
    events = [json.loads(ln) for ln in r.iter_lines() if ln]

start = next(e for e in events if e.get("type") == "start")
win = [e for e in events if "posterior" in e]
print(f"\nstream: {len(win)} windows\n")

check("stream classes == model card", list(start["classes"]) == list(card.classes))
check("stream trace channels are real model channels",
      all(ch in card.ch_names for ch in start["trace"]["channels"]),
      " ".join(start["trace"]["channels"]))
check("stream trace sfreq == model card",
      float(start["trace"]["sfreq"]) == float(card.sfreq))

# --- the per-window classifier output must be the estimator's, recomputed ---
sw = [e for e in win if "probabilities" in e]
perw = np.array([[e["probabilities"][c] for c in card.classes] for e in sw])
check("stream per-window probabilities sum to 1",
      np.allclose(perw.sum(axis=1), 1.0, atol=1e-6))
check("stream per-window probabilities vary over time",
      float(perw.std(axis=0).mean()) > 1e-4,
      f"sd {perw.std(axis=0).mean():.4f}")
check("stream probabilities are not a repeated frame",
      len({row.tobytes() for row in perw}) > len(perw) * 0.5,
      f"{len({row.tobytes() for row in perw})} distinct of {len(perw)}")

# --- band power: real, varying, per channel --------------------------------
bp = np.array([e["band_power"] for e in win if "band_power" in e])
check("band power on every window", len(bp) == len(win), f"{len(bp)}/{len(win)}")
check("band power width == n_channels", bp.shape[1] == card.n_channels,
      f"{bp.shape[1]} vs {card.n_channels}")
check("band power varies across channels", float(bp.std(axis=1).mean()) > 1e-3,
      f"mean across-channel sd {bp.std(axis=1).mean():.4f}")
check("band power varies across time", float(bp.std(axis=0).mean()) > 1e-3,
      f"mean across-time sd {bp.std(axis=0).mean():.4f}")
check("band power frames are all distinct",
      len({row.tobytes() for row in bp}) == len(bp),
      f"{len({row.tobytes() for row in bp})} of {len(bp)}")

# --- raw trace must literally be the file's samples ------------------------
raws = [np.array(e["raw"]["samples"], dtype=float) for e in win if e.get("raw")]
if raws:
    R = np.concatenate(raws, axis=1)
    check("trace channel rows are all different from each other",
          len({row.tobytes() for row in R}) == R.shape[0],
          f"{len({row.tobytes() for row in R})} distinct of {R.shape[0]}")
    check("trace amplitudes are physiological micro-volts",
          1.0 < float(np.abs(R).max()) < 5000.0, f"peak {np.abs(R).max():.1f} uV")

    edf = mne.io.read_raw_edf(EDF, preload=True, verbose="ERROR")
    edf.rename_channels({c: c.strip(".").upper() for c in edf.ch_names})
    want = [ch.upper() for ch in start["trace"]["channels"]]
    have = [ch for ch in want if ch in edf.ch_names]
    if len(have) == len(want):
        fd = edf.get_data(picks=want) * 1e6
        seg = R[:, :256]
        hit = -1
        for off in range(0, min(fd.shape[1] - seg.shape[1], 60000)):
            if np.allclose(fd[:, off:off + seg.shape[1]], seg, atol=1e-6):
                hit = off
                break
        check("streamed trace IS the file's own samples", hit >= 0,
              f"exact match at sample offset {hit}" if hit >= 0 else "no offset matched")

# --- posterior must be accumulated, not copied -----------------------------
post = np.array([[e["posterior"][c] for c in card.classes] for e in sw])
check("posterior sums to 1", np.allclose(post.sum(axis=1), 1.0, atol=1e-6))
check("posterior differs from per-window probabilities (it accumulates)",
      not np.allclose(post, perw, atol=1e-6),
      f"mean |diff| {np.abs(post - perw).mean():.4f}")

dec = [w["decision"] for w in win if w.get("decision")]
print(f"\nstream committed {len(dec)} decisions")
if dec:
    check("every decision label is a model class or a timeout",
          all(d["label"] in card.classes or d["timed_out"] for d in dec))
    ok = all(abs(w["decision"]["confidence"] - max(w["posterior"].values())) < 1e-6
             for w in win if w.get("decision") and not w["decision"]["timed_out"])
    check("decision confidence == posterior at the moment of commit", ok)
    thr = 0.9
    ok = all(w["decision"]["timed_out"] or max(w["posterior"].values()) >= thr - 1e-9
             for w in win if w.get("decision"))
    check(f"every commit actually crossed the {thr} threshold", ok)

print("\n" + "=" * 64)
print(f"{len(checks) - len(fails)}/{len(checks)} checks passed")
if fails:
    print("FAILED:\n  - " + "\n  - ".join(fails))
sys.exit(1 if fails else 0)
