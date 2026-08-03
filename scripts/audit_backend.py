"""Adversarial audit of the HTTP layer.

Every request here is an attempt to make the server do something it should not:
crash, leak a file, accept a malformed recording, honour an absurd parameter,
hang, or expose a stack trace.
"""
import io
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

BASE = "http://127.0.0.1:5000"
ROOT = Path(r"C:\Users\Rashi Shah\Documents\brainwave_to_text")
GOOD = ROOT / "raw_data" / "S003" / "S003R04.edf"

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


def post(payload, params="", timeout=300, path="/api/v1/predict"):
    return requests.post(BASE + path + params, files={"file": payload}, timeout=timeout)


# ==================================================== malformed uploads
head("malformed uploads must be rejected, never crash")
raw = GOOD.read_bytes()

cases = {
    "empty file": b"",
    "one byte": b"\x00",
    "truncated EDF header": raw[:100],
    "EDF header then garbage": raw[:256] + os.urandom(4096),
    "text pretending to be edf": b"hello world" * 500,
    "null bytes": b"\x00" * 100000,
    "EDF with bytes flipped": bytes(b ^ 0xFF for b in raw[:20000]),
}
for name, payload in cases.items():
    try:
        r = post(("probe.edf", io.BytesIO(payload), "application/octet-stream"))
        body = r.text[:400]
        ok = 400 <= r.status_code < 500
        check(f"{name}: 4xx not 5xx", ok, f"HTTP {r.status_code}")
        check(f"{name}: no stack trace leaked",
              "Traceback" not in body and "File \"" not in body, body[:80])
    except requests.exceptions.RequestException as exc:
        check(f"{name}: server stayed up", False, f"connection died: {exc}")

# A truncated EDF is still a valid EDF prefix: MNE reads the header and the
# annotations that survived, so decoding the trials that are actually present
# is correct. What matters is that the count is honest, not that it 4xxs.
r = post(("half.edf", io.BytesIO(raw[: len(raw) // 2]), "application/octet-stream"),
         timeout=600)
check("a truncated EDF decodes only the trials it still contains",
      r.status_code == 200 and 0 < r.json()["n_epochs"] < 15,
      f"HTTP {r.status_code} n_epochs={r.json().get('n_epochs')}")

check("server still alive after malformed uploads",
      requests.get(BASE + "/healthz", timeout=30).status_code == 200)

# ==================================================== filename attacks
head("filename must never influence the path written to")
for bad in ["../../../../etc/passwd.edf", "..\\..\\windows\\system32\\evil.edf",
            "a" * 400 + ".edf", "nul.edf", "with space and ;rm -rf.edf",
            "unicode\u202e.edf"]:
    try:
        r = post((bad, io.BytesIO(raw[:5000]), "application/octet-stream"))
        check(f"rejects/handles {bad[:28]!r}", r.status_code < 500, f"HTTP {r.status_code}")
    except Exception as exc:
        check(f"rejects/handles {bad[:28]!r}", False, str(exc)[:60])

# ==================================================== wrong extension
head("extension and content-type policing")
r = post(("recording.EDF", io.BytesIO(raw), "application/octet-stream"), timeout=600)
check("uppercase .EDF is accepted (case should not matter)",
      r.status_code == 200, f"HTTP {r.status_code}")
r = post(("recording.edf.txt", io.BytesIO(raw[:5000]), "application/octet-stream"))
check("double extension .edf.txt is rejected", 400 <= r.status_code < 500,
      f"HTTP {r.status_code}")
r = post(("noextension", io.BytesIO(raw[:5000]), "application/octet-stream"))
check("no extension is rejected", 400 <= r.status_code < 500, f"HTTP {r.status_code}")

# ==================================================== size limit
head("upload size limit")
big = b"\x00" * (70 * 1024 * 1024)          # over the 64 MB cap
try:
    r = post(("huge.edf", io.BytesIO(big), "application/octet-stream"), timeout=300)
    check("oversized upload rejected with 413", r.status_code == 413,
          f"HTTP {r.status_code}")
    check("413 body is JSON, not HTML error page",
          r.headers.get("content-type", "").startswith("application/json"),
          r.headers.get("content-type"))
except Exception as exc:
    check("oversized upload rejected cleanly", False, str(exc)[:80])
del big

# ==================================================== parameter abuse
head("stream parameters must be validated, not trusted")
params = {
    "threshold=0": "?threshold=0",
    "threshold=1": "?threshold=1",
    "threshold=-5": "?threshold=-5",
    "threshold=99": "?threshold=99",
    "threshold=abc": "?threshold=abc",
    "step=0": "?step=0",
    "step=-1": "?step=-1",
    "step=99999": "?step=99999",
    "speed=-1": "?speed=-1",
    "max_windows=0": "?max_windows=0",
    "max_windows=-3": "?max_windows=-3",
    "max_windows=999999": "?max_windows=999999",
}
for name, q in params.items():
    try:
        with open(GOOD, "rb") as fh:
            r = requests.post(BASE + "/api/v1/stream" + q, files={"file": fh},
                              timeout=180, stream=True)
            first = next(r.iter_lines(), b"")
            r.close()
        ok = r.status_code < 500
        body = first.decode("utf-8", "replace")[:120]
        check(f"{name}: no 5xx", ok, f"HTTP {r.status_code} {body[:70]}")
        if name.startswith(("threshold=", "step=", "speed=", "max_windows="))                 and not name.endswith("abc"):
            bad = name.split("=")[1]
            outside = (bad in {"0", "-1", "-3", "-5", "1", "99", "99999", "999999"})
            if outside:
                check(f"{name}: rejected rather than silently accepted",
                      r.status_code == 400, f"HTTP {r.status_code}")
    except Exception as exc:
        check(f"{name}: handled", False, type(exc).__name__ + " " + str(exc)[:60])

check("server alive after parameter abuse",
      requests.get(BASE + "/healthz", timeout=30).status_code == 200)

# ==================================================== temp file hygiene
head("temp file hygiene")
tmp = Path(tempfile.gettempdir())
before = {p.name for p in tmp.glob("*") if p.is_file()}
for _ in range(3):
    with open(GOOD, "rb") as fh:
        requests.post(BASE + "/api/v1/predict", files={"file": fh}, timeout=600)
    post(("bad.edf", io.BytesIO(b"\x00" * 4096), "application/octet-stream"))
time.sleep(1)
after = {p.name for p in tmp.glob("*") if p.is_file()}
leaked = after - before
check("no temp files left behind by success or failure",
      not leaked, str(sorted(leaked)[:5]))

uploads = ROOT / "uploads"
if uploads.is_dir():
    check("uploads/ directory is not accumulating files",
          len(list(uploads.iterdir())) == 0,
          str([p.name for p in list(uploads.iterdir())[:5]]))

# ==================================================== concurrency
head("concurrency")
results = []


def one():
    try:
        with open(GOOD, "rb") as fh:
            r = requests.post(BASE + "/api/v1/predict", files={"file": fh}, timeout=900)
        results.append((r.status_code, r.json().get("majority_label")))
    except Exception as exc:
        results.append(("ERR", str(exc)[:60]))


threads = [threading.Thread(target=one) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
codes = [c for c, _ in results]
check("4 concurrent decodes all return 200", all(c == 200 for c in codes), str(codes))
labels = {lab for _, lab in results}
check("concurrent decodes agree with each other", len(labels) == 1, str(labels))

# ==================================================== client disconnect
head("client disconnect mid-stream")
try:
    with open(GOOD, "rb") as fh:
        r = requests.post(BASE + "/api/v1/stream?speed=1", files={"file": fh},
                          timeout=60, stream=True)
        for i, _ in enumerate(r.iter_lines()):
            if i > 3:
                break
        r.close()
    time.sleep(2)
    check("server survives a client hanging up mid-stream",
          requests.get(BASE + "/healthz", timeout=30).status_code == 200)
except Exception as exc:
    check("server survives a client hanging up mid-stream", False, str(exc)[:80])

# ==================================================== response invariants
head("response invariants under repetition")
seen = []
for _ in range(3):
    with open(GOOD, "rb") as fh:
        seen.append(requests.post(BASE + "/api/v1/predict", files={"file": fh},
                                  timeout=600).json())
labs = [tuple(p["label"] for p in d["predictions"]) for d in seen]
check("same file decoded 3 times gives identical labels",
      len(set(labs)) == 1, f"{len(set(labs))} distinct outcomes")
confs = [tuple(round(p["confidence"], 6) for p in d["predictions"]) for d in seen]
check("same file decoded 3 times gives identical confidences",
      len(set(confs)) == 1)
check("request_id differs between requests",
      len({d["request_id"] for d in seen}) == 3)

# ==================================================== methods and headers
head("methods and misc")
check("GET on a POST-only endpoint is 405",
      requests.get(BASE + "/api/v1/predict", timeout=30).status_code == 405,
      str(requests.get(BASE + "/api/v1/predict", timeout=30).status_code))
r = requests.post(BASE + "/api/v1/predict", data=b"not multipart",
                  headers={"Content-Type": "application/json"}, timeout=60)
check("non-multipart POST is rejected cleanly", 400 <= r.status_code < 500,
      f"HTTP {r.status_code}")
r = requests.get(BASE + "/healthz", timeout=30)
check("healthz does not leak filesystem paths",
      "C:\\" not in r.text and "/home/" not in r.text, r.text[:80])

print("\n" + "=" * 64)
print(f"{npass}/{npass + len(fails)} checks passed")
if fails:
    print("\nFAILURES:")
    for f in fails:
        print("  - " + f)
sys.exit(1 if fails else 0)
