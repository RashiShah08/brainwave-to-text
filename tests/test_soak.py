"""Sustained mixed traffic against a live server, watching for slow leaks.

Six clients hit every kind of request at random for ``BWT_SOAK_SECONDS``
(default 120): decodes good and bad, fast and paced streams, streams abandoned
mid-way, oversized uploads, form posts, metadata, 404s. The process is sampled
throughout. Nothing may fail, and nothing may creep: resident memory, thread
count, spooled temporary files, replay slots, and /healthz latency.

Connections the local machine never manages to open are counted apart from bad
answers: at this request rate Windows runs out of fresh local ports and reuses
one still in TIME_WAIT, which costs the client a connection but says nothing
about the server.

Long by design, so it runs only with ``BWT_SOAK=1``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import statistics
import tempfile
import threading
import time
from collections import Counter

import pytest

import _factories as fx

pytestmark = pytest.mark.skipif(os.environ.get("BWT_SOAK") != "1",
                                reason="set BWT_SOAK=1 to run the sustained-load test")

SECONDS = float(os.environ.get("BWT_SOAK_SECONDS", "120"))
CLIENTS = 6
WARM_UP = 20.0


def test_sustained_mixed_traffic_leaks_nothing(tmp_path_factory, monkeypatch, caplog):
    import psutil
    import requests

    spool = tmp_path_factory.mktemp("soak_spool")
    monkeypatch.setattr(tempfile, "tempdir", str(spool))
    d = tmp_path_factory.mktemp("soak")
    artifact = fx.build_artifact(d / "lr")
    rec = {
        "good": fx.write_edf(d / "good.edf", duration=30.0).read_bytes(),
        "short": fx.write_edf(d / "short.edf", duration=8.0).read_bytes(),
        "rate": fx.write_edf(d / "rate.edf", duration=10.0, sfreq=128.0).read_bytes(),
        "garbage": b"\x00" * 6000,
        "huge": b"\x00" * (5 * 1024 * 1024),
    }
    app = fx.make_app(artifact, max_upload_mb=4, threads=4)
    # One INFO line per request, plus waitress's queue-depth warnings, add up to
    # hundreds of thousands of records over a long run. pytest keeps every one
    # of them (log capture and stderr capture both), which grew this very
    # process by gigabytes and was misread as a server leak. Only errors are
    # worth keeping here; they are also what a failure needs to show.
    caplog.set_level(logging.ERROR)

    def upload(session, url, payload, **kwargs):
        return session.post(url, files={"file": ("rec.edf", payload)}, timeout=120, **kwargs)

    def one(kind, session, base, latencies):
        if kind == "healthz":
            started = time.perf_counter()
            r = session.get(base + "/healthz", timeout=30)
            latencies.append(time.perf_counter() - started)
            return r.status_code in {200}
        if kind == "model":
            r = session.get(base + "/api/v1/model", timeout=30)
            return r.status_code == 200 and "card" in r.json()
        if kind == "decode":
            r = upload(session, base + "/api/v1/predict", rec["good"])
            return r.status_code == 200 and r.json()["n_epochs"] > 0
        if kind == "garbage":
            return upload(session, base + "/api/v1/predict", rec["garbage"]).status_code == 400
        if kind == "wrong_rate":
            return upload(session, base + "/api/v1/predict", rec["rate"]).status_code == 400
        if kind == "stream":
            r = upload(session, base + "/api/v1/stream?speed=0", rec["short"])
            last = r.text.strip().splitlines()[-1] if r.text.strip() else "{}"
            return r.status_code == 200 and json.loads(last).get("type") == "end"
        if kind == "paced":
            r = upload(session, base + "/api/v1/stream?speed=20", rec["short"])
            return r.status_code in {200, 503}
        if kind == "abandon":
            r = upload(session, base + "/api/v1/stream?speed=1", rec["good"], stream=True)
            ok = r.status_code in {200, 503}
            if r.status_code == 200:
                next(r.iter_lines())
            r.close()
            return ok
        if kind == "oversized":
            return upload(session, base + "/api/v1/predict", rec["huge"]).status_code == 413
        if kind == "form":
            r = upload(session, base + "/predict", rec["good"])
            return r.status_code == 200 and "Decoded" in r.text
        if kind == "missing":
            return session.get(base + "/api/v1/missing", timeout=30).status_code == 404
        raise AssertionError(kind)

    kinds = ["healthz", "healthz", "model", "decode", "garbage", "wrong_rate", "stream",
             "paced", "abandon", "oversized", "form", "missing"]
    process = psutil.Process()
    with fx.LiveServer(app, threads=4) as live:
        stop = threading.Event()
        lock = threading.Lock()
        counts, failures, latencies, refused = Counter(), [], [], []

        def client(seed):
            rng = random.Random(seed)
            session = requests.Session()
            while not stop.is_set():
                kind = rng.choice(kinds)
                try:
                    ok = one(kind, session, live.url, latencies)
                except requests.ConnectTimeout as exc:
                    # Not the server: a watchdog connecting once a second went
                    # through even as these failed. Tens of thousands of short
                    # connections leave thousands of sockets in TIME_WAIT, and
                    # Windows eventually reuses a local port still held by one,
                    # whose SYN is then dropped until the client gives up (21 s).
                    # An artifact of one machine hammering one port; counted
                    # rather than ignored, because a flood of them is a symptom.
                    ok = True
                    with lock:
                        refused.append(f"{kind}: {exc!r}"[:160])
                except (requests.RequestException, ValueError, StopIteration) as exc:
                    ok = False
                    kind = f"{kind}: {exc!r}"[:160]
                with lock:
                    counts[kind.split(":")[0]] += 1
                    if not ok:
                        failures.append(kind)

        workers = [threading.Thread(target=client, args=(i,), daemon=True)
                   for i in range(CLIENTS)]
        started = time.time()
        for w in workers:
            w.start()
        samples = []
        while time.time() - started < SECONDS:
            time.sleep(5)
            samples.append((time.time() - started, process.memory_info().rss,
                            threading.active_count(), len(os.listdir(spool))))
        stop.set()
        for w in workers:
            w.join(timeout=180)

        slots = app.extensions["bwt_replay_slots"]
        deadline = time.time() + 30
        while (slots._value != 3 or os.listdir(spool)) and time.time() < deadline:
            time.sleep(0.5)
        leftover_slots, leftover_files = slots._value, os.listdir(spool)

    total = sum(counts.values())
    assert total > SECONDS, f"too little traffic to mean anything: {counts}"
    assert failures == [], f"{len(failures)} bad answers, e.g. {failures[:5]}"
    assert len(refused) < max(10, total // 500), (
        f"{len(refused)} connections never got through of {total}, too many to blame on"
        f" local port reuse, e.g. {refused[:3]}")
    assert leftover_slots == 3, "a replay slot leaked"
    assert leftover_files == [], "temporary files leaked"

    warm = [s for s in samples if s[0] >= WARM_UP]
    assert len(warm) >= 4, "run longer than the warm-up to judge growth"
    third = max(1, len(warm) // 3)
    early = statistics.median(s[1] for s in warm[:third])
    late = statistics.median(s[1] for s in warm[-third:])
    assert late - early < 150 * 1024 * 1024, (
        f"resident memory crept {(late - early) / 1e6:.0f} MB over the run; samples (s, MB): "
        f"{[(round(s[0]), s[1] // 1048576) for s in samples]}")
    assert max(s[2] for s in warm) <= warm[0][2] + 12, "threads accumulated"
    assert statistics.quantiles(latencies, n=20)[-1] < 2.0, "healthz p95 over 2 s"
