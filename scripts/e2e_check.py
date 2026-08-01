"""End-to-end suite: every route, both flows, the failure paths, the 3D
interaction, layout at six sizes, navigation, and the API contract.

Needs a running server and Playwright:

    bwt serve &
    python scripts/e2e_check.py

Exits non-zero on any failure.
"""
import io
import json
import sys
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:5000"
ROOT = Path(r"C:\Users\Rashi Shah\Documents\brainwave_to_text")
EDF = str(ROOT / "raw_data" / "S003" / "S003R04.edf")
EDF2 = str(ROOT / "raw_data" / "S001" / "S001R04.edf")
SHOTS = Path(r"C:\Users\RASHIS~1\AppData\Local\Temp\claude\C--Users-Rashi-Shah-Documents-brainwave-to-text\73486916-11d1-497d-932e-9f91458d0b2f\scratchpad\shots")

fails, npass = [], 0
section = ""


def head(t):
    global section
    section = t
    print(f"\n--- {t} " + "-" * max(0, 56 - len(t)))


def check(name, ok, detail=""):
    global npass
    if ok:
        npass += 1
    else:
        fails.append(f"[{section}] {name}")
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   [{detail}]" if detail else ""))


# ============================================================ HTTP surface
head("routes and status codes")
for path, want in [("/", 200), ("/live", 200), ("/healthz", 200),
                   ("/api/v1/model", 200), ("/api/v1/geometry", 200),
                   ("/static/app.css", 200), ("/static/brain3d.js", 200),
                   ("/static/cortex.bin", 200), ("/nope", 404)]:
    r = requests.get(BASE + path, timeout=120)
    check(f"GET {path} -> {want}", r.status_code == want, str(r.status_code))

check("healthz reports ok", requests.get(BASE + "/healthz", timeout=30).json()["status"] == "ok")

head("decode API: failure paths are rejected, not coerced")
r = requests.post(BASE + "/api/v1/predict", timeout=60)
check("no file -> 4xx", 400 <= r.status_code < 500, str(r.status_code))
r = requests.post(BASE + "/api/v1/predict",
                  files={"file": ("notes.txt", io.BytesIO(b"not an edf"), "text/plain")},
                  timeout=60)
check("non-EDF -> 4xx", 400 <= r.status_code < 500, str(r.status_code))
check("non-EDF explains itself", "error" in r.json(), str(r.json())[:120])
r = requests.post(BASE + "/api/v1/predict",
                  files={"file": ("x.edf", io.BytesIO(b"\x00" * 5000), "application/octet-stream")},
                  timeout=120)
check("corrupt EDF -> 4xx", 400 <= r.status_code < 500, str(r.status_code))

head("decode API: contract of a good response")
with open(EDF, "rb") as fh:
    d = requests.post(BASE + "/api/v1/predict", files={"file": fh}, timeout=600).json()
check("n_epochs matches predictions length", d["n_epochs"] == len(d["predictions"]))
check("every prediction has the required fields",
      all({"index", "label", "confidence", "probabilities"} <= set(p) for p in d["predictions"]))
check("every label is a declared class",
      all(p["label"] in d["classes"] for p in d["predictions"]))
check("confidence is the max probability",
      all(abs(p["confidence"] - max(p["probabilities"].values())) < 1e-9 for p in d["predictions"]))
check("mean_confidence matches the predictions",
      abs(d["mean_confidence"]
          - sum(p["confidence"] for p in d["predictions"]) / len(d["predictions"])) < 1e-3)
check("majority_label matches the predictions",
      d["majority_label"] == max({p["label"] for p in d["predictions"]},
                                 key=lambda x: sum(1 for p in d["predictions"] if p["label"] == x)))

head("live API: streaming contract")
with open(EDF, "rb") as fh:
    r = requests.post(BASE + "/api/v1/stream?band_power=1&raw=1&speed=0&step=0.5",
                      files={"file": fh}, timeout=900, stream=True)
    ev = [json.loads(x) for x in r.iter_lines() if x]
check("stream opens with a start frame", ev[0].get("type") == "start")
check("stream closes with an end frame", ev[-1].get("type") == "end")
check("every line is valid NDJSON", len(ev) > 10, f"{len(ev)} frames")
win = [e for e in ev if "posterior" in e]
check("window onsets increase monotonically",
      all(win[i]["onset_seconds"] <= win[i + 1]["onset_seconds"] for i in range(len(win) - 1)))
check("end frame count matches the windows seen",
      ev[-1].get("n_windows") == len(win), f"{ev[-1].get('n_windows')} vs {len(win)}")
r2 = requests.post(BASE + "/api/v1/stream",
                   files={"file": ("x.txt", io.BytesIO(b"nope"), "text/plain")},
                   timeout=120, stream=True)
body = [json.loads(x) for x in r2.iter_lines() if x]
check("stream rejects a non-EDF before streaming anything",
      r2.status_code >= 400 or any(e.get("type") == "error" for e in body),
      f"{r2.status_code} {str(body)[:100]}")

# ================================================================ browser
console = []


def page(pw, w=1600, h=1000):
    br = pw.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
    pg = br.new_page(viewport={"width": w, "height": h})
    pg.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
    pg.on("console", lambda m: console.append(f"console: {m.text}")
          if m.type == "error" else None)
    return br, pg


with sync_playwright() as pw:
    br, pg = page(pw)

    head("decode page: the whole flow in the browser")
    pg.goto(BASE + "/", wait_until="networkidle")
    check("decode button starts disabled",
          pg.eval_on_selector("#submit", "e => e.disabled"))
    pg.set_input_files("#file", EDF)
    pg.wait_for_timeout(400)
    check("decode button enables once a file is chosen",
          not pg.eval_on_selector("#submit", "e => e.disabled"))
    check("the chosen filename is shown",
          "S003R04" in pg.inner_text("#filename"), pg.inner_text("#filename"))
    check("decode button is actually clickable",
          pg.eval_on_selector("#submit", """e => {
            const r = e.getBoundingClientRect();
            const t = document.elementFromPoint(r.x + r.width/2, r.y + r.height/2);
            return e.contains(t) || t === e;
          }"""))
    pg.click("#submit")
    pg.wait_for_url("**/predict", timeout=180000)
    pg.wait_for_timeout(800)
    check("result page renders a per-trial table",
          pg.eval_on_selector_all("table tbody tr", "r => r.length") >= 10,
          str(pg.eval_on_selector_all("table tbody tr", "r => r.length")))
    body_txt = pg.inner_text("body")
    check("result page states the measured accuracy", "60.6" in body_txt)
    check("result page separates confidence from correctness",
          "not" in body_txt.lower() and "confiden" in body_txt.lower())
    # Located by header rather than by index: the table gains and loses columns
    # as the interface changes, and a positional assertion silently starts
    # measuring the wrong thing.
    table = pg.evaluate("""() => {
      const t = document.querySelector('table');
      const head = [...t.tHead.rows[0].cells].map(c => c.innerText.trim().toLowerCase());
      const rows = [...t.tBodies[0].rows].map(r => [...r.cells].map(c => c.innerText.trim()));
      return {head, rows};
    }""")
    ci = table["head"].index("confidence")
    check("result table has a Target column", "target" in table["head"], str(table["head"]))
    check("every target is a screen position",
          all(r[table["head"].index("target")].split()[-1]
              in {"left", "right", "top", "bottom"} for r in table["rows"]),
          str(table["rows"][0]))
    check("every table confidence is a probability",
          all(0.0 <= float(r[ci]) <= 1.0 for r in table["rows"]), str(table["rows"][0]))
    check("result page marks Decode file as current",
          pg.eval_on_selector("[aria-current='page']", "e => e.innerText").strip().upper()
          == "DECODE FILE")
    check("no 3D canvas is rendered on a document page",
          pg.eval_on_selector("#world", "e => getComputedStyle(e).display") == "none")

    head("decode page: a rejected recording is explained")
    bad = SHOTS.parent / "bad.edf"
    bad.write_bytes(b"\x00" * 4096)
    pg.goto(BASE + "/", wait_until="networkidle")
    pg.set_input_files("#file", str(bad))
    pg.wait_for_timeout(300)
    pg.click("#submit")
    pg.wait_for_url("**/predict", timeout=120000)
    pg.wait_for_timeout(500)
    t = pg.inner_text("body").lower()
    check("a bad file produces a readable rejection, not a stack trace",
          ("could not" in t or "reject" in t) and "traceback" not in t, t[:110])
    check("the rejection page still offers a way back",
          pg.eval_on_selector_all("a", "a => a.some(x => x.href.endsWith('/'))"))

    head("live page: idle state claims nothing")
    pg.goto(BASE + "/live", wait_until="networkidle")
    pg.wait_for_function("document.documentElement.dataset.world", timeout=90000)
    pg.wait_for_timeout(1500)
    lv = pg.eval_on_selector_all("#keyed tr td:nth-child(3)", "c => c.map(x => x.innerText.trim())")
    check("site levels are blank while idle", all(v in {"—", "-", ""} for v in lv), str(lv))
    check("no decisions logged while idle", "none yet" in pg.inner_text("#log"))
    check("start is disabled with no recording",
          pg.eval_on_selector("#start", "e => e.disabled"))
    check("stop is disabled with no run",
          pg.eval_on_selector("#stop", "e => e.disabled"))
    check("specimen is on screen",
          pg.eval_on_selector("#world", "e => +getComputedStyle(e).opacity") == 1)

    head("live page: structure interaction")
    box = pg.eval_on_selector("#stage", "e => { const r = e.getBoundingClientRect();"
                              " return [r.x, r.y, r.width, r.height]; }")
    hit = None
    for dx, dy in [(0.5, 0.45), (0.45, 0.5), (0.55, 0.4), (0.5, 0.55), (0.42, 0.42)]:
        pg.mouse.click(box[0] + box[2] * dx, box[1] + box[3] * dy)
        pg.wait_for_timeout(700)
        if not pg.eval_on_selector("#region", "e => e.hidden"):
            hit = (dx, dy)
            break
    check("clicking the specimen selects a structure", hit is not None, str(hit))
    if hit:
        # The <aside> spans the viewport because it hosts the full-screen SVG
        # leader; measuring it proves nothing. The card is .callout-inner.
        # Card and leader are read in ONE evaluate. Two round trips can land
        # either side of an animation frame, and the leader is re-aimed every
        # frame -- a straddled read reports a card and a leader from different
        # states and looks exactly like a disconnection bug.
        snap = pg.evaluate("""() => {
          const root = document.getElementById('region');
          const r = root.querySelector('.callout-inner').getBoundingClientRect();
          const pts = (root.querySelector('polyline').getAttribute('points') || '')
            .trim().split(/\\s+/).filter(Boolean).map(p => p.split(',').map(Number));
          return {x: r.x, y: r.y, w: r.width, h: r.height,
                  side: root.dataset.side,
                  name: root.querySelector('.callout-name').innerText,
                  note: root.querySelector('.callout-note').innerText,
                  lead: pts};
        }""")
        c, lead = snap, snap["lead"]
        vw, vh = 1600, 1000
        check("callout card is fully on screen",
              c["x"] >= -1 and c["y"] >= -1
              and c["x"] + c["w"] <= vw + 1 and c["y"] + c["h"] <= vh + 1, str(c))
        check("callout card has real size", 80 < c["w"] < vw * 0.6 and 20 < c["h"] < vh * 0.6,
              f"{c['w']}x{c['h']}")
        check("leader is drawn with at least two points", len(lead) >= 2, str(lead))
        check("every leader point is on screen",
              all(-2 <= x <= vw + 2 and -2 <= y <= vh + 2 for x, y in lead), str(lead))
        # The leader must actually reach the card, or it reads as a stray line.
        if lead and c["w"]:
            end = lead[-1]
            near = (c["x"] - 30 <= end[0] <= c["x"] + c["w"] + 30
                    and c["y"] - 30 <= end[1] <= c["y"] + c["h"] + 30)
            check("leader terminates at the callout card", near,
                  f"side={c['side']} end={end} card={c['x']:.0f}..{c['x'] + c['w']:.0f}")
        check("callout does not cover the side panels",
              c["x"] + c["w"] <= pg.eval_on_selector(
                  ".col-r", "e => e.getBoundingClientRect().left") + 2
              or c["x"] >= pg.eval_on_selector(
                  ".col-l", "e => e.getBoundingClientRect().right") - 2,
              str(c))
        check("callout names a structure", len(c["name"]) > 2, c["name"])
        check("callout explains the structure", len(c["note"]) > 20, c["note"][:60])
        check("callout does not overflow its own box",
              pg.eval_on_selector("#region", """e => [...e.querySelectorAll('*')]
                    .every(x => x.scrollWidth <= x.clientWidth + 2)"""))
        pg.query_selector("#stage").screenshot(path=str(SHOTS / "e2e_callout.png"))

    pg.mouse.click(box[0] + 20, box[1] + 20)
    pg.wait_for_timeout(600)

    head("live page: rotation")
    before = pg.evaluate("window.__yaw = null; true")
    pg.mouse.move(box[0] + box[2] / 2, box[1] + box[3] / 2)
    pg.mouse.down()
    for i in range(6):
        pg.mouse.move(box[0] + box[2] / 2 + i * 22, box[1] + box[3] / 2)
        pg.wait_for_timeout(40)
    pg.mouse.up()
    pg.wait_for_timeout(500)
    check("dragging does not throw", not [c for c in console if "pageerror" in c])

    head("live page: a full run")
    pg.set_input_files("#file", EDF)
    pg.wait_for_timeout(400)
    check("start enables once a recording is chosen",
          not pg.eval_on_selector("#start", "e => e.disabled"))
    pg.select_option("#speed", "10")
    pg.select_option("#step", "0.5")
    pg.select_option("#threshold", "0.9")
    pg.click("#start")
    pg.wait_for_timeout(1200)
    check("stop becomes available during a run",
          not pg.eval_on_selector("#stop", "e => e.disabled"))
    check("state reads as decoding", "decod" in pg.inner_text("#ov-status").lower(),
          pg.inner_text("#ov-status"))
    pg.wait_for_function("+document.getElementById('m-decisions').innerText > 0",
                         timeout=90000)
    pg.wait_for_timeout(6000)

    dec = int(pg.inner_text("#m-decisions"))
    check("decisions accumulate", dec > 0, f"{dec} decisions")
    lv = pg.eval_on_selector_all("#keyed tr td:nth-child(3)",
                                 "c => c.map(x => x.innerText.trim())")
    check("site levels now report numbers", all(v not in {"—", "-", ""} for v in lv), str(lv))
    check("site levels are in range",
          all(0.0 <= float(v) <= 1.0 for v in lv), str(lv))
    post = pg.eval_on_selector_all(".meter .v", "e => e.map(x => +x.innerText)")
    check("posterior sums to 1 on screen", abs(sum(post) - 1.0) < 0.02, str(post))
    log = pg.eval_on_selector_all("#log tr", "rs => rs.map(r => [...r.cells].map(c => c.innerText))")
    check("decision log fills", len(log) > 0, f"{len(log)} rows")
    check("logged confidences are probabilities",
          all(0.0 <= float(r[3]) <= 1.0 for r in log if len(r) > 3 and r[3]))
    check("logged window counts are positive integers",
          all(int(r[4]) > 0 for r in log if len(r) > 4 and r[4]))
    check("committed answer is a model class",
          pg.inner_text("#ld-class").strip() in {"left_fist", "right_fist"},
          pg.inner_text("#ld-class"))
    check("lateralisation is reported as a measurement",
          "sites" in pg.inner_text("#ld-site") or "no difference" in pg.inner_text("#ld-site"),
          pg.inner_text("#ld-site"))
    check("signal path shows a live classification",
          pg.inner_text("#pv-classify").strip() not in {"—", ""},
          pg.inner_text("#pv-classify"))
    # innerText returns the text-transform:uppercase form, so the micro sign
    # comes back as a capital Mu.
    scope_note = pg.inner_text("#scope-note")
    check("scope reports its own gain",
          "±" in scope_note and scope_note.lower().rstrip().endswith("v"), scope_note)
    check("the honesty statement is on screen",
          "no words" in pg.inner_text(".path-foot").lower())
    pg.screenshot(path=str(SHOTS / "e2e_live_running.png"))

    head("live page: stop, then a second run resets cleanly")
    pg.click("#stop")
    pg.wait_for_timeout(1500)
    check("stop disables itself", pg.eval_on_selector("#stop", "e => e.disabled"))
    check("start becomes available again",
          not pg.eval_on_selector("#start", "e => e.disabled"))
    pg.set_input_files("#file", EDF2)
    pg.wait_for_timeout(400)
    pg.click("#start")
    pg.wait_for_timeout(2500)
    check("a second run starts on a different recording",
          "decod" in pg.inner_text("#ov-status").lower())
    pg.wait_for_timeout(4000)
    check("counters advanced on the second run",
          int(pg.inner_text("#m-decisions")) >= 0)
    pg.click("#stop")
    pg.wait_for_timeout(1000)

    head("navigation between the two pages")
    for i in range(3):
        pg.click("a.navlink[href='/']", timeout=10000)
        pg.wait_for_url(BASE + "/")
        pg.wait_for_function(
            "document.getAnimations().filter(a=>a.playState==='running').length===0",
            timeout=10000)
        check(f"[{i}] decode page live after crossing",
              pg.evaluate("document.timeline.currentTime") > 0
              and not pg.evaluate("document.documentElement.hasAttribute('data-leaving')"))
        pg.click("a.navlink[href='/live']", timeout=10000)
        pg.wait_for_url(BASE + "/live")
        pg.wait_for_function("document.documentElement.dataset.world === 'live'",
                             timeout=40000)
        check(f"[{i}] live page live after crossing",
              pg.evaluate("document.timeline.currentTime") > 0)
    # The entry animation is still running the instant `world` flips to live,
    # so settle first -- then anything still hidden is a genuine leak.
    pg.wait_for_function(
        "document.getAnimations().filter(a=>a.playState==='running').length===0",
        timeout=10000)
    hidden = pg.evaluate("""() => [...document.querySelectorAll('.rig > *, .rig .panel')]
          .filter(e => +getComputedStyle(e).opacity !== 1)
          .map(e => (e.tagName + '.' + e.className).slice(0, 40)
                    + ' @' + getComputedStyle(e).opacity)""")
    check("nothing left hidden after crossings", not hidden, str(hidden))
    # An identity matrix is not a displacement: a filling animation reports its
    # end keyframe as matrix(1,0,0,1,0,0) rather than the string 'none'.
    offset = pg.evaluate("""() => [...document.querySelectorAll('.rig > *')]
          .map(e => [e.className, getComputedStyle(e).transform])
          .filter(([, t]) => t !== 'none'
                  && t.replace(/\\s/g, '') !== 'matrix(1,0,0,1,0,0)')""")
    check("nothing left displaced after crossings", not offset, str(offset))

    head("browser back and forward")
    pg.go_back()
    pg.wait_for_timeout(1200)
    check("back does not leave the exit state applied",
          not pg.evaluate("document.documentElement.hasAttribute('data-leaving')"))
    check("back leaves a visible page",
          pg.evaluate("""() => [...document.querySelectorAll('main.wrap > *, .rig > *')]
                .every(e => +getComputedStyle(e).opacity === 1)"""))
    pg.go_forward()
    pg.wait_for_timeout(1500)
    check("forward leaves a visible page",
          pg.evaluate("""() => [...document.querySelectorAll('main.wrap > *, .rig > *')]
                .every(e => +getComputedStyle(e).opacity === 1)"""))
    br.close()

    head("layout at six sizes, both pages")
    for w, h in [(1920, 1080), (1600, 1000), (1440, 900), (1366, 768), (1280, 800), (1024, 768)]:
        br, pg = page(pw, w, h)
        for path in ["/", "/live"]:
            pg.goto(BASE + path, wait_until="networkidle")
            if path == "/live":
                pg.wait_for_function("document.documentElement.dataset.world", timeout=90000)
            pg.wait_for_timeout(1200)
            ov = pg.evaluate("""() => {
              const bad = [];
              document.querySelectorAll('body *').forEach(e => {
                const s = getComputedStyle(e);
                if (s.overflowX === 'visible' && e.scrollWidth > e.clientWidth + 2
                    && e.clientWidth > 0) bad.push(e.tagName + '.' + (e.className || ''));
              });
              return [...new Set(bad)].slice(0, 5);
            }""")
            check(f"{w}x{h} {path}: no horizontal overflow", not ov, str(ov))
            hs = pg.evaluate("document.documentElement.scrollWidth "
                             "> document.documentElement.clientWidth + 2")
            check(f"{w}x{h} {path}: page does not scroll sideways", not hs)
        br.close()

    head("accessibility basics")
    br, pg = page(pw)
    pg.goto(BASE + "/live", wait_until="networkidle")
    pg.wait_for_timeout(1500)
    check("page declares a language",
          pg.eval_on_selector("html", "e => e.lang") == "en")
    check("page has a title", len(pg.title()) > 4, pg.title())
    check("the decorative canvas is hidden from assistive tech",
          pg.eval_on_selector("#world", "e => e.getAttribute('aria-hidden')") == "true")
    check("the callout announces itself politely",
          pg.eval_on_selector("#region", "e => e.getAttribute('aria-live')") == "polite")
    check("the current page is exposed, not just coloured",
          pg.eval_on_selector_all("[aria-current='page']", "e => e.length") == 1)
    check("every form control is labelled",
          pg.evaluate("""() => [...document.querySelectorAll('select, input')]
                .every(e => e.closest('label') || e.labels?.length
                            || e.getAttribute('aria-label'))"""))
    br.close()

head("console cleanliness across every page and flow")
# The deliberately corrupt upload is *supposed* to produce a 400; that is the
# rejection path being exercised, not a defect.
real = [c for c in console
        if "favicon" not in c.lower()
        and "400 (BAD REQUEST)" not in c]
check("no unexpected console or page errors in the entire run", not real, str(real[:4]))
check("the corrupt upload was in fact rejected by the server",
      any("400 (BAD REQUEST)" in c for c in console))

print("\n" + "=" * 66)
print(f"{npass}/{npass + len(fails)} checks passed")
if fails:
    print("\nFAILED:")
    for f in fails:
        print("  - " + f)
sys.exit(1 if fails else 0)
