"""Adversarial audit of the browser layer.

Every interaction here is an attempt to get the interface into a state it does
not handle: double submits, stopping before starting, navigating mid-stream,
swapping the file after choosing one, running twice without a reload, and
resizing to sizes nobody designed for. Anything that prints NaN, undefined,
[object Object], leaves a control stuck, or throws is a fault.
"""
import contextlib
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:5000"
ROOT = Path(r"C:\Users\Rashi Shah\Documents\brainwave_to_text")
LR = str(ROOT / "raw_data" / "S003" / "S003R04.edf")
FF = str(ROOT / "raw_data" / "S003" / "S003R06.edf")
BAD = ROOT / "raw_data" / "S001" / "S001R01.edf"      # baseline run, no cues
SHOTS = Path(r"C:\Users\RASHIS~1\AppData\Local\Temp\claude\C--Users-Rashi-Shah-Documents-brainwave-to-text\73486916-11d1-497d-932e-9f91458d0b2f\scratchpad\shots")

fails, npass, section = [], 0, ""
console = []


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


BADTEXT = re.compile(r"\bNaN\b|\bundefined\b|\[object Object\]|\bInfinity\b|\bnull\b")


def scan_text(pg, where):
    txt = pg.inner_text("body")
    hits = sorted(set(BADTEXT.findall(txt)))
    check(f"{where}: no NaN/undefined/[object Object] on screen", not hits, str(hits))
    return txt


with sync_playwright() as pw:
    br = pw.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
    pg = br.new_page(viewport={"width": 1600, "height": 1000})
    pg.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
    pg.on("console", lambda m: console.append(f"console: {m.text}")
          if m.type == "error" else None)

    # ============================================ decode page abuse
    head("decode page: control state")
    pg.goto(BASE + "/", wait_until="networkidle")
    pg.wait_for_timeout(800)
    scan_text(pg, "decode idle")
    check("submit disabled before a file is chosen",
          pg.eval_on_selector("#submit", "e => e.disabled"))
    # choose, then clear, then choose again
    pg.set_input_files("#file", LR)
    pg.wait_for_timeout(300)
    pg.set_input_files("#file", [])
    pg.wait_for_timeout(300)
    check("clearing the file disables submit again",
          pg.eval_on_selector("#submit", "e => e.disabled"),
          pg.inner_text("#filename"))
    pg.set_input_files("#file", FF)
    pg.wait_for_timeout(300)
    check("re-choosing a different file updates the name",
          "S003R06" in pg.inner_text("#filename"), pg.inner_text("#filename"))

    head("decode page: double submit")
    pg.eval_on_selector("#form", "f => { window.__n = 0; f.addEventListener('submit', () => window.__n++); }")
    pg.click("#submit")
    with contextlib.suppress(Exception):
        pg.click("#submit", timeout=1200)
    pg.wait_for_url("**/predict", timeout=240000)
    pg.wait_for_timeout(800)
    check("landed on a result page after double-clicking submit",
          "/predict" in pg.url, pg.url)
    txt = scan_text(pg, "result page")

    head("result page: content sanity")
    rows = pg.evaluate("""() => {
      const t = document.querySelector('table');
      if (!t) return null;
      const head = [...t.tHead.rows[0].cells].map(c => c.innerText.trim().toLowerCase());
      const rs = [...t.tBodies[0].rows].map(r => [...r.cells].map(c => c.innerText.trim()));
      return {head, rs};
    }""")
    check("result table exists", rows is not None)
    if rows:
        ti = rows["head"].index("target")
        ci = rows["head"].index("confidence")
        check("every target cell is one of the four positions",
              all(r[ti].split()[-1] in {"left", "right", "top", "bottom"} for r in rows["rs"]),
              str({r[ti] for r in rows["rs"]}))
        check("every confidence parses as a probability",
              all(0.0 <= float(r[ci]) <= 1.0 for r in rows["rs"]))
        check("confidence bars do not exceed 100%",
              pg.evaluate("""() => [...document.querySelectorAll('.bar > i')]
                    .every(i => parseFloat(i.style.width) <= 100.5)"""))
    check("result page has no horizontal overflow",
          not pg.evaluate("document.documentElement.scrollWidth > "
                          "document.documentElement.clientWidth + 2"))

    head("decode page: a recording with no cues")
    pg.goto(BASE + "/", wait_until="networkidle")
    pg.set_input_files("#file", str(BAD))
    pg.wait_for_timeout(300)
    pg.click("#submit")
    pg.wait_for_url("**/predict", timeout=240000)
    pg.wait_for_timeout(800)
    t = pg.inner_text("body")
    check("a cue-less recording is explained, not crashed",
          "Traceback" not in t and len(t) > 200, t[:100])
    scan_text(pg, "cue-less result")

    # ============================================ live page abuse
    head("live page: controls before a run")
    pg.goto(BASE + "/live", wait_until="networkidle")
    pg.wait_for_function("document.documentElement.dataset.world", timeout=120000)
    pg.wait_for_timeout(1500)
    scan_text(pg, "live idle")
    check("start disabled with no file",
          pg.eval_on_selector("#start", "e => e.disabled"))
    check("stop disabled with no run",
          pg.eval_on_selector("#stop", "e => e.disabled"))
    # clicking a disabled stop must not throw
    pg.eval_on_selector("#stop", "e => e.click()")
    pg.wait_for_timeout(400)
    check("clicking a disabled Stop is harmless",
          not [c for c in console if "pageerror" in c])

    head("live page: double start and stop-during-run")
    pg.set_input_files("#file", LR)
    pg.wait_for_timeout(400)
    pg.select_option("#speed", "10")
    pg.click("#start")
    with contextlib.suppress(Exception):
        pg.click("#start", timeout=1000)
    pg.wait_for_timeout(2500)
    check("start becomes disabled once running",
          pg.eval_on_selector("#start", "e => e.disabled"))
    n1 = int(pg.inner_text("#m-decisions") or 0)
    pg.click("#stop")
    pg.wait_for_timeout(2000)
    n2 = int(pg.inner_text("#m-decisions") or 0)
    pg.wait_for_timeout(2500)
    n3 = int(pg.inner_text("#m-decisions") or 0)
    check("stop actually halts the stream", n3 == n2, f"{n1} -> {n2} -> {n3}")
    check("stop re-enables start",
          not pg.eval_on_selector("#start", "e => e.disabled"))
    check("stop disables itself", pg.eval_on_selector("#stop", "e => e.disabled"))

    head("live page: second run must not inherit the first")
    pg.set_input_files("#file", FF)
    pg.wait_for_timeout(400)
    pg.click("#start")
    pg.wait_for_timeout(1200)
    check("counters reset on a new run",
          int(pg.inner_text("#m-decisions") or 0) <= n3,
          f"{pg.inner_text('#m-decisions')} vs previous {n3}")
    check("decisions log cleared on a new run",
          pg.eval_on_selector_all("#log tr", "r => r.length") <= max(2, n3),
          str(pg.eval_on_selector_all("#log tr", "r => r.length")))
    lit = pg.evaluate("async () => { const e = await window.neuralWorld; "
                      "return e ? e.flashRegions.length : -1; }")
    check("no structure left lit from the previous run at start", lit <= 1, str(lit))
    pg.wait_for_function("+document.getElementById('m-decisions').innerText > 0",
                         timeout=180000)
    pg.wait_for_timeout(3000)
    scan_text(pg, "live running")

    head("live page: values during a run are sane")
    post = pg.eval_on_selector_all(".meter .v", "e => e.map(x => parseFloat(x.innerText))")
    check("posterior values are probabilities", all(0 <= v <= 1 for v in post), str(post))
    check("posterior sums to 1", abs(sum(post) - 1) < 0.02, str(sum(post)))
    lv = pg.eval_on_selector_all("#keyed tr td:nth-child(3)",
                                 "c => c.map(x => x.innerText.trim())")
    check("site levels are numeric and in range",
          all(v not in {"—", ""} and 0 <= float(v) <= 1 for v in lv), str(lv))
    cue = pg.inner_text("#ld-cue").strip()
    check("target is one of the four positions",
          cue.split()[-1] in {"left", "right", "top", "bottom"}, repr(cue))
    log = pg.eval_on_selector_all(
        "#log tr", "rs => rs.map(r => [...r.cells].map(c => c.innerText.trim()))")
    check("every log confidence is a probability",
          all(0 <= float(r[3]) <= 1 for r in log if len(r) > 3 and r[3]),
          str([r[3] for r in log[:4] if len(r) > 3]))
    check("every log window count is a positive integer",
          all(int(r[4]) > 0 for r in log if len(r) > 4 and r[4]))

    head("live page: navigating away mid-stream")
    pg.click("a.navlink[href='/']", timeout=15000)
    pg.wait_for_url(BASE + "/")
    pg.wait_for_timeout(1500)
    check("navigating away mid-stream leaves a working page",
          pg.eval_on_selector("#submit", "e => e.offsetParent !== null"))
    check("no errors from tearing down an active stream",
          not [c for c in console if "pageerror" in c], str(console[:3]))
    pg.click("a.navlink[href='/live']", timeout=15000)
    pg.wait_for_url(BASE + "/live")
    pg.wait_for_function("document.documentElement.dataset.world === 'live'", timeout=120000)
    pg.wait_for_timeout(1200)
    check("returning to live gives a clean idle instrument",
          pg.eval_on_selector("#start", "e => e.disabled")
          and "none yet" in pg.inner_text("#log"))
    lv = pg.eval_on_selector_all("#keyed tr td:nth-child(3)",
                                 "c => c.map(x => x.innerText.trim())")
    check("site levels are blank again after returning",
          all(v in {"—", "-", ""} for v in lv), str(lv))

    br.close()

    # ============================================ extreme viewports
    head("extreme viewports")
    for w, h in [(3840, 2160), (2560, 1440), (1152, 700), (1024, 640), (900, 600)]:
        br = pw.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        pg = br.new_page(viewport={"width": w, "height": h})
        errs: list[str] = []
        pg.on("pageerror", lambda e, _e=errs: _e.append(str(e)))
        for path in ["/", "/live"]:
            pg.goto(BASE + path, wait_until="networkidle")
            if path == "/live":
                with contextlib.suppress(Exception):
                    pg.wait_for_function("document.documentElement.dataset.world",
                                         timeout=120000)
            pg.wait_for_timeout(1000)
            hs = pg.evaluate("document.documentElement.scrollWidth > "
                             "document.documentElement.clientWidth + 2")
            clipped = pg.evaluate("""() => [...document.querySelectorAll('.panel, .card, .metric')]
                  .filter(e => e.scrollHeight > e.clientHeight + 4
                               && getComputedStyle(e).overflowY === 'visible')
                  .map(e => (e.className || '').slice(0, 24)).slice(0, 3)""")
            check(f"{w}x{h} {path}: no sideways scroll", not hs)
            check(f"{w}x{h} {path}: nothing clipped vertically", not clipped, str(clipped))
        check(f"{w}x{h}: no page errors", not errs, str(errs[:2]))
        br.close()

print("\n" + "=" * 64)
real = [c for c in console if "favicon" not in c.lower() and "400 (BAD REQUEST)" not in c]
check("no console or page errors across the whole session", not real, str(real[:4]))
print(f"{npass}/{npass + len(fails)} checks passed")
if fails:
    print("\nFAILURES:")
    for f in fails:
        print("  - " + f)
sys.exit(1 if fails else 0)
