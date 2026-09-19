"""Browser tests with Playwright: both pages, every flow, and the ways they break.

The application runs under waitress on an ephemeral port with a synthetic model
and synthetic recordings (``tests/_factories.py``), so the suite is hermetic.
Chromium renders WebGL through SwiftShader, which lets the 3D specimen run
headless.

Three kinds of test live here:

* **Parity** -- what the page shows is compared against the JSON API for the
  same recording, so a template or script that mangles, rounds or reorders a
  value is caught rather than merely rendered.
* **Adversarial** -- double submits, swapping files mid-run, aborting,
  malformed stream frames, hostile filenames and class names, network failure,
  no WebGL.
* **Layout and accessibility** -- viewport sweeps, keyboard-only use, labels.

Run with ``pytest -m e2e``. Needs ``pip install playwright`` and
``python -m playwright install chromium``. As in the backend suites, a defect
in the current page is ``xfail(strict=True)`` with the reason written out.
"""

from __future__ import annotations

import json
import re
import time

import pytest
import requests
from playwright.sync_api import TimeoutError as PlaywrightTimeout

import _factories as fx

sync_api = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.e2e

#: Text that means a script rendered a broken value.
BROKEN_TEXT = re.compile(r"\bNaN\b|\bundefined\b|\[object Object\]|\bInfinity\b")
SWIFTSHADER = ["--use-gl=swiftshader", "--enable-unsafe-swiftshader",
               "--ignore-gpu-blocklist"]
RUN_TIMEOUT = 180_000
#: Over the suite server's 4 MB upload limit.
HUGE_UPLOAD = {"name": "huge.edf", "mimeType": "application/octet-stream",
               "buffer": b"\x00" * (5 * 1024 * 1024)}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def recordings(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e_edfs")
    return {
        "good": fx.write_edf(d / "good_recording.edf", duration=60.0),
        "second": fx.write_edf(d / "second_recording.edf", duration=60.0, seed=11),
        "short": fx.write_edf(d / "short.edf", duration=10.0),
        "no_cues": fx.write_edf(d / "no_cues.edf", duration=40.0, cues=[]),
        "rate128": fx.write_edf(d / "rate128.edf", duration=20.0, sfreq=128.0),
        "garbage": _write(d / "garbage.edf", b"\x00" * 6000),
        "text": _write(d / "notes.txt", b"not a recording"),
    }


def _write(path, data: bytes):
    path.write_bytes(data)
    return path


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    artifact = fx.build_artifact(tmp_path_factory.mktemp("e2e_model") / "model")
    with fx.LiveServer(fx.make_app(artifact, max_upload_mb=4), threads=8) as live:
        yield live


@pytest.fixture(scope="session")
def model(server):
    return requests.get(server.url + "/api/v1/model", timeout=30).json()


@pytest.fixture(scope="session")
def playwright_instance():
    try:
        pw = sync_api.sync_playwright().start()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"playwright unavailable: {exc}")
    yield pw
    pw.stop()


@pytest.fixture(scope="session")
def browser(playwright_instance):
    try:
        browser = playwright_instance.chromium.launch(args=SWIFTSHADER)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"chromium not installed ({exc}); "
                    "run `python -m playwright install chromium`")
    yield browser
    browser.close()


class PageErrors(list):
    """Uncaught exceptions raised inside the page."""


def _open(browser, **context_args):
    context_args.setdefault("viewport", {"width": 1600, "height": 1000})
    context = browser.new_context(**context_args)
    page = context.new_page()
    page.errors = PageErrors()
    page.on("pageerror", lambda exc: page.errors.append(str(exc)))
    # Every browser test is also a Content-Security-Policy test: anything the
    # policy blocks is logged to the console, and counts as a page error.
    page.on("console", lambda msg: page.errors.append(f"console: {msg.text}")
            if "Content Security Policy" in msg.text else None)
    page.on("dialog", lambda dialog: (page.errors.append(
        f"unexpected dialog: {dialog.message}"), dialog.dismiss()))
    page.set_default_timeout(30_000)
    return context, page


@pytest.fixture
def page(browser):
    context, page = _open(browser)
    yield page
    errors = list(page.errors)
    context.close()
    assert not errors, f"uncaught page errors: {errors}"


def goto_live(page, server):
    page.goto(server.url + "/live", wait_until="networkidle")
    page.wait_for_function("() => " "document.documentElement.dataset.world",
                           timeout=120_000)


def api_predict(server, path):
    with open(path, "rb") as fh:
        return requests.post(server.url + "/api/v1/predict", files={"file": fh},
                             timeout=300).json()


def api_stream(server, path, **params):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    with open(path, "rb") as fh:
        response = requests.post(f"{server.url}/api/v1/stream?{query}",
                                 files={"file": fh}, timeout=300)
    return [json.loads(line) for line in response.text.splitlines() if line]


def text_of(page, selector) -> str:
    return page.inner_text(selector).strip()


def assert_no_broken_text(page):
    hits = sorted(set(BROKEN_TEXT.findall(page.inner_text("body"))))
    assert not hits, f"broken values rendered: {hits}"


def click_enabled(page, selector, timeout=30_000):
    """Click a control once it is enabled, saying so when it never is.

    A plain click on a disabled button only reports that the click timed out,
    which leaves it unclear whether the page was slow or never armed the
    control at all. This separates the two.
    """
    try:
        page.wait_for_function("s => !document.querySelector(s).disabled",
                               arg=selector, timeout=timeout)
    except PlaywrightTimeout:
        raise AssertionError(
            f"{selector} was still disabled after {timeout} ms; "
            f"status={text_of(page, '#status')!r} state={text_of(page, '#ov-status')!r}"
        ) from None
    page.click(selector)


def start_run(page, path, *, speed="0", step="0.5", threshold="0.9"):
    page.set_input_files("#file", str(path))
    page.select_option("#speed", speed)
    page.select_option("#step", step)
    page.select_option("#threshold", threshold)
    click_enabled(page, "#start")


def wait_for_state(page, state, timeout=RUN_TIMEOUT):
    page.wait_for_function(
        "s => document.getElementById('ov-status').textContent === s",
        arg=state, timeout=timeout)


# --------------------------------------------------------------------------- #
# Decode page
# --------------------------------------------------------------------------- #


class TestDecodePage:
    def test_idle_page_states_the_served_model_truthfully(self, page, server,
                                                          model):
        page.goto(server.url + "/", wait_until="networkidle")
        card, performance = model["card"], model["performance"]

        assert page.title() == "Decode a recording"
        assert page.eval_on_selector("html", "e => e.lang") == "en"
        assert text_of(page, "[aria-current='page']").upper() == "DECODE FILE"
        assert page.eval_on_selector("#submit", "e => e.disabled")
        assert text_of(page, "#filename") == ""

        body = page.inner_text("body")
        assert f"{performance['cross_subject_accuracy'] * 100:.1f}%" in body
        assert f"{performance['within_subject_accuracy'] * 100:.1f}%" in body
        assert f"{card['n_channels']} channels" in body
        assert f"{card['n_times']} samples" in body
        assert "does not read words" in body
        assert page.eval_on_selector("#world", "e => getComputedStyle(e).display") == "none"
        assert_no_broken_text(page)

    def test_choosing_and_clearing_a_file_drives_the_button(self, page, server,
                                                            recordings):
        page.goto(server.url + "/", wait_until="networkidle")
        page.set_input_files("#file", str(recordings["good"]))
        assert not page.eval_on_selector("#submit", "e => e.disabled")
        size_mb = recordings["good"].stat().st_size / 1048576
        # textContent, not innerText: innerText collapses the double space.
        assert page.eval_on_selector("#filename", "e => e.textContent") == \
            f"good_recording.edf  ({size_mb:.1f} MB)"

        page.set_input_files("#file", [])
        assert page.eval_on_selector("#submit", "e => e.disabled")
        assert text_of(page, "#filename") == ""

        page.set_input_files("#file", str(recordings["second"]))
        assert "second_recording.edf" in text_of(page, "#filename")

    def test_rendered_results_are_exactly_the_api_results(self, page, server,
                                                          recordings, model):
        expected = api_predict(server, recordings["good"])
        cue_positions = model["cue_positions"]
        arrows = {"left": "←", "right": "→", "top": "↑", "bottom": "↓"}

        page.goto(server.url + "/", wait_until="networkidle")
        page.set_input_files("#file", str(recordings["good"]))
        page.click("#submit")
        page.wait_for_url("**/predict", timeout=RUN_TIMEOUT)

        table = page.evaluate("""() => {
          const t = document.querySelector('table');
          return {
            head: [...t.tHead.rows[0].cells].map(c => c.innerText.trim().toLowerCase()),
            rows: [...t.tBodies[0].rows].map(r => [...r.cells].map(c => c.innerText.trim())),
            bars: [...t.querySelectorAll('.bar > i')].map(i => parseFloat(i.style.width)),
          };
        }""")
        assert table["head"] == ["#", "onset", "target", "movement", "confidence",
                                 "distribution"]
        assert len(table["rows"]) == expected["n_epochs"]
        for row, p, bar in zip(table["rows"], expected["predictions"], table["bars"],
                               strict=True):
            index, onset, target, movement, confidence, _ = row
            assert int(index) == p["index"]
            assert onset == f"{p['onset_seconds']:.2f}s"
            position = cue_positions[p["label"]]
            assert target == f"{arrows[position]} {position}"
            assert movement == p["label"]
            assert confidence == f"{p['confidence']:.3f}"
            assert bar == pytest.approx(p["confidence"] * 100, abs=0.051)
            assert 0 <= bar <= 100

        body = page.inner_text("body")
        assert expected["majority_label"] in body
        assert f"{expected['mean_confidence'] * 100:.1f}%" in body
        assert "not" in body and "confiden" in body.lower()
        assert text_of(page, "[aria-current='page']").upper() == "DECODE FILE"
        assert_no_broken_text(page)

    def test_double_submit_sends_one_request_and_shows_progress(self, page, server,
                                                                recordings):
        posts = []
        page.on("request", lambda r: posts.append(r.url)
                if r.method == "POST" and r.url.endswith("/predict") else None)
        page.goto(server.url + "/", wait_until="networkidle")
        page.set_input_files("#file", str(recordings["good"]))
        # Record the page's own state from inside the submit event, after the
        # page's listener has run and before the navigation tears it down.
        page.evaluate("""() => {
          document.getElementById('form').addEventListener('submit', () => {
            sessionStorage.setItem('atSubmit', JSON.stringify({
              overlay: !document.getElementById('overlay').hidden,
              disabled: document.getElementById('submit').disabled,
            }));
          });
        }""")
        page.dblclick("#submit")
        page.keyboard.press("Enter")
        page.wait_for_url("**/predict", timeout=RUN_TIMEOUT)
        page.wait_for_load_state("networkidle")

        assert len(posts) == 1, posts
        state = json.loads(page.evaluate("sessionStorage.getItem('atSubmit')"))
        assert state == {"overlay": True, "disabled": True}

    def test_keyboard_only_user_can_decode(self, page, server, recordings):
        page.goto(server.url + "/", wait_until="networkidle")
        page.set_input_files("#file", str(recordings["short"]))
        for _ in range(40):
            page.keyboard.press("Tab")
            if page.evaluate("document.activeElement.id") == "submit":
                break
        else:
            pytest.fail("the Decode button is not reachable with the keyboard")
        page.keyboard.press("Enter")
        page.wait_for_url("**/predict", timeout=RUN_TIMEOUT)
        assert page.locator("table tbody tr").count() > 0

    def test_drag_and_drop_selects_the_dropped_file(self, page, server):
        page.goto(server.url + "/", wait_until="networkidle")
        state = page.evaluate("""() => {
          const zone = document.getElementById('zone');
          const dt = new DataTransfer();
          dt.items.add(new File([new Uint8Array(4096)], 'dropped.edf'));
          zone.dispatchEvent(new DragEvent('dragenter', {dataTransfer: dt, bubbles: true, cancelable: true}));
          const hot = zone.classList.contains('hot');
          zone.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true, cancelable: true}));
          return {hot, cooled: !zone.classList.contains('hot'),
                  name: document.getElementById('filename').textContent,
                  disabled: document.getElementById('submit').disabled,
                  files: document.getElementById('file').files.length};
        }""")
        assert state == {"hot": True, "cooled": True,
                         "name": "dropped.edf  (0.0 MB)", "disabled": False,
                         "files": 1}

    @pytest.mark.parametrize(("key", "needle"), [
        ("garbage", "Could not decode"),
        ("rate128", "128 Hz"),
    ])
    def test_rejected_recording_is_explained_with_a_way_back(self, page, server,
                                                             recordings, key,
                                                             needle):
        page.goto(server.url + "/", wait_until="networkidle")
        page.set_input_files("#file", str(recordings[key]))
        page.click("#submit")
        page.wait_for_url("**/predict", timeout=RUN_TIMEOUT)
        assert page.title() == "Decoded"
        body = page.inner_text("body")
        assert "Rejected" in body and needle in body
        assert "Traceback" not in body
        page.click("text=Try another file")
        page.wait_for_url(server.url + "/")
        assert page.eval_on_selector("#submit", "e => e.disabled")

    def test_a_recording_without_cues_warns_exactly_once(self, page, server,
                                                         recordings):
        page.goto(server.url + "/", wait_until="networkidle")
        page.set_input_files("#file", str(recordings["no_cues"]))
        page.click("#submit")
        page.wait_for_url("**/predict", timeout=RUN_TIMEOUT)
        warnings = page.locator(".notice-warn").all_inner_texts()
        assert len(warnings) == 1, warnings
        assert "No cue markers" in warnings[0]
        assert "sliding window" in page.inner_text("body")

    HOSTILE = 'x.<img src=x onerror="window.__pwned=1">'

    def test_hostile_filename_is_never_executed(self, page, server):
        page.goto(server.url + "/", wait_until="networkidle")
        page.set_input_files("#file", {"name": self.HOSTILE,
                                       "mimeType": "application/octet-stream",
                                       "buffer": b"\x00" * 64})
        assert page.locator("#zone img").count() == 0
        assert self.HOSTILE in text_of(page, "#filename")
        page.click("#submit")
        page.wait_for_url("**/predict", timeout=RUN_TIMEOUT)
        assert page.evaluate("window.__pwned") is None
        assert page.locator("img[onerror]").count() == 0
        # A name without .edf is refused before it is echoed; what matters is
        # that the refusal page rendered and nothing was interpreted.
        assert "Rejected" in page.inner_text("body")
        assert "onerror=" not in page.evaluate("document.body.innerHTML")

    def test_oversized_file_is_refused_before_it_is_sent(self, page, server):
        posts = []
        page.on("request", lambda r: posts.append(r.url)
                if r.method == "POST" else None)
        page.goto(server.url + "/", wait_until="networkidle")
        assert "max 4 MB" in page.inner_text("#zone")
        page.set_input_files("#file", HUGE_UPLOAD)
        assert page.eval_on_selector("#submit", "e => e.disabled")
        assert "4 MB upload limit" in text_of(page, "#status")
        assert page.eval_on_selector("#status", "e => e.classList.contains('is-bad')")
        # Enter on the form must not get past the check either.
        page.focus("#submit")
        page.keyboard.press("Enter")
        page.wait_for_timeout(500)
        assert posts == [] and page.url == server.url + "/"

        # Choosing a file that fits clears the refusal.
        page.set_input_files("#file", {"name": "ok.edf",
                                       "mimeType": "application/octet-stream",
                                       "buffer": b"\x00" * 1024})
        assert not page.eval_on_selector("#submit", "e => e.disabled")
        assert not page.eval_on_selector("#status", "e => e.classList.contains('is-bad')")

    def test_oversized_upload_that_bypasses_the_page_check_gets_a_page(self, page,
                                                                       server):
        page.goto(server.url + "/", wait_until="networkidle")
        page.set_input_files("#file", HUGE_UPLOAD)
        # form.submit() skips the submit event and the disabled button, which
        # is what an old cached page or a scripted client would do. Called via
        # the prototype: the button named "submit" shadows form.submit.
        with page.expect_navigation(timeout=RUN_TIMEOUT):
            page.evaluate("HTMLFormElement.prototype.submit.call("
                          "document.getElementById('form'))")
        body = page.inner_text("body")
        assert not body.lstrip().startswith("{")
        assert "Rejected" in body and "4 MB limit" in body
        page.click("text=Try another file")
        page.wait_for_url(server.url + "/")

    def test_back_from_a_result_leaves_a_usable_form(self, page, server,
                                                     recordings):
        page.goto(server.url + "/", wait_until="networkidle")
        page.set_input_files("#file", str(recordings["short"]))
        page.click("#submit")
        page.wait_for_url("**/predict", timeout=RUN_TIMEOUT)
        page.go_back()
        page.wait_for_load_state("load")
        assert page.url == server.url + "/"
        overlay_visible = page.eval_on_selector("#overlay", "e => !e.hidden")
        assert not overlay_visible, "the decoding veil is still covering the page"
        page.set_input_files("#file", str(recordings["short"]))
        assert not page.eval_on_selector("#submit", "e => e.disabled")


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #


class TestNavigation:
    def test_round_trips_between_pages_leave_nothing_hidden(self, page, server):
        page.goto(server.url + "/", wait_until="networkidle")
        for _ in range(3):
            page.click("a.navlink[href='/live']")
            page.wait_for_url(server.url + "/live")
            page.wait_for_function("() => " "document.documentElement.dataset.world",
                                   timeout=120_000)
            assert text_of(page, "[aria-current='page']").upper() == "LIVE"
            page.click("a.navlink[href='/']")
            page.wait_for_url(server.url + "/")
            assert text_of(page, "[aria-current='page']").upper() == "DECODE FILE"
        assert not page.evaluate(
            "document.documentElement.hasAttribute('data-leaving')")

        page.go_back()
        page.wait_for_load_state("load")
        page.wait_for_timeout(600)
        assert not page.evaluate(
            "document.documentElement.hasAttribute('data-leaving')")
        page.go_forward()
        page.wait_for_load_state("load")
        # The content fades in with a staggered entrance animation; judge the
        # page once it has finished, not mid-fade.
        page.wait_for_function(
            "() => " "document.getAnimations().every(a => a.playState !== 'running')",
            timeout=10_000)
        assert page.evaluate("""() => [...document.querySelectorAll('main.wrap > *, .rig > *')]
              .every(e => +getComputedStyle(e).opacity === 1)""")

    def test_clicking_the_current_page_link_does_nothing(self, page, server):
        page.goto(server.url + "/", wait_until="networkidle")
        navigations = []
        page.on("framenavigated", lambda f: navigations.append(f.url))
        page.click("a.navlink[href='/']")
        page.wait_for_timeout(500)
        assert navigations == []
        assert not page.evaluate(
            "document.documentElement.hasAttribute('data-leaving')")

    def test_modified_clicks_are_left_to_the_browser(self, page, server):
        page.goto(server.url + "/", wait_until="networkidle")
        with page.context.expect_page() as new_tab:
            page.click("a.navlink[href='/live']", modifiers=["Control"])
        new_tab.value.close()
        assert page.url == server.url + "/"
        assert not page.evaluate(
            "document.documentElement.hasAttribute('data-leaving')")

    def test_reduced_motion_navigates_without_animation(self, browser, server):
        context, page = _open(browser, reduced_motion="reduce")
        try:
            page.goto(server.url + "/", wait_until="networkidle")
            started = time.perf_counter()
            page.click("a.navlink[href='/live']")
            page.wait_for_url(server.url + "/live")
            assert time.perf_counter() - started < 5
            assert not page.errors
        finally:
            context.close()


# --------------------------------------------------------------------------- #
# Live page
# --------------------------------------------------------------------------- #


class TestLiveIdle:
    def test_idle_instrument_claims_nothing(self, page, server, model):
        goto_live(page, server)
        assert page.evaluate("document.documentElement.dataset.world") == "live"
        assert page.eval_on_selector("#start", "e => e.disabled")
        assert page.eval_on_selector("#stop", "e => e.disabled")
        assert text_of(page, "#status") == "Idle"
        assert text_of(page, "#ov-status").lower() == "idle"
        assert "none yet" in text_of(page, "#log")
        for selector in ("#m-decisions", "#m-windows", "#m-timeouts"):
            assert text_of(page, selector) == "0"
        levels = page.eval_on_selector_all("#keyed tr td:nth-child(3)",
                                           "c => c.map(x => x.innerText.trim())")
        assert levels and all(v == "—" for v in levels)
        assert text_of(page, "#ph-axis") not in {"", "—"}
        assert f"{model['performance']['cross_subject_accuracy'] * 100:.1f}%" in \
            page.inner_text(".strip")
        assert_no_broken_text(page)

    def test_clicking_controls_with_nothing_loaded_is_harmless(self, page, server):
        goto_live(page, server)
        page.eval_on_selector("#stop", "e => e.click()")
        page.eval_on_selector("#form", "f => f.requestSubmit()")
        page.wait_for_timeout(300)
        assert text_of(page, "#status") == "Idle"
        assert page.eval_on_selector("#start", "e => e.disabled")

    def test_accessibility_basics(self, page, server):
        goto_live(page, server)
        assert page.eval_on_selector("#world", "e => e.getAttribute('aria-hidden')") == "true"
        assert page.eval_on_selector("#region", "e => e.getAttribute('aria-live')") == "polite"
        assert page.locator("[aria-current='page']").count() == 1
        unlabelled = page.evaluate("""() => [...document.querySelectorAll('select, input, button')]
            .filter(e => !(e.closest('label') || e.labels?.length
                           || e.getAttribute('aria-label') || e.innerText.trim()))
            .map(e => e.id || e.outerHTML.slice(0, 60))""")
        assert unlabelled == []

    def test_selecting_a_structure_draws_a_callout_on_screen(self, page, server):
        goto_live(page, server)
        box = page.eval_on_selector("#stage", """e => { const r = e.getBoundingClientRect();
                                                         return [r.x, r.y, r.width, r.height]; }""")
        for dx, dy in [(0.5, 0.45), (0.45, 0.5), (0.55, 0.4), (0.5, 0.55), (0.42, 0.42),
                       (0.6, 0.5), (0.4, 0.6)]:
            page.mouse.click(box[0] + box[2] * dx, box[1] + box[3] * dy)
            page.wait_for_timeout(700)
            if not page.eval_on_selector("#region", "e => e.hidden"):
                break
        else:
            pytest.fail("no click on the specimen selected a structure")
        card = page.evaluate("""() => {
          const root = document.getElementById('region');
          const r = root.querySelector('.callout-inner').getBoundingClientRect();
          return {x: r.x, y: r.y, w: r.width, h: r.height,
                  name: root.querySelector('.callout-name').innerText,
                  note: root.querySelector('.callout-note').innerText};
        }""")
        assert card["x"] >= -1 and card["y"] >= -1
        assert card["x"] + card["w"] <= 1601 and card["y"] + card["h"] <= 1001
        assert len(card["name"]) > 2 and len(card["note"]) > 20

        page.mouse.move(box[0] + box[2] / 2, box[1] + box[3] / 2)
        page.mouse.down()
        for i in range(8):
            page.mouse.move(box[0] + box[2] / 2 + i * 25, box[1] + box[3] / 2 + i * 4)
        page.mouse.up()
        page.wait_for_timeout(300)


class TestLiveRun:
    def test_a_full_run_shows_exactly_what_the_stream_api_computes(
            self, page, server, recordings, model):
        events = api_stream(server, recordings["good"], speed=0, step=0.5,
                            threshold=0.9, band_power=1, raw=1)
        windows = [e for e in events if e.get("type") == "window"]
        decisions = [w["decision"] for w in windows if w.get("decision")]
        committed = [d for d in decisions if not d["timed_out"]]
        timeouts = [d for d in decisions if d["timed_out"]]
        assert committed, "fixture recording must produce at least one decision"

        goto_live(page, server)
        start_run(page, recordings["good"])
        wait_for_state(page, "finished")

        assert text_of(page, "#status") == f"{events[-1]['n_windows']} windows decoded"
        assert int(text_of(page, "#m-decisions")) == len(committed)
        assert int(text_of(page, "#m-timeouts")) == len(timeouts)
        assert text_of(page, "#m-mean") == (
            f"{sum(d['n_windows'] for d in committed) / len(committed):.1f}")

        rows = page.eval_on_selector_all(
            "#log tr", "rs => rs.map(r => [...r.cells].map(c => c.innerText.trim()))")
        assert len(rows) == min(60, len(decisions))
        newest_first = list(reversed(decisions))[:60]
        decided_at = [w["onset_seconds"] for w in windows if w.get("decision")][::-1]
        for row, d, at in zip(rows, newest_first, decided_at, strict=False):
            assert row[1] == f"{at:.1f}s"
            assert row[2] == ("timed out" if d["timed_out"] else d["label"])
            assert row[3] == f"{d['confidence']:.2f}"
            assert int(row[4]) == d["n_windows"]
        assert [int(r[0]) for r in rows] == list(range(len(decisions), len(decisions) - len(rows), -1))

        last = committed[-1]
        assert text_of(page, "#ld-class") == last["label"]
        position = model["cue_positions"][last["label"]]
        assert text_of(page, "#ld-cue").endswith(position)

        final_posterior = windows[-1]["posterior"]
        shown = page.evaluate("""() => Object.fromEntries(
            [...document.querySelectorAll('.meter .v')].map(v => [v.dataset.for, v.textContent]))""")
        assert shown == {c: f"{p:.2f}" for c, p in final_posterior.items()}
        assert text_of(page, "#ov-time") == f"{windows[-1]['onset_seconds']:.1f} s"

        levels = page.eval_on_selector_all("#keyed tr td:nth-child(3)",
                                           "c => c.map(x => x.innerText.trim())")
        assert all(0.0 <= float(v) <= 1.0 for v in levels)
        # The note is styled uppercase, which turns the micro sign into a
        # capital Mu, so compare case-insensitively on the parts that survive.
        scope = text_of(page, "#scope-note").upper()
        assert "160 HZ" in scope and scope.endswith("V")
        assert page.eval_on_selector("#start", "e => !e.disabled")
        assert page.eval_on_selector("#stop", "e => e.disabled")
        assert_no_broken_text(page)

    def test_stop_halts_the_stream_and_releases_the_controls(self, page, server,
                                                             recordings):
        failed = []
        page.on("requestfailed", lambda r: failed.append(r.url)
                if "/api/v1/stream" in r.url else None)
        goto_live(page, server)
        start_run(page, recordings["good"], speed="1")
        page.wait_for_function(
            "() => " "+document.getElementById('m-windows').textContent > 1", timeout=60_000)
        assert page.eval_on_selector("#start", "e => e.disabled")

        page.click("#stop")
        frozen = text_of(page, "#ov-time")
        page.wait_for_timeout(2500)
        assert text_of(page, "#ov-time") == frozen
        assert text_of(page, "#status") == "stopped"
        assert text_of(page, "#ov-status") == "stopped"
        assert page.eval_on_selector("#start", "e => !e.disabled")
        assert page.eval_on_selector("#stop", "e => e.disabled")
        assert failed, "the stream request was not actually aborted"

    def test_a_second_run_does_not_inherit_the_first(self, page, server,
                                                     recordings):
        goto_live(page, server)
        start_run(page, recordings["short"])
        wait_for_state(page, "finished")
        page.set_input_files("#file", str(recordings["second"]))
        reset = page.evaluate("""() => {
          document.getElementById('start').click();
          const t = id => document.getElementById(id).textContent;
          return {decisions: t('m-decisions'), timeouts: t('m-timeouts'),
                  windows: t('m-windows'), mean: t('m-mean'), log: t('log').trim(),
                  cls: t('ld-class'), status: t('status')};
        }""")
        assert reset == {"decisions": "0", "timeouts": "0", "windows": "0",
                         "mean": "—", "log": "none yet", "cls": "—",
                         "status": "decoding…"}
        wait_for_state(page, "finished")

    def test_changing_the_file_mid_run_cannot_start_a_second_stream(
            self, page, server, recordings):
        streams = []
        page.on("request", lambda r: streams.append(r.url)
                if "/api/v1/stream" in r.url else None)
        goto_live(page, server)
        start_run(page, recordings["good"], speed="1")
        page.wait_for_function(
            "() => " "+document.getElementById('m-windows').textContent > 0", timeout=60_000)
        page.set_input_files("#file", str(recordings["second"]))
        try:
            assert page.eval_on_selector("#start", "e => e.disabled")
        finally:
            page.click("#stop", force=True)

    def test_an_undecodable_recording_reports_the_server_message(self, page,
                                                                 server,
                                                                 recordings):
        goto_live(page, server)
        start_run(page, recordings["garbage"])
        page.wait_for_function(
            "() => " "document.getElementById('status').classList.contains('is-bad')",
            timeout=60_000)
        assert text_of(page, "#status") == "the recording could not be decoded"
        page.wait_for_function("() => " "!document.getElementById('start').disabled")

    def test_an_error_does_not_leave_the_state_reading_decoding(self, page, server,
                                                                recordings):
        goto_live(page, server)
        start_run(page, recordings["garbage"])
        page.wait_for_function("() => " "!document.getElementById('start').disabled",
                               timeout=60_000)
        assert text_of(page, "#ov-status").lower() != "decoding"

    def test_a_refused_upload_is_reported_on_the_live_page(self, page, server,
                                                           recordings):
        goto_live(page, server)
        start_run(page, recordings["text"])
        page.wait_for_function("() => " "!document.getElementById('start').disabled",
                               timeout=60_000)
        assert "unsupported file type" in text_of(page, "#status")

    def test_network_failure_is_reported_and_recoverable(self, page, server,
                                                         recordings):
        goto_live(page, server)
        page.route("**/api/v1/stream*", lambda route: route.abort("connectionreset"))
        start_run(page, recordings["short"])
        page.wait_for_function(
            "() => " "document.getElementById('status').textContent === 'connection failed'")
        assert page.eval_on_selector("#status", "e => e.classList.contains('is-bad')")
        assert page.eval_on_selector("#start", "e => !e.disabled")
        assert page.eval_on_selector("#stop", "e => e.disabled")

        page.unroute("**/api/v1/stream*")
        page.click("#start")
        wait_for_state(page, "finished")

    def test_navigating_away_mid_stream_tears_down_cleanly(self, page, server,
                                                           recordings):
        goto_live(page, server)
        start_run(page, recordings["good"], speed="1")
        page.wait_for_function(
            "() => " "+document.getElementById('m-windows').textContent > 0", timeout=60_000)
        page.click("a.navlink[href='/']")
        page.wait_for_url(server.url + "/")
        assert page.is_visible("#submit")
        page.click("a.navlink[href='/live']")
        page.wait_for_url(server.url + "/live")
        page.wait_for_function("() => " "document.documentElement.dataset.world")
        assert "none yet" in text_of(page, "#log")
        assert text_of(page, "#ov-status").lower() == "idle"


def _ndjson(*frames) -> str:
    return "".join((f if isinstance(f, str) else json.dumps(f)) + "\n" for f in frames)


def _window(i, posterior, decision=None):
    frame = {"type": "window", "window": i, "onset_seconds": i * 0.5,
             "probabilities": posterior, "top_label": max(posterior, key=posterior.get),
             "posterior": posterior}
    if decision:
        frame["decision"] = decision
    return frame


class TestLiveAgainstAMockedStream:
    """The page's stream parser, fed frames no honest server would send."""

    def _run_with(self, page, server, recordings, body):
        goto_live(page, server)
        page.route("**/api/v1/stream*", lambda route: route.fulfill(
            status=200, content_type="application/x-ndjson", body=body))
        start_run(page, recordings["short"])

    def test_malformed_and_unknown_frames_are_survivable(self, page, server,
                                                         recordings):
        classes = ["left_fist", "right_fist"]
        body = _ndjson(
            {"type": "start", "classes": classes,
             "cue_positions": {"left_fist": "left", "right_fist": "right"},
             "cue_axes": ["horizontal"], "n_windows": 4, "window_seconds": 3,
             "step_seconds": 0.5},
            "{this is not json",
            {"type": "mystery", "payload": [1, 2, 3]},
            _window(0, {"left_fist": 0.6, "right_fist": 0.4}),
            _window(1, {"left_fist": 0.5, "right_fist": 0.5},
                    {"label": None, "confidence": 0.5, "n_windows": 2,
                     "elapsed_seconds": 0.1, "posterior": {}, "timed_out": True}),
            _window(2, {"left_fist": 0.97, "right_fist": 0.03},
                    {"label": "left_fist", "confidence": 0.97, "n_windows": 1,
                     "elapsed_seconds": 0.1, "posterior": {}, "timed_out": False}),
            {"type": "end", "n_windows": 3},
        )
        self._run_with(page, server, recordings, body)
        wait_for_state(page, "finished", timeout=30_000)
        assert text_of(page, "#m-decisions") == "1"
        assert text_of(page, "#m-timeouts") == "1"
        rows = page.eval_on_selector_all(
            "#log tr", "rs => rs.map(r => [...r.cells].map(c => c.innerText.trim()))")
        assert [r[2] for r in rows] == ["left_fist", "timed out"]
        assert text_of(page, "#ld-cue").endswith("left")
        assert text_of(page, "#status") == "3 windows decoded"

    def test_a_frame_split_across_network_chunks_is_reassembled(self, page, server,
                                                               recordings):
        """The whole body arrives at once here; the parser must still split it
        on newlines only, never on chunk boundaries, and keep the tail."""
        classes = ["left_fist", "right_fist"]
        frames = [
            {"type": "start", "classes": classes, "cue_positions": {},
             "cue_axes": [], "n_windows": 50, "window_seconds": 3,
             "step_seconds": 0.5},
            *[_window(i, {"left_fist": 0.5, "right_fist": 0.5}) for i in range(50)],
            {"type": "end", "n_windows": 50},
        ]
        self._run_with(page, server, recordings, _ndjson(*frames))
        wait_for_state(page, "finished", timeout=30_000)
        assert text_of(page, "#ov-time") == "24.5 s"
        assert text_of(page, "#status") == "50 windows decoded"

    def test_a_stream_that_ends_without_an_end_frame_releases_the_controls(
            self, page, server, recordings):
        body = _ndjson({"type": "start", "classes": ["a", "b"], "cue_positions": {},
                        "cue_axes": [], "n_windows": 1, "window_seconds": 3,
                        "step_seconds": 0.5},
                       _window(0, {"a": 0.5, "b": 0.5}))
        self._run_with(page, server, recordings, body)
        page.wait_for_function("() => " "!document.getElementById('start').disabled",
                               timeout=30_000)
        assert page.eval_on_selector("#stop", "e => e.disabled")

    def test_class_names_are_never_interpreted_as_markup(self, page, server,
                                                         recordings):
        hostile = 'x"><img src=x onerror="window.__pwned=1">'
        body = _ndjson({"type": "start", "classes": [hostile, "y"],
                        "cue_positions": {}, "cue_axes": [], "n_windows": 1,
                        "window_seconds": 3, "step_seconds": 0.5},
                       _window(0, {hostile: 0.5, "y": 0.5}),
                       {"type": "end", "n_windows": 1})
        self._run_with(page, server, recordings, body)
        wait_for_state(page, "finished", timeout=30_000)
        page.wait_for_timeout(300)
        assert page.evaluate("window.__pwned") is None
        assert page.locator("#bars img").count() == 0


class TestHardenedLivePage:
    """The live page's fixes, pushed: refusals, late settles, cut-off streams."""

    def _mock(self, page, server, **fulfill):
        goto_live(page, server)
        page.route("**/api/v1/stream*", lambda route: route.fulfill(**fulfill))

    def test_a_busy_server_is_explained_not_left_decoding(self, page, server,
                                                          recordings):
        message = "too many paced replays are running; try again shortly"
        self._mock(page, server, status=503, content_type="application/json",
                   body=json.dumps({"error": "service_unavailable",
                                    "message": message}))
        start_run(page, recordings["short"])
        page.wait_for_function("() => " "!document.getElementById('start').disabled",
                               timeout=30_000)
        assert text_of(page, "#status") == message
        assert page.eval_on_selector("#status", "e => e.classList.contains('is-bad')")
        assert text_of(page, "#ov-status") == "error"
        assert text_of(page, "#ov-tag").lower() == "error"
        assert page.eval_on_selector("#stop", "e => e.disabled")

    def test_a_refusal_that_is_not_json_still_says_something(self, page, server,
                                                             recordings):
        self._mock(page, server, status=502, content_type="text/html",
                   body="<html><body><b>Bad gateway</b></body></html>")
        start_run(page, recordings["short"])
        page.wait_for_function("() => " "!document.getElementById('start').disabled",
                               timeout=30_000)
        assert text_of(page, "#status") == "the server refused the request (HTTP 502)"
        assert page.locator("#status b").count() == 0

    def test_the_real_replay_budget_refusal_reaches_the_page_and_recovers(
            self, page, server, recordings):
        # The budget is taken directly rather than by holding real replays
        # open: how long a held replay lasts depends on the machine, so that
        # version passed here and failed on CI, where the holders had finished
        # before the page ever asked. The semaphore is the thing under test.
        slots = server.app.extensions["bwt_replay_slots"]
        held = 0
        while slots.acquire(blocking=False):
            held += 1
        assert held, "the server has no replay budget to exhaust"
        try:
            goto_live(page, server)
            start_run(page, recordings["short"], speed="10")
            page.wait_for_function("() => " "!document.getElementById('start').disabled",
                                   timeout=30_000)
            assert "paced replays" in text_of(page, "#status")
            assert text_of(page, "#ov-status") == "error"
        finally:
            for _ in range(held):
                slots.release()

        # With the budget back, the same paced run goes through.
        page.click("#start")
        wait_for_state(page, "finished")
        assert not page.eval_on_selector("#status", "e => e.classList.contains('is-bad')")

    def test_a_stopped_run_that_settles_late_cannot_release_the_next_run(
            self, browser, server, recordings):
        context, page = _open(browser)
        page.add_init_script("""(() => {
          const real = window.fetch.bind(window);
          let calls = 0;
          window.fetch = (url, options) => {
            if (String(url).includes('/api/v1/stream') && ++calls === 1) {
              // Ignores its abort signal and fails late, like a network error
              // that arrives after the user has already moved on. The test
              // decides when: a timer here raced the test's own clicks, and
              // on a slow machine fired before Stop could be pressed.
              return new Promise((_, reject) => {
                window.__failFirstRun = () => reject(new TypeError('late failure'));
              });
            }
            return real(url, options);
          };
        })()""")
        try:
            goto_live(page, server)
            start_run(page, recordings["short"])
            click_enabled(page, "#stop")
            start_run(page, recordings["good"], speed="1")
            page.wait_for_function(
                "() => " "+document.getElementById('m-windows').textContent > 0",
                timeout=60_000)
            page.evaluate("() => window.__failFirstRun()")  # the first run fails now
            page.wait_for_timeout(500)  # let its rejection handlers run
            assert page.eval_on_selector("#start", "e => e.disabled")
            assert page.eval_on_selector("#stop", "e => !e.disabled")
            assert text_of(page, "#status") != "connection failed"
            assert text_of(page, "#ov-status") == "decoding"
            click_enabled(page, "#stop")
            assert page.errors == []
        finally:
            context.close()

    def test_a_stream_cut_off_mid_way_says_so(self, page, server, recordings):
        body = _ndjson({"type": "start", "classes": ["a", "b"], "cue_positions": {},
                        "cue_axes": [], "n_windows": 9, "window_seconds": 3,
                        "step_seconds": 0.5},
                       _window(0, {"a": 0.5, "b": 0.5}))
        self._mock(page, server, status=200, content_type="application/x-ndjson",
                   body=body)
        start_run(page, recordings["short"])
        page.wait_for_function("() => " "!document.getElementById('start').disabled",
                               timeout=30_000)
        assert text_of(page, "#ov-status") == "ended"
        assert text_of(page, "#status") == "the stream ended before it finished"

    def test_double_clicking_start_opens_one_stream(self, page, server, recordings):
        streams = []
        page.on("request", lambda r: streams.append(r.url)
                if "/api/v1/stream" in r.url else None)
        goto_live(page, server)
        page.set_input_files("#file", str(recordings["short"]))
        page.select_option("#speed", "0")
        page.dblclick("#start")
        page.keyboard.press("Enter")
        wait_for_state(page, "finished")
        assert len(streams) == 1, streams

    def test_hostile_class_names_are_text_everywhere_they_appear(self, page, server,
                                                                recordings):
        hostile = '"><img src=x onerror="window.__pwned=1">'
        committed = {"label": hostile, "confidence": 0.97, "n_windows": 2,
                     "elapsed_seconds": 0.1, "posterior": {}, "timed_out": False}
        body = _ndjson({"type": "start", "classes": [hostile, "y"],
                        "cue_positions": {hostile: "left"}, "cue_axes": [],
                        "n_windows": 2, "window_seconds": 3, "step_seconds": 0.5},
                       _window(0, {hostile: 0.6, "y": 0.4}),
                       _window(1, {hostile: 0.97, "y": 0.03}, committed),
                       {"type": "end", "n_windows": 2})
        self._mock(page, server, status=200, content_type="application/x-ndjson",
                   body=body)
        start_run(page, recordings["short"])
        wait_for_state(page, "finished", timeout=30_000)
        page.wait_for_timeout(300)
        assert page.evaluate("window.__pwned") is None
        assert page.locator("img").count() == 0
        assert page.eval_on_selector("#ld-class", "e => e.textContent") == hostile
        assert page.eval_on_selector("#log tr td:nth-child(3)", "e => e.textContent") == hostile
        assert page.eval_on_selector("#bars .n", "e => e.textContent") == hostile
        assert page.eval_on_selector("#bars .v", "e => e.dataset.for") == hostile
        assert text_of(page, "#ld-cue").endswith("left")


class TestHardenedDecodePage:
    def test_dropping_an_oversized_file_is_refused_too(self, page, server):
        page.goto(server.url + "/", wait_until="networkidle")
        state = page.evaluate("""() => {
          const zone = document.getElementById('zone');
          const dt = new DataTransfer();
          dt.items.add(new File([new Uint8Array(5 * 1048576)], 'dropped.edf'));
          zone.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true, cancelable: true}));
          return {disabled: document.getElementById('submit').disabled,
                  status: document.getElementById('status').textContent};
        }""")
        assert state["disabled"] and "4 MB upload limit" in state["status"]

    def test_the_form_state_always_matches_the_input_after_history_moves(
            self, page, server, recordings):
        page.goto(server.url + "/", wait_until="networkidle")
        page.set_input_files("#file", str(recordings["short"]))
        page.click("a.navlink[href='/live']")
        page.wait_for_url(server.url + "/live")
        for move in (page.go_back, page.go_forward, page.go_back):
            move()
            page.wait_for_load_state("load")
            if page.url != server.url + "/":
                continue
            page.wait_for_timeout(300)
            state = page.evaluate("""() => ({
              files: document.getElementById('file').files.length,
              disabled: document.getElementById('submit').disabled,
              name: document.getElementById('filename').textContent,
              veil: !document.getElementById('overlay').hidden,
            })""")
            assert state["disabled"] == (state["files"] == 0), state
            assert bool(state["name"]) == (state["files"] > 0), state
            assert not state["veil"], state

    def test_result_page_fits_the_narrowest_phone(self, browser, server, recordings):
        context, page = _open(browser, viewport={"width": 320, "height": 568},
                              is_mobile=True, has_touch=True)
        try:
            page.goto(server.url + "/", wait_until="networkidle")
            page.set_input_files("#file", str(recordings["good"]))
            page.locator("#submit").scroll_into_view_if_needed()
            page.click("#submit")
            page.wait_for_url("**/predict", timeout=RUN_TIMEOUT)
            assert page.evaluate(
                "document.documentElement.scrollWidth "
                "<= document.documentElement.clientWidth + 2")
            assert page.locator("table tbody tr").count() > 0
        finally:
            context.close()


CONTRAST_PROBE = """(selectors) => {
  const parse = (c) => (c.match(/[\\d.]+/g) || []).map(Number);
  const lum = ([r, g, b]) => [r, g, b].map((v) => {
    v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  }).reduce((s, v, i) => s + v * [0.2126, 0.7152, 0.0722][i], 0);
  const background = (el) => {
    for (let e = el; e; e = e.parentElement) {
      const c = parse(getComputedStyle(e).backgroundColor);
      if (c.length >= 3 && (c.length === 3 || c[3] > 0)) return c.slice(0, 3);
    }
    return parse(getComputedStyle(document.body).backgroundColor).slice(0, 3);
  };
  const out = [];
  for (const sel of selectors) {
    for (const el of document.querySelectorAll(sel)) {
      if (!el.offsetParent || !el.textContent.trim()) continue;
      if (el.closest(':disabled, [aria-disabled="true"]')) continue;
      const fg = lum(parse(getComputedStyle(el).color));
      const bg = lum(background(el));
      const ratio = (Math.max(fg, bg) + 0.05) / (Math.min(fg, bg) + 0.05);
      out.push([sel, el.textContent.trim().slice(0, 30), Math.round(ratio * 100) / 100]);
    }
  }
  return out;
}"""

CONTRAST_SELECTORS = [".strip-key", ".strip-val", ".panel-head", ".hint", ".row2 .k",
                      "dl.spec dt", "dl.spec dd", ".status", ".note", ".metric .sub",
                      ".metric .label", ".footer p", ".tbl th", ".log th", ".log td",
                      ".path-n", ".path-t em", ".path-foot", ".drop em", ".navlink",
                      ".lede", "h2", ".pill", ".tag"]


class TestSecurityAndAccessibility:
    def test_pages_are_served_under_the_policy_and_nothing_trips_it(
            self, page, server, recordings):
        response = page.goto(server.url + "/", wait_until="networkidle")
        assert "'nonce-" in response.headers["content-security-policy"]
        assert response.headers["x-frame-options"] == "DENY"
        page.set_input_files("#file", str(recordings["short"]))
        page.click("#submit")
        page.wait_for_url("**/predict", timeout=RUN_TIMEOUT)
        goto_live(page, server)
        assert page.evaluate("document.documentElement.dataset.world") == "live"
        start_run(page, recordings["short"])
        wait_for_state(page, "finished")
        # The page fixture fails the test on any policy violation it logged.

    def test_an_injected_inline_script_is_refused_by_the_policy(self, browser, server):
        context, page = _open(browser)
        try:
            page.goto(server.url + "/", wait_until="networkidle")
            page.evaluate("""() => {
              const s = document.createElement('script');
              s.textContent = 'window.__injected = 1';
              document.body.appendChild(s);
            }""")
            page.wait_for_timeout(300)
            assert page.evaluate("window.__injected") is None
            assert any("Content Security Policy" in e for e in page.errors)
        finally:
            context.close()

    def test_the_site_cannot_be_framed(self, browser, server):
        context, page = _open(browser)
        try:
            page.goto(server.url + "/healthz")
            page.set_content(f'<iframe id="f" src="{server.url}/" width="800" '
                             'height="600"></iframe>')
            page.wait_for_timeout(2000)
            framed = [f for f in page.frames if f is not page.main_frame]
            for frame in framed:
                try:
                    title = frame.title()
                except Exception:
                    title = ""
                assert title != "Decode a recording"
        finally:
            context.close()

    def test_the_decode_page_works_with_javascript_disabled(self, browser, server,
                                                            recordings):
        context = browser.new_context(java_script_enabled=False,
                                      viewport={"width": 1280, "height": 900})
        try:
            page = context.new_page()
            page.goto(server.url + "/")
            assert not page.eval_on_selector("#submit", "e => e.disabled")
            page.set_input_files("#file", str(recordings["short"]))
            page.click("#submit")
            page.wait_for_url("**/predict", timeout=RUN_TIMEOUT)
            assert page.locator("table tbody tr").count() > 0

            page.goto(server.url + "/live")
            assert page.is_visible(".noscript")
            assert "needs JavaScript" in page.inner_text(".noscript")
        finally:
            context.close()

    @pytest.mark.parametrize("path", ["/", "/live"])
    def test_a_keyboard_user_can_open_the_file_chooser(self, page, server, recordings,
                                                       path):
        # Listen before any key is pressed: Playwright intercepts the file
        # dialog only once a listener exists, and on the busy WebGL page that
        # set-up can otherwise land after the keypress that opens it.
        choosers = []
        page.on("filechooser", lambda chooser: choosers.append(chooser))
        if path == "/live":
            goto_live(page, server)
        else:
            page.goto(server.url + path, wait_until="networkidle")
        for _ in range(40):
            page.keyboard.press("Tab")
            if page.evaluate("document.activeElement.id") == "file":
                break
        else:
            pytest.fail("the file input is not reachable with Tab")
        outline = page.eval_on_selector("#zone", "e => getComputedStyle(e).outlineStyle")
        assert outline != "none", "keyboard focus on the file zone is invisible"
        page.keyboard.press("Space")
        deadline = time.time() + 15
        while not choosers and time.time() < deadline:
            page.wait_for_timeout(100)
        assert choosers, "Space on the focused file input did not open the chooser"
        choosers[0].set_files(str(recordings["short"]))
        button = "#submit" if path == "/" else "#start"
        assert not page.eval_on_selector(button, "e => e.disabled")

    @pytest.mark.parametrize("path", ["/", "/live"])
    def test_text_meets_wcag_aa_contrast(self, page, server, path):
        if path == "/live":
            goto_live(page, server)
        else:
            page.goto(server.url + path, wait_until="networkidle")
        page.wait_for_function(
            "() => " "document.getAnimations().every(a => a.playState !== 'running')",
            timeout=10_000)
        measured = page.evaluate(CONTRAST_PROBE, CONTRAST_SELECTORS)
        assert measured, "nothing was measured"
        failing = [m for m in measured if m[2] < 4.5]
        assert failing == [], failing

    @pytest.mark.parametrize("name", ["‮FDE.edf", "脑电图🧠 recording.edf",
                                      "x" * 300 + ".edf"])
    def test_unusual_file_names_are_shown_exactly_and_never_widen_the_page(
            self, browser, server, name):
        context, page = _open(browser, viewport={"width": 390, "height": 844},
                              is_mobile=True, has_touch=True)
        try:
            page.goto(server.url + "/", wait_until="networkidle")
            page.set_input_files("#file", {"name": name,
                                           "mimeType": "application/octet-stream",
                                           "buffer": b"\x00" * 10})
            shown = page.eval_on_selector("#filename", "e => e.textContent")
            assert shown == f"{name}  (0.0 MB)"
            assert page.evaluate("document.documentElement.scrollWidth "
                                 "<= document.documentElement.clientWidth + 2")
            assert page.errors == []
        finally:
            context.close()

    def test_three_thousand_frames_keep_the_page_bounded(self, page, server,
                                                         recordings):
        classes = ["left_fist", "right_fist"]
        frames = [{"type": "start", "classes": classes,
                   "cue_positions": {"left_fist": "left", "right_fist": "right"},
                   "cue_axes": ["horizontal"], "n_windows": 3000,
                   "window_seconds": 3, "step_seconds": 0.5}]
        for i in range(3000):
            decision = None
            if i % 10 == 9:
                decision = {"label": classes[i % 20 == 19], "confidence": 0.95,
                            "n_windows": 10, "elapsed_seconds": 0.1,
                            "posterior": {}, "timed_out": False}
            frames.append(_window(i, {"left_fist": 0.6, "right_fist": 0.4}, decision))
        frames.append({"type": "end", "n_windows": 3000})
        goto_live(page, server)
        baseline = page.evaluate("document.getElementsByTagName('*').length")
        page.route("**/api/v1/stream*", lambda route: route.fulfill(
            status=200, content_type="application/x-ndjson", body=_ndjson(*frames)))
        start_run(page, recordings["short"])
        wait_for_state(page, "finished", timeout=RUN_TIMEOUT)
        assert text_of(page, "#status") == "3000 windows decoded"
        assert page.locator("#log tr").count() == 60
        assert text_of(page, "#m-decisions") == "300"
        grown = page.evaluate("document.getElementsByTagName('*').length") - baseline
        # The log keeps its 60 newest rows (a tr and five cells each); nothing
        # else may grow with the length of the stream.
        assert grown <= 60 * 6 + 40, grown
        heap = page.evaluate("performance.memory ? performance.memory.usedJSHeapSize : 0")
        assert heap < 200 * 1024 * 1024

    def test_rapid_start_stop_cycles_leak_no_server_replays(self, page, server,
                                                            recordings):
        goto_live(page, server)
        page.set_input_files("#file", str(recordings["good"]))
        page.select_option("#speed", "1")
        for _ in range(15):
            page.click("#start")
            page.wait_for_function(
                "() => " "+document.getElementById('m-windows').textContent > 0"
                " || document.getElementById('ov-status').textContent === 'error'",
                timeout=60_000)
            page.click("#stop")
        assert text_of(page, "#ov-status") == "stopped"
        assert page.eval_on_selector("#start", "e => !e.disabled")

        # Every replay the page abandoned must give its slot back once the
        # server notices the hang-up on its next paced write. Read the budget
        # itself: probing with requests would hold slots of its own.
        slots = server.app.extensions["bwt_replay_slots"]
        budget = max(1, server.app.extensions["bwt_config"].serve.threads - 1)
        deadline = time.time() + 30
        while slots._value != budget and time.time() < deadline:
            time.sleep(0.25)
        assert slots._value == budget, f"{budget - slots._value} replay slot(s) leaked"

    def test_going_offline_mid_session_is_reported_and_recoverable(self, page, server,
                                                                  recordings):
        goto_live(page, server)
        page.context.set_offline(True)
        try:
            start_run(page, recordings["short"])
            page.wait_for_function(
                "() => " "document.getElementById('status').textContent === 'connection failed'",
                timeout=30_000)
            assert text_of(page, "#ov-status") == "error"
        finally:
            page.context.set_offline(False)
        page.click("#start")
        wait_for_state(page, "finished")


class TestWithoutWebGL:
    def test_decoding_works_when_the_specimen_cannot_render(self, browser,
                                                            playwright_instance,
                                                            server, recordings):
        # Depends on the shared `browser` fixture only so that a missing
        # Chromium skips this test the same way it skips the rest.
        browser = playwright_instance.chromium.launch(
            args=["--disable-gpu", "--disable-webgl", "--disable-3d-apis",
                  "--disable-software-rasterizer"])
        context, page = _open(browser)
        try:
            goto_live(page, server)
            assert page.evaluate("document.documentElement.dataset.world") == "off"
            assert text_of(page, "#ov-status") == "no webgl"
            start_run(page, recordings["good"])
            wait_for_state(page, "finished")
            assert int(text_of(page, "#m-decisions")) > 0
            assert page.errors == []
        finally:
            context.close()
            browser.close()


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


OVERFLOW_PROBE = """() => {
  const sideways = document.documentElement.scrollWidth
                   > document.documentElement.clientWidth + 2;
  const clipped = [...document.querySelectorAll('.panel, .card, .metric, .btn')]
    .filter(e => e.offsetParent !== null)
    .filter(e => {
      const r = e.getBoundingClientRect();
      return r.right > innerWidth + 2 || r.left < -2;
    })
    .map(e => (e.className || e.tagName).slice(0, 30));
  return {sideways, clipped: [...new Set(clipped)].slice(0, 5)};
}"""


PHONE = pytest.param(390, 844, id="390-844")


class TestLayout:
    @pytest.mark.parametrize(("width", "height"), [
        (3840, 2160), (1920, 1080), (1440, 900), (1366, 768), (1280, 720),
        (1024, 768),
    ])
    @pytest.mark.parametrize("path", ["/", "/live"])
    def test_desktop_viewports_fit(self, browser, server, width, height, path):
        context, page = _open(browser, viewport={"width": width, "height": height})
        try:
            page.goto(server.url + path, wait_until="networkidle")
            if path == "/live":
                page.wait_for_function("() => " "document.documentElement.dataset.world",
                                       timeout=120_000)
            page.wait_for_timeout(500)
            probe = page.evaluate(OVERFLOW_PROBE)
            assert probe == {"sideways": False, "clipped": []}, probe
            assert page.errors == []
        finally:
            context.close()

    @pytest.mark.parametrize(("width", "height"), [(768, 1024), PHONE, (320, 568)])
    def test_decode_page_is_usable_on_a_tablet_and_a_phone(self, browser, server,
                                                           recordings, width,
                                                           height):
        context, page = _open(browser, viewport={"width": width, "height": height},
                              is_mobile=width < 500, has_touch=width < 500)
        try:
            page.goto(server.url + "/", wait_until="networkidle")
            probe = page.evaluate(OVERFLOW_PROBE)
            assert probe == {"sideways": False, "clipped": []}, probe
            page.set_input_files("#file", str(recordings["short"]))
            page.locator("#submit").scroll_into_view_if_needed()
            page.click("#submit")
            page.wait_for_url("**/predict", timeout=RUN_TIMEOUT)
            assert page.locator("table tbody tr").count() > 0
            assert not page.evaluate("document.documentElement.scrollWidth "
                                     "> document.documentElement.clientWidth + 2")
        finally:
            context.close()

    @pytest.mark.parametrize(("width", "height"), [(768, 1024), PHONE, (320, 568)])
    def test_live_page_on_a_tablet_and_a_phone(self, browser, server, width, height):
        context, page = _open(browser, viewport={"width": width, "height": height},
                              is_mobile=width < 500, has_touch=width < 500)
        try:
            goto_live(page, server)
            probe = page.evaluate(OVERFLOW_PROBE)
            assert probe == {"sideways": False, "clipped": []}, probe
            assert page.is_visible("#start")
        finally:
            context.close()

    @pytest.mark.parametrize(("width", "height"), [(360, 740), (414, 896), (844, 390)])
    @pytest.mark.parametrize("path", ["/", "/live"])
    def test_every_small_screen_scrolls_only_vertically(self, browser, server,
                                                         width, height, path):
        context, page = _open(browser, viewport={"width": width, "height": height},
                              is_mobile=width < 500, has_touch=True)
        try:
            page.goto(server.url + path, wait_until="networkidle")
            if path == "/live":
                page.wait_for_function("() => " "document.documentElement.dataset.world",
                                       timeout=120_000)
            page.wait_for_function(
                "() => " "document.getAnimations().every(a => a.playState !== 'running')",
                timeout=10_000)
            probe = page.evaluate(OVERFLOW_PROBE)
            assert probe == {"sideways": False, "clipped": []}, probe
            # Everything stays reachable: the last thing on the page can be
            # scrolled to and is on screen once it is.
            last = "#scope" if path == "/live" else ".footer"
            page.locator(last).scroll_into_view_if_needed()
            assert page.is_visible(last)
            assert page.evaluate(
                "document.documentElement.scrollWidth "
                "<= document.documentElement.clientWidth + 2")
            assert page.errors == []
        finally:
            context.close()
