"""Guards on the browser assets.

These exist because of a real, repeated failure: a comment written inside a
GLSL template literal used backticks, which closed the JavaScript string and
took the entire page down with a syntax error. It happened twice, and both
times nothing caught it until the page was driven in a browser -- the module
still returned HTTP 200, so the server looked healthy.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "static"
MODULES = ["brain3d.js", "inkfont.js"]


def _shader_literals(source: str) -> list[str]:
    """Every backtick-delimited template literal that looks like GLSL."""
    out = []
    for match in re.finditer(r"`([^`]*)`", source, re.S):
        body = match.group(1)
        if "void main" in body or "gl_Position" in body or "gl_FragColor" in body:
            out.append(body)
    return out


@pytest.mark.parametrize("name", MODULES)
def test_module_has_balanced_template_literals(name: str) -> None:
    """An odd number of backticks means a literal is left open."""
    source = (STATIC / name).read_text(encoding="utf-8")
    assert source.count("`") % 2 == 0, (
        f"{name} has an unbalanced backtick, so a template literal never "
        "closes. Check for backticks inside a shader comment."
    )


@pytest.mark.parametrize("name", MODULES)
def test_shader_source_contains_no_backticks(name: str) -> None:
    """The specific mistake: prose quoting inside GLSL kills the whole file."""
    source = (STATIC / name).read_text(encoding="utf-8")
    for shader in _shader_literals(source):
        assert "`" not in shader, f"{name}: backtick inside shader source"


@pytest.mark.parametrize("name", MODULES)
def test_module_parses(name: str) -> None:
    """Parse as a real ES module, which is how the browser will read it."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    path = STATIC / name
    # --check infers module vs script from content; .js with imports is fine.
    result = subprocess.run(
        [node, "--check", str(path)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, f"{name} does not parse:\n{result.stderr}"


def test_cortex_mesh_header_matches_its_payload() -> None:
    """The baked mesh must describe itself, or the loader reads past its end."""
    blob = (STATIC / "cortex.bin").read_bytes()
    assert blob[:4] == b"CTX1", "cortex.bin is missing its magic"

    n_verts, n_tris = struct.unpack_from("<II", blob, 4)
    assert n_verts > 0 and n_tris > 0

    # header + positions + depth + region (padded to 4) + indices
    expected = 12 + n_verts * 12 + n_verts * 4
    expected += n_verts + (-n_verts % 4)
    expected += n_tris * 12
    assert len(blob) == expected, (
        f"cortex.bin is {len(blob)} bytes but its header implies {expected}"
    )


def test_cortex_regions_are_within_range() -> None:
    """Region ids index shader uniform arrays; out of range corrupts lighting."""
    blob = (STATIC / "cortex.bin").read_bytes()
    n_verts, _ = struct.unpack_from("<II", blob, 4)
    start = 12 + n_verts * 12 + n_verts * 4
    regions = set(blob[start:start + n_verts])
    assert regions, "no region data"
    assert max(regions) < 11, f"region id out of range: {max(regions)}"
    # Primary motor cortex is the structure this whole decoder is about.
    assert 1 in regions and 2 in regions, "motor cortex missing from the atlas"


def test_region_tables_agree_between_python_and_javascript() -> None:
    """The build script and the viewer must number regions identically."""
    js = (STATIC / "brain3d.js").read_text(encoding="utf-8")
    block = re.search(r"export const REGION = \{(.*?)\};", js, re.S)
    assert block, "REGION table not found in brain3d.js"
    js_ids = dict(re.findall(r"(\w+):\s*(\d+)", block.group(1)))

    build = (Path(__file__).resolve().parents[1] / "scripts" / "build_cortex.py")
    py = build.read_text(encoding="utf-8")
    for name, value in js_ids.items():
        assert re.search(rf"^{name} = {value}$", py, re.M), (
            f"{name} is {value} in brain3d.js but differs in build_cortex.py"
        )

    names = re.search(r"export const REGION_NAME = \[(.*?)\];", js, re.S)
    notes = re.search(r"export const REGION_NOTE = \[(.*?)\];", js, re.S)
    assert names and notes
    n_names = len(re.findall(r"'", names.group(1))) // 2
    n_notes = len(re.findall(r"'", notes.group(1))) // 2
    assert n_names == len(js_ids), "a region has no name"
    assert n_notes == len(js_ids), "a region has no note"
