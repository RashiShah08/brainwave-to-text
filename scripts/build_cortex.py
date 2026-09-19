"""Bake the fsaverage cortical surface into a compact mesh for the browser.

The web view used to draw a brain from sine waves. It looked like a rendered
blob, because that is the ceiling of approximating anatomy with equations. This
takes the real thing instead: fsaverage, a cortical surface reconstructed from
real MRI scans, which ships with MNE.

Three things come out of it, and the second two matter as much as the geometry:

  * the pial surface, so every gyrus and sulcus is where it actually is;
  * ``?h.sulc``, FreeSurfer's average convexity, which gives per-vertex sulcal
    depth — real shading in the folds rather than a guess from a noise field;
  * ``?h.aparc.annot``, the Desikan-Killiany parcellation, which names the
    precentral gyrus. For a motor-imagery decoder that is the one structure
    that has to be anatomically correct, and it is the difference between
    highlighting real primary motor cortex and highlighting where I guessed the
    central sulcus runs.

The full pial surface is ~650k triangles, far too heavy for a page. It is
decimated through an icosahedral source space, which subsamples vertices
*from the real surface* and retriangulates them, so the result stays real
anatomy rather than a smoothed approximation of it.

Run with ``python scripts/build_cortex.py``. Writes ``static/cortex.bin``.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

# Region ids. Order is load-bearing: it indexes the shader's uniform arrays,
# and must match REGION in static/brain3d.js.
FRONTAL = 0
MOTOR_LEFT = 1
MOTOR_RIGHT = 2
SOMATOSENSORY = 3
PARIETAL = 4
TEMPORAL = 5
OCCIPITAL = 6
CEREBELLUM = 7
BRAINSTEM = 8
CINGULATE = 9
INSULA = 10

#: Desikan-Killiany label -> display region. ``precentral`` and ``paracentral``
#: are split by hemisphere at build time, so they are marked here and resolved
#: below; paracentral is included because it carries the medial (foot) strip of
#: the motor homunculus.
_MOTOR = {"precentral", "paracentral"}

_ATLAS = {
    "postcentral": SOMATOSENSORY,
    "superiorfrontal": FRONTAL,
    "rostralmiddlefrontal": FRONTAL,
    "caudalmiddlefrontal": FRONTAL,
    "parsopercularis": FRONTAL,
    "parstriangularis": FRONTAL,
    "parsorbitalis": FRONTAL,
    "lateralorbitofrontal": FRONTAL,
    "medialorbitofrontal": FRONTAL,
    "frontalpole": FRONTAL,
    "superiorparietal": PARIETAL,
    "inferiorparietal": PARIETAL,
    "supramarginal": PARIETAL,
    "precuneus": PARIETAL,
    "superiortemporal": TEMPORAL,
    "middletemporal": TEMPORAL,
    "inferiortemporal": TEMPORAL,
    "bankssts": TEMPORAL,
    "fusiform": TEMPORAL,
    "transversetemporal": TEMPORAL,
    "temporalpole": TEMPORAL,
    "entorhinal": TEMPORAL,
    "parahippocampal": TEMPORAL,
    "lateraloccipital": OCCIPITAL,
    "lingual": OCCIPITAL,
    "cuneus": OCCIPITAL,
    "pericalcarine": OCCIPITAL,
    "rostralanteriorcingulate": CINGULATE,
    "caudalanteriorcingulate": CINGULATE,
    "posteriorcingulate": CINGULATE,
    "isthmuscingulate": CINGULATE,
    "corpuscallosum": CINGULATE,
    "unknown": CINGULATE,
    "insula": INSULA,
}

MAGIC = b"CTX3"
#: ico6 is 40,962 vertices per hemisphere. ico5 was a quarter of that and it
#: showed: decimation smooths sulci from tight, deep creases into rolling
#: hills, which is most of why the surface read as moulded rather than real.
SPACING = "ico6"
TARGET_RADIUS = 0.92      # cortex sits just inside the unit scalp sphere
#: Neighbours sampled per vertex when baking occlusion.
AO_NEIGHBOURS = 48
#: Falloff distance for that occlusion, in millimetres.
AO_RADIUS_MM = 9.0


def _annot_regions(subjects_dir: Path, hemi: str, vertno: np.ndarray) -> np.ndarray:
    """Per-vertex display region from the Desikan-Killiany parcellation."""
    from nibabel.freesurfer.io import read_annot

    path = subjects_dir / "fsaverage" / "label" / f"{hemi}.aparc.annot"
    labels, _ctab, names = read_annot(str(path))
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]

    motor = MOTOR_LEFT if hemi == "lh" else MOTOR_RIGHT
    lookup = np.full(len(names), FRONTAL, dtype=np.uint8)
    for i, name in enumerate(names):
        if name in _MOTOR:
            lookup[i] = motor
        else:
            lookup[i] = _ATLAS.get(name, FRONTAL)

    idx = labels[vertno]
    # read_annot marks unassigned vertices with -1.
    idx = np.where(idx < 0, names.index("unknown") if "unknown" in names else 0, idx)
    return lookup[idx]


def _sulcal_depth(subjects_dir: Path, hemi: str, vertno: np.ndarray) -> np.ndarray:
    """Signed sulcal convexity, scaled so ~1.0 is the floor of a deep sulcus.

    Deliberately *not* clamped. The viewer needs both halves of this: clamped
    to 0..1 it is shading depth, and left signed it is the scalar field whose
    contours the engraving hatches along. Level sets of convexity run around
    each gyrus, which is how an engraver lays strokes on a folded surface —
    hatching at fixed screen angles instead reads as flat texture.
    """
    from nibabel.freesurfer.io import read_morph_data

    path = subjects_dir / "fsaverage" / "surf" / f"{hemi}.sulc"
    sulc = read_morph_data(str(path))[vertno].astype(np.float32)
    # FreeSurfer convexity is positive in sulci. ~6 mm covers the usable range.
    return np.clip(sulc / 6.0, -2.0, 2.0)


def _vertex_normals(rr: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, needed before occlusion can be baked."""
    a, b, c = rr[tris[:, 0]], rr[tris[:, 1]], rr[tris[:, 2]]
    face = np.cross(b - a, c - a)          # length is twice the area: the weight
    out = np.zeros_like(rr)
    for col in range(3):
        np.add.at(out, tris[:, col], face)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


def _ambient_occlusion(rr: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Bake how enclosed each vertex is by the rest of the surface.

    Sulcal convexity says which way the surface bends; it does not say how much
    of the sky a point can actually see. Two walls of a deep sulcus facing each
    other are barely convex yet almost fully occluded, and shading from
    convexity alone is exactly why the folds looked soft. This measures the real
    thing: for each vertex, how much of its hemisphere neighbouring surface
    blocks, weighted by distance.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(rr)
    # k+1 because the first hit is always the vertex itself.
    _dist, idx = tree.query(rr, k=AO_NEIGHBOURS + 1, workers=-1)
    idx = idx[:, 1:]

    vecs = rr[idx] - rr[:, None, :]
    dist = np.linalg.norm(vecs, axis=2)
    dirs = vecs / np.maximum(dist, 1e-9)[..., None]

    # Only neighbours above the tangent plane can occlude.
    facing = np.clip(np.einsum("ijk,ik->ij", dirs, normals), 0.0, None)
    weight = facing / (1.0 + (dist / AO_RADIUS_MM) ** 2)
    ao = weight.sum(axis=1)

    # Normalise against the bulk rather than the extreme, so one pathological
    # vertex cannot flatten the whole range.
    ao = ao / max(np.percentile(ao, 99.0), 1e-9)
    return np.clip(ao, 0.0, 1.0).astype(np.float32)


def build() -> Path:
    import mne
    from mne.datasets import fetch_fsaverage

    fs = Path(fetch_fsaverage(verbose=False))
    subjects_dir = fs.parent

    print(f"fsaverage: {fs}")
    print(f"decimating pial surface via {SPACING} source space ...")
    src = mne.setup_source_space(
        "fsaverage",
        spacing=SPACING,
        surface="pial",
        subjects_dir=str(subjects_dir),
        add_dist=False,
        verbose=False,
    )

    positions: list[np.ndarray] = []
    regions: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    fines: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0

    for hemi, s in zip(("lh", "rh"), src, strict=True):
        vertno = s["vertno"]
        # rr is in metres; use_tris indexes the *full* surface, so remap it onto
        # the subsampled vertex list.
        rr = s["rr"][vertno] * 1000.0
        remap = np.full(s["np"], -1, dtype=np.int64)
        remap[vertno] = np.arange(len(vertno))
        tris = remap[s["use_tris"]]
        assert tris.min() >= 0, "decimated triangle referenced a dropped vertex"

        # Normals taken from the FULL pial surface, not from the decimated
        # triangulation. The decimated vertices are a subset of the real ones,
        # so their true normals are available -- and they carry fold detail
        # finer than this triangle count could ever represent. Computing
        # normals from the coarse mesh instead throws that detail away twice.
        full_rr, full_tris = mne.read_surface(
            str(fs / "surf" / f"{hemi}.pial"), verbose=False)
        fine = _vertex_normals(full_rr.astype(np.float64), full_tris)[vertno]

        positions.append(rr)
        regions.append(_annot_regions(subjects_dir, hemi, vertno))
        depths.append(_sulcal_depth(subjects_dir, hemi, vertno))
        fines.append(fine)
        faces.append(tris + offset)
        offset += len(vertno)
        print(f"  {hemi}: {len(vertno):,} of {len(full_rr):,} vertices, "
              f"{len(tris):,} triangles")

    rr = np.concatenate(positions).astype(np.float64)
    region = np.concatenate(regions).astype(np.uint8)
    depth = np.concatenate(depths).astype(np.float32)
    tris = np.concatenate(faces).astype(np.uint32)

    # FreeSurfer surface RAS is x=right, y=anterior, z=superior. The viewer uses
    # x=right, y=up, z=posterior, matching /api/v1/geometry.
    xyz = np.column_stack([rr[:, 0], rr[:, 2], -rr[:, 1]])
    xyz -= (xyz.max(axis=0) + xyz.min(axis=0)) / 2.0
    xyz *= TARGET_RADIUS / np.linalg.norm(xyz, axis=1).max()

    # Same rotation as the positions, and it is a proper rotation, so normals
    # go through it unchanged. The uniform scale does not affect them.
    fine = np.concatenate(fines)
    nrm = np.column_stack([fine[:, 0], fine[:, 2], -fine[:, 1]])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)

    print("baking ambient occlusion ...")
    ao = _ambient_occlusion(xyz, _vertex_normals(xyz, tris))
    print(f"  occlusion: mean {ao.mean():.3f}, "
          f"p95 {np.percentile(ao, 95):.3f}")

    out = Path(__file__).resolve().parents[1] / "static" / "cortex.bin"
    with out.open("wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<II", len(xyz), len(tris)))
        fh.write(xyz.astype("<f4").tobytes())
        fh.write(nrm.astype("<f4").tobytes())
        fh.write(depth.astype("<f4").tobytes())
        fh.write(ao.astype("<f4").tobytes())
        fh.write(region.tobytes())
        fh.write(b"\x00" * (-len(region) % 4))          # keep indices aligned
        fh.write(tris.astype("<u4").tobytes())

    counts = np.bincount(region, minlength=11)
    names = ["frontal", "motor-L", "motor-R", "somatosens", "parietal",
             "temporal", "occipital", "cerebellum", "brainstem", "cingulate",
             "insula"]
    print("\nvertices per region:")
    for name, c in zip(names, counts, strict=True):
        if c:
            print(f"  {name:<12} {c:>7,}")

    print(f"\nwrote {out}  ({out.stat().st_size / 1024:.0f} kB)")
    print(f"  {len(xyz):,} vertices, {len(tris):,} triangles")
    return out


if __name__ == "__main__":
    try:
        build()
    except ImportError as exc:  # pragma: no cover - developer tooling
        print(f"missing dependency: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
