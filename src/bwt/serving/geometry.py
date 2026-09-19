"""Electrode geometry for the 3D visualiser.

The positions come from the same 10-05 montage the decoder uses, so what the
browser draws is where the electrodes actually sat on the head. Nothing here is
invented for looks: if the scene shows the left sensorimotor strip lighting up,
that is genuinely C3 and its neighbours.

Served live from the loaded model's channel list rather than baked into a static
file, so the scene cannot drift out of step with the model it is visualising.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from bwt.logging_utils import get_logger

log = get_logger(__name__)

#: Electrodes over the hand area of sensorimotor cortex, split by hemisphere.
#: Motor imagery desynchronises these contralaterally, so the visualiser can
#: highlight the side that should respond to each decoded class.
LEFT_MOTOR = ("FC5", "FC3", "FC1", "C5", "C3", "C1", "CP5", "CP3", "CP1")
RIGHT_MOTOR = ("FC2", "FC4", "FC6", "C2", "C4", "C6", "CP2", "CP4", "CP6")
MIDLINE_MOTOR = ("FCz", "Cz", "CPz")


@lru_cache(maxsize=8)
def _montage_positions(montage_name: str = "standard_1005") -> dict[str, tuple]:
    import mne

    montage = mne.channels.make_standard_montage(montage_name)
    positions = montage.get_positions()["ch_pos"]
    return {
        name.upper(): tuple(float(v) for v in coords)
        for name, coords in positions.items()
        if coords is not None and not np.isnan(coords).any()
    }


def electrode_geometry(
    ch_names: Sequence[str],
    *,
    montage_name: str = "standard_1005",
) -> dict:
    """Return browser-ready 3D coordinates for the given channels.

    The montage frame is x=right, y=front, z=up in metres. WebGL convention is
    y=up and z toward the viewer, so the axes are remapped and the whole cloud is
    recentred on its own centroid and scaled to unit radius — the scene cares
    about shape, not absolute head size.
    """
    lookup = _montage_positions(montage_name)
    resolved: list[tuple[str, tuple]] = []
    unplaced: list[str] = []

    for name in ch_names:
        coords = lookup.get(name.upper())
        if coords is None:
            unplaced.append(name)
            continue
        resolved.append((name, coords))

    if not resolved:
        raise ValueError(
            f"none of the {len(ch_names)} channels are in the {montage_name} "
            "montage; cannot build geometry"
        )
    if unplaced:
        log.warning("%d channel(s) have no montage position: %s",
                    len(unplaced), unplaced[:6])

    raw = np.array([c for _, c in resolved], dtype=float)
    centred = raw - raw.mean(axis=0)
    scale = float(np.abs(centred).max()) or 1.0
    centred /= scale

    # Upper-case both sides: midline labels carry a lower-case 'z' (Cz, FCz,
    # CPz), so comparing them against name.upper() silently never matches.
    left = {n.upper() for n in LEFT_MOTOR}
    right = {n.upper() for n in RIGHT_MOTOR}
    mid = {n.upper() for n in MIDLINE_MOTOR}
    electrodes = []
    for (name, _), point in zip(resolved, centred, strict=True):
        upper = name.upper()
        region = (
            "left_motor" if upper in left
            else "right_motor" if upper in right
            else "midline_motor" if upper in mid
            else "other"
        )
        electrodes.append({
            "name": name,
            # x = right, y = up, z = back
            "x": round(float(point[0]), 4),
            "y": round(float(point[2]), 4),
            "z": round(float(-point[1]), 4),
            "region": region,
        })

    return {
        "montage": montage_name,
        "n_channels": len(electrodes),
        "unplaced": unplaced,
        "frame": "x=right, y=up, z=back; centred and scaled to unit radius",
        "electrodes": electrodes,
        "regions": {
            "left_motor": sorted(left),
            "right_motor": sorted(right),
            "midline_motor": sorted(mid),
        },
    }


__all__ = [
    "LEFT_MOTOR",
    "MIDLINE_MOTOR",
    "RIGHT_MOTOR",
    "electrode_geometry",
]
