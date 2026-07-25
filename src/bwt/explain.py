"""Showing that the model uses physiology rather than artifacts.

A motor-imagery decoder that scores above chance has not necessarily learned
anything about motor cortex. It could be reading eye movement, jaw tension, or
a timing quirk that happens to correlate with the cue. The way to tell is to
look at *where on the head* and *in which frequency band* the discriminative
information sits, and check it against what neurophysiology predicts:

* Imagining left-hand movement desynchronises the mu (8-12 Hz) and beta
  (13-30 Hz) rhythms over the **right** sensorimotor cortex, and vice versa --
  the effect is contralateral.
* The relevant electrodes are C3, C4 and Cz and their immediate neighbours.
* The effect is a power *decrease* during imagery relative to rest, known as
  event-related desynchronisation (ERD).

:func:`csp_patterns` renders the learned spatial filters as scalp maps, and
:func:`erd_curve` measures the band-power time course directly from the data.
If the topographies are focused over the hand area and the ERD is contralateral,
the decoder is doing what it claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from bwt.data.epochs import EpochBundle
from bwt.logging_utils import get_logger

log = get_logger(__name__)

#: Electrodes over the hand area of sensorimotor cortex.
MOTOR_CHANNELS = ("C3", "Cz", "C4")


def _montage_info(ch_names: Sequence[str], sfreq: float):
    """Build an MNE info object with 10-05 sensor positions attached."""
    import mne

    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(montage, on_missing="ignore", verbose="ERROR")
    return info


def _find_csp(model):
    """Locate a fitted CSP inside a pipeline, if there is one."""
    steps = getattr(model, "named_steps", {})
    for step in steps.values():
        if hasattr(step, "patterns_") and hasattr(step, "filters_"):
            return step
    return None


def csp_patterns(
    model,
    card,
    *,
    n_components: int = 4,
    output: Path | None = None,
    show: bool = False,
):
    """Plot CSP spatial patterns as scalp topographies.

    Patterns (not filters) are plotted, because the pattern is what can be read
    as "where the signal comes from"; a filter's weights are shaped by noise
    suppression and are not directly interpretable as sources. See Haufe et al.
    (2014), "On the interpretation of weight vectors of linear models in
    neuroimaging".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csp = _find_csp(model)
    if csp is None:
        raise ValueError(
            "this pipeline has no CSP step; spatial patterns are only "
            "available for csp_lda / fbcsp_lda"
        )

    info = _montage_info(card.ch_names, card.sfreq)
    patterns = np.asarray(csp.patterns_)
    n_components = min(n_components, patterns.shape[0])

    fig, axes = plt.subplots(1, n_components, figsize=(3.1 * n_components, 3.4))
    axes = np.atleast_1d(axes)
    for index in range(n_components):
        import mne

        mne.viz.plot_topomap(
            patterns[index], info, axes=axes[index], show=False,
            contours=4, cmap="RdBu_r",
        )
        axes[index].set_title(f"CSP {index + 1}", fontsize=11)
    fig.suptitle(
        f"CSP spatial patterns - {card.task} ({' vs '.join(card.classes)})",
        fontsize=12,
    )
    fig.tight_layout()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=140, bbox_inches="tight")
        log.info("wrote %s", output)
    if show:  # pragma: no cover
        plt.show()
    else:
        plt.close(fig)
    return output


#: Epoch window that includes a pre-cue baseline. ERD is defined as a change
#: *relative to rest*, so measuring it requires data from before the cue --
#: the decoding window (0.5-3.5 s) does not contain any.
ERD_WINDOW = (-1.5, 4.0)


def band_power_timecourse(
    bundle: EpochBundle,
    channel: str,
    band: tuple[float, float] = (8.0, 30.0),
    *,
    window_seconds: float = 0.25,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Band power over time for one channel, averaged within each class.

    Returns ``(times, {class_name: power})`` with power as a percentage change
    from the pre-cue baseline, the conventional ERD/ERS units.

    The baseline is the mean over all windows before ``t=0``. If the bundle does
    not extend before the cue -- the decoding window starts at 0.5 s -- there is
    no rest period to compare against, so the first window is used instead and a
    warning is emitted. Numbers computed that way are *not* ERD in the usual
    sense and should not be read as such; load the bundle with
    ``tmin=ERD_WINDOW[0]`` to get the real thing.
    """
    from scipy.signal import butter, sosfiltfilt

    try:
        index = [c.upper() for c in bundle.ch_names].index(channel.upper())
    except ValueError:
        raise ValueError(
            f"channel {channel!r} not in this recording; have "
            f"{list(bundle.ch_names)[:8]}..."
        ) from None

    nyquist = bundle.sfreq / 2
    sos = butter(4, [band[0] / nyquist, band[1] / nyquist],
                 btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, bundle.X[:, index, :].astype(np.float64), axis=-1)
    squared = filtered ** 2

    width = max(1, int(round(window_seconds * bundle.sfreq)))
    n_windows = squared.shape[1] // width
    trimmed = squared[:, : n_windows * width]
    binned = trimmed.reshape(len(trimmed), n_windows, width).mean(axis=2)

    times = bundle.tmin + (np.arange(n_windows) + 0.5) * window_seconds
    pre_cue = times < 0.0
    if not pre_cue.any():
        log.warning(
            "bundle starts at %.2fs so there is no pre-cue baseline; "
            "normalising to the first window instead, which understates ERD. "
            "Re-load with tmin=%.1f for a valid measurement.",
            bundle.tmin, ERD_WINDOW[0],
        )

    out: dict[str, np.ndarray] = {}
    for label, name in enumerate(bundle.classes):
        rows = binned[bundle.y == label]
        if not len(rows):
            continue
        mean = rows.mean(axis=0)
        baseline = float(mean[pre_cue].mean()) if pre_cue.any() else float(mean[0])
        if baseline <= 0:
            baseline = 1.0
        out[name] = 100.0 * (mean - baseline) / baseline
    return times, out


def erd_curve(
    bundle: EpochBundle,
    *,
    channels: Sequence[str] = MOTOR_CHANNELS,
    band: tuple[float, float] = (8.0, 30.0),
    output: Path | None = None,
    show: bool = False,
):
    """Plot mu/beta power over time at the motor electrodes, per class.

    For a left-vs-right hand task the expected signature is a crossover: C3
    (left hemisphere) shows the larger decrease for *right*-hand imagery and C4
    the larger decrease for *left*-hand imagery.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = [c for c in channels
                if c.upper() in {n.upper() for n in bundle.ch_names}]
    if not channels:
        raise ValueError("none of the requested channels are in this recording")

    fig, axes = plt.subplots(1, len(channels), figsize=(4.4 * len(channels), 3.6),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for axis, channel in zip(axes, channels):
        times, curves = band_power_timecourse(bundle, channel, band)
        for name, values in curves.items():
            axis.plot(times, values, label=name, linewidth=1.9)
        axis.axhline(0, color="0.6", linewidth=0.8, linestyle="--")
        axis.set_title(channel, fontsize=11)
        axis.set_xlabel("time from cue (s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(f"{band[0]:g}-{band[1]:g} Hz power change (%)")
    axes[-1].legend(fontsize=9)
    fig.suptitle(
        f"Event-related desynchronisation - {bundle.task}", fontsize=12
    )
    fig.tight_layout()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=140, bbox_inches="tight")
        log.info("wrote %s", output)
    if show:  # pragma: no cover
        plt.show()
    else:
        plt.close(fig)
    return output


def lateralisation_index(
    bundle: EpochBundle,
    band: tuple[float, float] = (8.0, 30.0),
) -> dict:
    """Quantify contralateral dominance without plotting anything.

    For each class, compares late-trial band power at C3 against C4. A
    left-vs-right hand task should produce indices of opposite sign for the two
    classes; a decoder scoring above chance on a task that does *not* show this
    is probably exploiting something other than sensorimotor rhythm.
    """
    available = {c.upper() for c in bundle.ch_names}
    if not {"C3", "C4"} <= available:
        raise ValueError("this recording lacks C3/C4")

    _, left = band_power_timecourse(bundle, "C3", band)
    _, right = band_power_timecourse(bundle, "C4", band)

    times, _ = band_power_timecourse(bundle, "C3", band)
    during = times > 0.5  # after the cue-evoked response has passed

    out: dict[str, float] = {}
    for name in bundle.classes:
        if name not in left or name not in right:
            continue
        c3 = float(np.mean(left[name][during]))
        c4 = float(np.mean(right[name][during]))
        out[name] = c3 - c4

    valid_baseline = bool((times < 0).any())
    return {
        "band": list(band),
        "has_pre_cue_baseline": valid_baseline,
        "index_per_class": out,
        "interpretation": (
            "C3 minus C4 power change during imagery, in percentage points, "
            "relative to the pre-cue baseline. Sensorimotor rhythm "
            "desynchronisation is contralateral, so right-hand imagery should "
            "suppress C3 more than C4 (negative index) and left-hand imagery "
            "the reverse (positive index)."
            + ("" if valid_baseline else
               " WARNING: this bundle has no pre-cue samples, so these values "
               "are not true ERD and should not be interpreted as such.")
        ),
    }


__all__ = [
    "MOTOR_CHANNELS",
    "band_power_timecourse",
    "csp_patterns",
    "erd_curve",
    "lateralisation_index",
]
