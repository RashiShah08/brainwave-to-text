"""Performance metrics for a motor-imagery decoder.

Accuracy alone does not describe a BCI: a decoder that is right 70% of the time
on two classes and one that is right 40% of the time on four are not comparable
by accuracy, and neither number says how much can actually be communicated per
minute. Information transfer rate is the standard figure that does.

This module replaced ``bwt.decoding``, which additionally turned decoded
commands into text by navigating a character tree. That was removed: it was a
selection interface bolted on top of the classifier, and putting letters on
screen next to a brain invited exactly the reading it was written to deny.
Scalp EEG carries no language, and nothing here claims otherwise.
"""

from __future__ import annotations

import math


def information_transfer_rate(
    accuracy: float, n_classes: int, seconds_per_trial: float
) -> float:
    """Wolpaw information transfer rate, in bits per minute.

    ``accuracy`` is per-trial decoding accuracy, ``n_classes`` the number of
    distinguishable commands. Returns 0 when accuracy is at or below chance,
    where the formula is not meaningful.
    """
    if n_classes < 2 or seconds_per_trial <= 0:
        return 0.0
    accuracy = min(max(accuracy, 0.0), 1.0)
    if accuracy <= 1.0 / n_classes:
        return 0.0
    if accuracy >= 1.0:
        bits = math.log2(n_classes)
    else:
        bits = (
            math.log2(n_classes)
            + accuracy * math.log2(accuracy)
            + (1 - accuracy) * math.log2((1 - accuracy) / (n_classes - 1))
        )
    return float(bits * 60.0 / seconds_per_trial)


__all__ = ["information_transfer_rate"]
