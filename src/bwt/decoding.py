"""Turning decoded mental commands into text.

**What this is, precisely.** The classifier upstream distinguishes a small set
of imagined movements. It does not read language, inner speech, or intended
words -- no scalp-EEG system does, and any claim otherwise is false. What a
motor-imagery BCI can do is emit a few *discrete commands per trial*, and text
is produced by using those commands to navigate a selection interface. That is
how real assistive spellers work, and it is what this module implements.

Two spellers are provided:

:class:`BinaryTreeSpeller`
    For a two-class decoder. Characters sit at the leaves of a binary tree; each
    decoded trial takes one branch. A 32-symbol alphabet needs 5 trials per
    character.
:class:`GridSpeller`
    For a four-class decoder. A 4-ary tree over the same alphabet needs only
    ceil(log4 32) = 3 trials per character.

The honest performance figure for a speller is not classifier accuracy but
**information transfer rate** (:func:`information_transfer_rate`), the standard
BCI metric, which accounts for how a modest per-trial accuracy compounds over
the several trials needed to commit one character.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

#: Default symbol set: the alphabet, space, and two editing commands. 32 slots
#: makes the binary tree exactly 5 levels deep with no wasted branches.
DEFAULT_ALPHABET: tuple[str, ...] = tuple(
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["_", ".", ",", "?", "<BS>", "<CLR>"]
)

BACKSPACE = "<BS>"
CLEAR = "<CLR>"


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


def expected_characters_per_minute(
    accuracy: float, n_classes: int, seconds_per_trial: float, n_symbols: int
) -> float:
    """Characters per minute, accounting for errors compounding across trials.

    A character needs ``d = ceil(log_n(symbols))`` correct decisions in a row,
    so the probability of selecting the intended character is ``accuracy ** d``.
    Expected trials per *successful* character is therefore ``d / accuracy**d``.
    """
    if accuracy <= 0 or n_classes < 2:
        return 0.0
    depth = max(1, math.ceil(math.log(n_symbols, n_classes)))
    success = accuracy ** depth
    if success <= 0:
        return 0.0
    seconds = depth * seconds_per_trial / success
    return float(60.0 / seconds)


@dataclass
class SpellerStep:
    """One decoded trial and what the speller did with it."""

    trial_index: int
    command: str
    confidence: float
    remaining: list[str] = field(default_factory=list)
    emitted: str | None = None

    @property
    def n_remaining(self) -> int:
        return len(self.remaining)


@dataclass
class SpellerResult:
    text: str
    steps: list[SpellerStep]
    partial: list[str] = field(default_factory=list)
    trials_per_character: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "trials_per_character": self.trials_per_character,
            "n_trials_consumed": len(self.steps),
            "partial_candidates": self.partial,
            "steps": [
                {
                    "trial": s.trial_index,
                    "command": s.command,
                    "confidence": round(s.confidence, 4),
                    "candidates_remaining": s.n_remaining,
                    "emitted": s.emitted,
                }
                for s in self.steps
            ],
        }


class _TreeSpeller:
    """Shared n-ary tree navigation over an alphabet."""

    def __init__(self, commands: Sequence[str],
                 alphabet: Sequence[str] = DEFAULT_ALPHABET):
        if len(commands) < 2:
            raise ValueError("a speller needs at least two distinct commands")
        if len(alphabet) < 2:
            raise ValueError("alphabet must have at least two symbols")
        self.commands = tuple(commands)
        self.alphabet = tuple(alphabet)
        self.branching = len(self.commands)
        self.depth = max(1, math.ceil(
            math.log(len(self.alphabet), self.branching)))

    def _partition(self, candidates: Sequence[str]) -> list[list[str]]:
        """Split candidates into ``branching`` contiguous, near-equal groups."""
        n = len(candidates)
        size = math.ceil(n / self.branching)
        groups = [list(candidates[i:i + size]) for i in range(0, n, size)]
        while len(groups) < self.branching:
            groups.append([])
        return groups

    def decode(
        self,
        commands: Sequence[str],
        confidences: Sequence[float] | None = None,
    ) -> SpellerResult:
        """Consume a sequence of decoded commands and emit the resulting text."""
        if confidences is None:
            confidences = [float("nan")] * len(commands)
        if len(confidences) != len(commands):
            raise ValueError("commands and confidences must be the same length")

        text_parts: list[str] = []
        steps: list[SpellerStep] = []
        candidates = list(self.alphabet)

        for index, (command, confidence) in enumerate(zip(commands, confidences)):
            if command not in self.commands:
                raise ValueError(
                    f"command {command!r} is not one of {self.commands}"
                )
            groups = self._partition(candidates)
            candidates = groups[self.commands.index(command)]

            emitted = None
            if len(candidates) == 1:
                symbol = candidates[0]
                if symbol == BACKSPACE:
                    if text_parts:
                        text_parts.pop()
                    emitted = BACKSPACE
                elif symbol == CLEAR:
                    text_parts.clear()
                    emitted = CLEAR
                else:
                    text_parts.append(" " if symbol == "_" else symbol)
                    emitted = symbol
                candidates = list(self.alphabet)
            elif not candidates:
                # Empty branch: nothing selectable, restart the character.
                candidates = list(self.alphabet)

            steps.append(
                SpellerStep(
                    trial_index=index,
                    command=command,
                    confidence=float(confidence),
                    remaining=list(candidates),
                    emitted=emitted,
                )
            )

        return SpellerResult(
            text="".join(text_parts),
            steps=steps,
            partial=[] if len(candidates) == len(self.alphabet) else list(candidates),
            trials_per_character=self.depth,
        )

    def decode_from_predictions(
        self, class_names: Sequence[str], probabilities: np.ndarray | None = None
    ) -> SpellerResult:
        """Decode from classifier output, using max class probability as confidence."""
        if probabilities is None:
            return self.decode(class_names)
        confidences = np.max(np.asarray(probabilities), axis=1)
        return self.decode(class_names, confidences)


class BinaryTreeSpeller(_TreeSpeller):
    """Two-command speller: each trial halves the candidate set."""

    def __init__(self, commands: Sequence[str] = ("left_fist", "right_fist"),
                 alphabet: Sequence[str] = DEFAULT_ALPHABET):
        if len(commands) != 2:
            raise ValueError("BinaryTreeSpeller requires exactly two commands")
        super().__init__(commands, alphabet)


class GridSpeller(_TreeSpeller):
    """Four-command speller: each trial narrows the candidate set to a quarter."""

    def __init__(
        self,
        commands: Sequence[str] = (
            "left_fist", "right_fist", "both_fists", "both_feet",
        ),
        alphabet: Sequence[str] = DEFAULT_ALPHABET,
    ):
        if len(commands) != 4:
            raise ValueError("GridSpeller requires exactly four commands")
        super().__init__(commands, alphabet)


def make_speller(class_names: Sequence[str],
                 alphabet: Sequence[str] = DEFAULT_ALPHABET) -> _TreeSpeller:
    """Pick the speller that matches a model's class set."""
    names = list(class_names)
    if len(names) == 2:
        return BinaryTreeSpeller(names, alphabet)
    if len(names) == 4:
        return GridSpeller(names, alphabet)
    return _TreeSpeller(names, alphabet)


__all__ = [
    "BACKSPACE",
    "CLEAR",
    "DEFAULT_ALPHABET",
    "BinaryTreeSpeller",
    "GridSpeller",
    "SpellerResult",
    "SpellerStep",
    "expected_characters_per_minute",
    "information_transfer_rate",
    "make_speller",
]
