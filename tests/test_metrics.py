"""Information transfer rate.

Accuracy on its own does not describe a BCI, and this is the figure that does.
What used to live beside it -- a character-tree speller and a
characters-per-minute estimate -- was removed with ``bwt.decoding``: it turned
decoded commands into letters, which is a selection interface rather than
decoding, and it implied a capability scalp EEG does not have.
"""

import pytest

from bwt.metrics import information_transfer_rate


class TestInformationTransferRate:
    def test_is_zero_at_chance(self):
        assert information_transfer_rate(0.5, 2, 3.0) == 0.0
        assert information_transfer_rate(0.25, 4, 3.0) == 0.0

    def test_is_zero_below_chance(self):
        assert information_transfer_rate(0.4, 2, 3.0) == 0.0

    def test_increases_with_accuracy(self):
        assert (information_transfer_rate(0.9, 2, 3.0)
                > information_transfer_rate(0.7, 2, 3.0))

    def test_at_perfect_accuracy_is_full_entropy(self):
        assert information_transfer_rate(1.0, 4, 60.0) == pytest.approx(2.0)

    def test_realistic_accuracy_gives_low_but_positive_rate(self):
        """A 63% two-class decoder is usable but slow -- state it numerically."""
        rate = information_transfer_rate(0.63, 2, 3.0)
        assert 0 < rate < 10

    def test_degenerate_arguments_are_zero_rather_than_an_error(self):
        assert information_transfer_rate(0.9, 1, 3.0) == 0.0
        assert information_transfer_rate(0.9, 2, 0.0) == 0.0

    def test_more_classes_carry_more_bits_at_equal_accuracy(self):
        """Four reliable commands say more per trial than two."""
        assert (information_transfer_rate(0.9, 4, 3.0)
                > information_transfer_rate(0.9, 2, 3.0))
