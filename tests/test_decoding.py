"""Tests for the mental-command speller and BCI throughput metrics."""

from __future__ import annotations

import math

import pytest

from bwt.decoding import (
    BACKSPACE,
    DEFAULT_ALPHABET,
    BinaryTreeSpeller,
    GridSpeller,
    expected_characters_per_minute,
    information_transfer_rate,
    make_speller,
)


class TestBinaryTreeSpeller:
    def test_depth_matches_alphabet_size(self):
        speller = BinaryTreeSpeller(("left_fist", "right_fist"))
        assert speller.depth == math.ceil(math.log2(len(DEFAULT_ALPHABET)))

    def test_all_left_selects_first_symbol(self):
        speller = BinaryTreeSpeller(("L", "R"), alphabet=tuple("ABCD"))
        result = speller.decode(["L", "L"])
        assert result.text == "A"

    def test_all_right_selects_last_symbol(self):
        speller = BinaryTreeSpeller(("L", "R"), alphabet=tuple("ABCD"))
        assert speller.decode(["R", "R"]).text == "D"

    def test_every_symbol_is_reachable(self):
        alphabet = tuple("ABCDEFGH")
        speller = BinaryTreeSpeller(("L", "R"), alphabet=alphabet)
        reached = set()
        for index in range(len(alphabet)):
            commands = [
                "R" if (index >> (2 - bit)) & 1 else "L" for bit in range(3)
            ]
            reached.add(speller.decode(commands).text)
        assert reached == set(alphabet)

    def test_selection_resets_after_a_character(self):
        speller = BinaryTreeSpeller(("L", "R"), alphabet=tuple("ABCD"))
        assert speller.decode(["L", "L", "R", "R"]).text == "AD"

    def test_partial_selection_reports_remaining_candidates(self):
        speller = BinaryTreeSpeller(("L", "R"), alphabet=tuple("ABCD"))
        result = speller.decode(["L"])
        assert result.text == ""
        assert set(result.partial) == {"A", "B"}

    def test_underscore_becomes_a_space(self):
        speller = BinaryTreeSpeller(("L", "R"), alphabet=("A", "_"))
        assert speller.decode(["R"]).text == " "

    def test_backspace_deletes_previous_character(self):
        speller = BinaryTreeSpeller(("L", "R"), alphabet=("A", BACKSPACE))
        assert speller.decode(["L", "L", "R"]).text == "A"

    def test_rejects_unknown_command(self):
        speller = BinaryTreeSpeller(("L", "R"), alphabet=tuple("AB"))
        with pytest.raises(ValueError, match="not one of"):
            speller.decode(["X"])

    def test_rejects_mismatched_confidences(self):
        speller = BinaryTreeSpeller(("L", "R"), alphabet=tuple("AB"))
        with pytest.raises(ValueError, match="same length"):
            speller.decode(["L", "R"], [0.9])

    def test_requires_two_commands(self):
        with pytest.raises(ValueError, match="exactly two"):
            BinaryTreeSpeller(("only_one",))


class TestGridSpeller:
    def test_four_commands_need_fewer_trials(self):
        binary = BinaryTreeSpeller(("L", "R"))
        grid = GridSpeller(("a", "b", "c", "d"))
        assert grid.depth < binary.depth

    def test_selects_a_character(self):
        speller = GridSpeller(("a", "b", "c", "d"), alphabet=tuple("ABCD"))
        assert speller.decode(["a"]).text == "A"
        assert speller.decode(["d"]).text == "D"

    def test_requires_four_commands(self):
        with pytest.raises(ValueError, match="exactly four"):
            GridSpeller(("a", "b"))


class TestMakeSpeller:
    def test_picks_binary_for_two_classes(self):
        assert isinstance(make_speller(["left_fist", "right_fist"]), BinaryTreeSpeller)

    def test_picks_grid_for_four_classes(self):
        assert isinstance(
            make_speller(["a", "b", "c", "d"]), GridSpeller
        )

    def test_handles_three_classes(self):
        speller = make_speller(["a", "b", "c"])
        assert speller.branching == 3


class TestThroughputMetrics:
    def test_itr_is_zero_at_chance(self):
        assert information_transfer_rate(0.5, 2, 3.0) == 0.0
        assert information_transfer_rate(0.25, 4, 3.0) == 0.0

    def test_itr_is_zero_below_chance(self):
        assert information_transfer_rate(0.4, 2, 3.0) == 0.0

    def test_itr_increases_with_accuracy(self):
        assert (information_transfer_rate(0.9, 2, 3.0)
                > information_transfer_rate(0.7, 2, 3.0))

    def test_itr_at_perfect_accuracy_is_full_entropy(self):
        assert information_transfer_rate(1.0, 4, 60.0) == pytest.approx(2.0)

    def test_realistic_accuracy_gives_low_but_positive_itr(self):
        """A 63% two-class decoder is usable but slow -- state it numerically."""
        rate = information_transfer_rate(0.63, 2, 3.0)
        assert 0 < rate < 10

    def test_characters_per_minute_accounts_for_compounding_errors(self):
        good = expected_characters_per_minute(0.95, 2, 3.0, 32)
        poor = expected_characters_per_minute(0.63, 2, 3.0, 32)
        assert good > poor > 0

    def test_characters_per_minute_is_zero_at_zero_accuracy(self):
        assert expected_characters_per_minute(0.0, 2, 3.0, 32) == 0.0
