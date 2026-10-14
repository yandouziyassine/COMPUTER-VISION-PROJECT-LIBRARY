"""Project 06: plate-text helpers. evaluate.py builds a RapidOCR engine and imports Ultralytics at import time,
so only the pure helpers are loaded (see conftest.load_defs)."""

import pytest

np = pytest.importorskip("numpy")

from conftest import load_defs  # noqa: E402

ev = load_defs("06", "evaluate.py", ["clean", "TO_LETTER", "TO_DIGIT", "uk_syntax_fix", "char_accuracy"], {"np": np})


@pytest.mark.parametrize(
    "raw, expected",
    [("ab-12 c", "AB12C"), ("MY69 JKI", "MY69JKI"), ("é·!", ""), ("", "")],
)
def test_clean_keeps_ascii_alnum_uppercase(raw, expected):
    assert ev.clean(raw) == expected


@pytest.mark.parametrize(
    "read, expected",
    [
        ("MY69JKI", "MY69JKI"),  # already valid: unchanged
        ("MY6OJKI", "MY60JKI"),  # letter O in a digit slot -> 0
        ("0Y69JK1", "OY69JKI"),  # digits in letter slots -> letters
        ("8S5SAB2", "BS55ABZ"),
        ("MY69JK", "MY69JK"),  # not 7 characters: left alone
        ("MY69JKIX", "MY69JKIX"),
    ],
)
def test_uk_syntax_fix(read, expected):
    assert ev.uk_syntax_fix(read) == expected


def test_digit_letter_maps_are_inverse():
    for digit, letter in ev.TO_LETTER.items():
        assert ev.TO_DIGIT[letter] == digit


@pytest.mark.parametrize(
    "pred, truth, expected",
    [
        ("MY69JKI", "MY69JKI", 1.0),
        ("MY69JKL", "MY69JKI", 1 - 1 / 7),  # one substitution
        ("MY69JK", "MY69JKI", 1 - 1 / 7),  # one deletion
        ("XMY69JKI", "MY69JKI", 1 - 1 / 7),  # one insertion
        ("", "MY69JKI", 0.0),
        ("ABCDEFGHIJ", "XY", 0.0),  # clipped at 0
    ],
)
def test_char_accuracy(pred, truth, expected):
    assert ev.char_accuracy(pred, truth) == pytest.approx(expected)


def test_char_accuracy_empty_truth_does_not_divide_by_zero():
    assert ev.char_accuracy("", "") == 1.0
