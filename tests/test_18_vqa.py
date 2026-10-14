"""Project 18: VQA answer normalization. evaluate.py imports the vision-language model wrapper (Transformers)
at import time, so only normalize() is loaded (see conftest.load_defs)."""

import re

import pytest
from conftest import load_defs

normalize = load_defs("18", "evaluate.py", ["NUMBERS", "normalize"], {"re": re}).normalize


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("Yes", "yes"),
        ("Yes.", "yes"),
        ("Two", "2"),
        ("two dogs", "2 dogs"),
        ("None", "0"),
        ("The cat", "cat"),
        ("an apple", "apple"),
        ("A red bus!", "red bus"),
        ("  Tennis racket  ", "tennis racket"),
        ("3", "3"),
        ("", ""),
    ],
)
def test_normalize(answer, expected):
    assert normalize(answer) == expected


def test_human_and_model_answers_match_after_normalization():
    assert normalize("Two.") == normalize("2") == normalize("two")
