"""Project 03: receipt field extraction rules (pure text, no OCR).

receipt.py imports RapidOCR/OpenVINO at import time, so only the text rules are loaded (see conftest.load_defs)."""

import re

import pytest
from conftest import load_defs

r = load_defs(
    "03",
    "receipt.py",
    [
        "DATE_PATTERNS",
        "MONTHS",
        "AMOUNT",
        "STRONG_TOTAL",
        "NOT_TOTAL",
        "normalize_date",
        "to_float",
        "find_amounts",
        "normalize_amount",
        "total_priority",
        "extract_fields",
    ],
    {"re": re},
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("12.50", "12.50"),
        ("RM 12.50", "12.50"),
        ("1,234.00", "1234.00"),  # comma thousands, dot decimal
        ("1.234,50", "1234.50"),  # dot thousands, comma decimal
        ("10,000.00", "10000.00"),
        ("1,234,567.89", "1234567.89"),
        ("TOTAL 7,50", "7.50"),
        ("no amount here", None),
        ("12.5", None),  # an amount needs 2 decimals
    ],
)
def test_normalize_amount(text, expected):
    assert r.normalize_amount(text) == expected


def test_normalize_amount_first_vs_last():
    line = "Qty 2 @ 3.00 = 6.00"
    assert r.normalize_amount(line) == "3.00"
    assert r.normalize_amount(line, last=True) == "6.00"


def test_normalize_amount_two_amounts_separated_by_space():
    # Was a real bug (spaces stripped before matching gave 3006.00); fixed in find_amounts().
    assert r.normalize_amount("3.00 6.00", last=True) == "6.00"


def test_normalize_amount_ignores_ocr_spaces():
    assert r.normalize_amount("1 234.00") == "1234.00"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("15/01/2019", "15/01/2019"),
        ("15/01/201911:05:16AM", "15/01/2019"),  # OCR glued the time to the year
        ("Date: 5-3-2018", "05/03/2018"),
        ("25/12/18", "25/12/2018"),
        ("2018-12-25", "25/12/2018"),
        ("25 Dec 2018", "25/12/2018"),
        ("3 January 2019", "03/01/2019"),
        ("45/13/2018", None),  # impossible day/month
        ("Tel 03-7865 1234", None),
        ("nothing", None),
    ],
)
def test_normalize_date(text, expected):
    assert r.normalize_date(text) == expected


@pytest.mark.parametrize(
    "line, expected",
    [
        ("TOTAL 12.00", 1),
        ("Total Incl. GST 12.72", 2),
        ("Grand Total 12.00", 3),
        ("Total Amount Payable 12.00", 3),
        ("Total Rounded 12.00", 3),
        ("SUBTOTAL 10.00", 0),
        ("Total GST 0.72", 0),
        ("Total Qty 3", 0),
        ("CASH 20.00", 0),
        ("Thank you", 0),
    ],
)
def test_total_priority(line, expected):
    assert r.total_priority(line) == expected


def test_extract_fields_amount_on_next_line():
    lines = [
        "ACME SDN BHD",
        "Date: 15/01/2019 10:02",
        "Subtotal 10.00",
        "Total GST 0.60",
        "TOTAL",
        "10.60",
        "CASH 20.00",
        "CHANGE 9.40",
    ]
    f = r.extract_fields(lines)
    assert f == {"company": "ACME SDN BHD", "date": "15/01/2019", "total": "10.60", "n_lines": len(lines)}


def test_extract_fields_stronger_total_wins():
    lines = ["SHOP", "Total 10.62", "Total Rounded 10.60"]
    assert r.extract_fields(lines)["total"] == "10.60"


def test_extract_fields_first_of_equal_priority_wins():
    lines = ["SHOP", "Total 5.00", "Total 7.00"]
    assert r.extract_fields(lines)["total"] == "5.00"


def test_extract_fields_falls_back_to_largest_amount():
    lines = ["SHOP", "Bread 2.50", "Milk 1,204.10", "Eggs 3.00"]
    f = r.extract_fields(lines)
    assert f["total"] == "1204.10"
    assert f["date"] is None


def test_extract_fields_empty():
    assert r.extract_fields([]) == {"company": None, "date": None, "total": None, "n_lines": 0}
