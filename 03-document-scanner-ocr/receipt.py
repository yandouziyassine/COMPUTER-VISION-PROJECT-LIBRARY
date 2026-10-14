"""OCR a receipt and pull out structured fields (company, date, total) with simple rules.

RapidOCR (PaddleOCR detection + recognition models) runs on CPU through OpenVINO, which is
~11x faster than its ONNX Runtime build on this laptop (0.48 s vs 5.2 s per receipt).
The field extraction is deliberately rule-based (regexes + layout heuristics): a transparent baseline.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common.ov_compat  # noqa: E402,F401  (must come before rapidocr_openvino; see that file)
from rapidocr_openvino import RapidOCR  # noqa: E402

_engine = None

DATE_PATTERNS = [
    # No trailing \b on the year: OCR often glues the time on ("15/01/201911:05:16AM").
    r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})",  # 25/12/2018
    r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})(?!\d)",  # 25/12/18
    r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b",  # 2018-12-25
    r"\b(\d{1,2})\s*([A-Za-z]{3})[A-Za-z]*\s*,?\s*(\d{2,4})\b",  # 25 Dec 2018
]
MONTHS = {
    m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)
}
# An amount ends in 2 decimals; it may have thousands separators ("1,234.00", "1.234,00", "1 234.00").
# A space only counts as a separator before exactly 3 digits, so "3.00 6.00" stays two amounts.
AMOUNT = re.compile(r"(?<![\d.,])(\d{1,3}(?:[., ]\d{3})+[.,]\d{2}|\d+[.,]\d{2})(?!\d)")
# Matched against the lowercased line with spaces removed (OCR often drops them: "TotalAmtIncl.GST@6%").
STRONG_TOTAL = ("payable", "amountdue", "grandtotal", "nettotal", "netttotal", "totalrounded", "roundedtotal")
NOT_TOTAL = ("subtotal", "excl", "qty", "item", "discount", "gstsummary", "tender", "change", "cash")


def engine():
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


def ocr_lines(image_bgr):
    """Run OCR and merge words into text lines, top to bottom."""
    result, _ = engine()(image_bgr, use_cls=False)  # receipts are upright: skip the 180° classifier
    if not result:
        return []
    boxes, slopes = [], []
    for box, text, _conf in result:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        (x0, y0), (x1, y1) = box[0], box[1]  # top-left → top-right edge of the text box
        if x1 - x0 > 3 * (max(ys) - min(ys)):  # only long boxes give a reliable slope
            slopes.append((y1 - y0) / (x1 - x0))
        boxes.append({"y": sum(ys) / 4, "xc": sum(xs) / 4, "h": max(ys) - min(ys), "x": min(xs), "text": text})
    # Deskew: a slightly rotated receipt puts right-column prices on the "next line".
    # Measure the dominant text slope and compare boxes along it instead of along pure horizontal.
    slope = sorted(slopes)[len(slopes) // 2] if slopes else 0.0
    for b in boxes:
        b["y"] -= slope * b["xc"]
    boxes.sort(key=lambda b: b["y"])
    # Same line = vertical centers closer than half a *typical* text height. Using each box's own
    # height lets one tall box (merged detection) chain several real lines together.
    tol = 0.5 * sorted(b["h"] for b in boxes)[len(boxes) // 2]
    lines, current = [], [boxes[0]]
    for b in boxes[1:]:
        if abs(b["y"] - sum(c["y"] for c in current) / len(current)) < tol:
            current.append(b)
        else:
            lines.append(current)
            current = [b]
    lines.append(current)
    return [" ".join(b["text"] for b in sorted(line, key=lambda b: b["x"])) for line in lines]


def normalize_date(s):
    """Return the first plausible date in the text as dd/mm/yyyy, or None."""
    for i, pat in enumerate(DATE_PATTERNS):
        for m in re.finditer(pat, s):
            a, b, c = m.groups()
            if i == 2:
                y, mo, d = int(a), int(b), int(c)
            elif i == 3:
                if b[:3].lower() not in MONTHS:
                    continue
                d, mo, y = int(a), MONTHS[b[:3].lower()], int(c)
            else:
                d, mo, y = int(a), int(b), int(c)
            if y < 100:
                y += 2000
            if 1 <= d <= 31 and 1 <= mo <= 12 and 1990 <= y <= 2035:
                return f"{d:02d}/{mo:02d}/{y}"
    return None


def to_float(amount):
    """'1,234.00' / '1.234,00' / '12.50' → float: the last separator is the decimal point,
    any earlier ones are thousands separators."""
    return float(re.sub(r"[., ]", "", amount[:-3]) + "." + amount[-2:])


def find_amounts(s):
    """Amounts in a string. Match on the text as-is first: stripping spaces up front glued neighbors
    together ('3.00 6.00' -> '3.006.00' -> 3006.00, found by a unit test). Only if nothing matches,
    retry without spaces, for OCR that splits a number ('193 .00')."""
    return AMOUNT.findall(s) or AMOUNT.findall(s.replace(" ", ""))


def normalize_amount(s, last=False):
    found = find_amounts(s)
    if not found:
        return None
    return f"{to_float(found[-1 if last else 0]):.2f}"


def total_priority(line):
    """0 = not a total line; 3 = 'payable'/'grand total'; 2 = 'total incl. tax'; 1 = plain 'total'."""
    s = line.lower().replace(" ", "")
    if "total" not in s and not any(k in s for k in STRONG_TOTAL):
        return 0
    if any(k in s for k in NOT_TOTAL):
        return 0
    if ("totalgst" in s or "gst/tax" in s or "totaltax" in s) and "incl" not in s and "inc." not in s:
        return 0  # the tax amount, not the total
    if any(k in s for k in STRONG_TOTAL):
        return 3
    if "incl" in s or "inc." in s:
        return 2
    return 1


def extract_fields(lines):
    text = "\n".join(lines)
    date = normalize_date(text)

    total, best = None, 0
    for i, line in enumerate(lines):
        p = total_priority(line)
        if p > best:  # strictly greater: among equal priorities, the first one wins
            # Amount on the same line (last number on it), or on the next line (two-column layouts).
            amount = normalize_amount(line, last=True) or (
                normalize_amount(lines[i + 1]) if i + 1 < len(lines) else None
            )
            if amount:
                total, best = amount, p
    if total is None:  # fallback: the largest amount on the receipt
        amounts = [to_float(a) for a in find_amounts(text)]
        total = f"{max(amounts):.2f}" if amounts else None

    company = lines[0] if lines else None
    return {"company": company, "date": date, "total": total, "n_lines": len(lines)}


def read_receipt(image_bgr):
    lines = ocr_lines(image_bgr)
    return extract_fields(lines), lines
