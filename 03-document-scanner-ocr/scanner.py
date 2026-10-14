"""Classical document scanner: find the page in a photo and warp it flat.

    photo → find the page (bright + colorless region, else edge contours) → 4 corners
          → order corners → perspective transform → flat "scan" → lighting correction

No deep learning: just OpenCV. Works when the paper contrasts with the background.
"""

import cv2
import numpy as np


def order_corners(pts):
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def quad_from_contour(c):
    """Simplify a contour to 4 corners (convex hull + polygon approximation), else its min-area rectangle."""
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return order_corners(approx)
    return order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))


def plausible_page(quad, shape):
    """Reject shapes no photographed page could have: touching the image border (the page must be
    fully in frame, as in scanner apps), concave, very sharp/obtuse corners, slivers.
    Without the border rule, background regions bounded by the photo frame look like clean rectangles."""
    h, w = shape[:2]
    margin = 0.01 * max(h, w)
    if (
        quad[:, 0].min() < margin
        or quad[:, 1].min() < margin
        or quad[:, 0].max() > w - 1 - margin
        or quad[:, 1].max() > h - 1 - margin
    ):
        return False
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2).astype(np.float32)):
        return False
    sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    if min(sides) < 0.2 * max(sides):
        return False
    for i in range(4):
        a, b = quad[i - 1] - quad[i], quad[(i + 1) % 4] - quad[i]
        angle = np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1, 1)))
        if not 55 <= angle <= 125:
            return False
    return True


def find_by_color(small):
    """Paper is bright and colorless. Threshold (brightness − saturation) at several levels and
    keep the blob that best looks like a page: large, and filling its quadrilateral almost exactly.

    One threshold is not enough: uneven lighting or a shadow can split the page (threshold too
    high) or merge it with a bright background (threshold too low)."""
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    score = cv2.GaussianBlur(cv2.subtract(hsv[..., 2], hsv[..., 1]), (5, 5), 0)
    otsu, _ = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = [score > t for t in {otsu, *np.percentile(score, [60, 70, 80, 90]).astype(int)}]
    # Local (adaptive) thresholds: "clearly brighter than the surroundings" survives lighting
    # gradients and soft shadows that a single global threshold cannot. The negative offset matters:
    # with 0, half of any flat background passes on noise alone.
    for block in (201, 301, 451):
        for c in (-10, -20):
            masks.append(cv2.adaptiveThreshold(score, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, c) > 0)
    k = np.ones((7, 7), np.uint8)
    img_area = small.shape[0] * small.shape[1]
    best, best_score = None, 0.0
    for m in masks:
        mask = m.astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3), cv2.MORPH_OPEN, k, iterations=2
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            frac = cv2.contourArea(c) / img_area
            if not 0.08 < frac < 0.85:
                continue
            quad = quad_from_contour(c)
            if not plausible_page(quad, small.shape):
                continue
            fill = cv2.contourArea(c) / max(cv2.contourArea(quad), 1)  # 1.0 = perfect quadrilateral
            s = frac * fill**8  # strongly prefer clean shapes
            if fill > 0.88 and s > best_score:
                best, best_score = quad, s
    return best


def find_document(image, work_height=600):
    """Return the 4 document corners in original-image pixels, or None if nothing plausible was found.
    Tries the color-based detector first, then edge contours."""
    scale = work_height / image.shape[0]
    small = cv2.resize(image, (int(image.shape[1] * scale), work_height))
    quad = find_by_color(small)
    if quad is not None:
        return quad / scale

    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    area_min = 0.1 * small.shape[0] * small.shape[1]

    # Try a few edge thresholds: low-contrast photos need gentler ones.
    for lo, hi in ((50, 150), (25, 90), (10, 50)):
        edges = cv2.dilate(cv2.Canny(gray, lo, hi), np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(c) < area_min:
                break
            peri = cv2.arcLength(c, True)
            for eps in (0.02, 0.03, 0.05):
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    return order_corners(approx / scale)
        # Fallback: the largest contour's minimum-area rectangle (handles torn/rounded corners).
        if contours:
            big = max(contours, key=cv2.contourArea)
            if cv2.contourArea(big) >= area_min:
                return order_corners(cv2.boxPoints(cv2.minAreaRect(big)) / scale)
    return None


def warp(image, corners):
    """Perspective-warp the quadrilateral to a straight-on rectangle."""
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(corners, dst), (w, h))


def enhance(scan):
    """Even out lighting/shadows so text is crisp: divide by a blurred background estimate."""
    gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 21)
    norm = cv2.divide(gray, background, scale=255)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


def scan(image):
    """Full pipeline. Returns (scanned image, corners or None). Falls back to the input if no page is found."""
    corners = find_document(image)
    if corners is None:
        return enhance(image), None
    return enhance(warp(image, corners)), corners
