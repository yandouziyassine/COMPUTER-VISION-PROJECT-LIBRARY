"""Project 03: classical page detection and perspective warp on synthetic images."""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from conftest import load_module  # noqa: E402

scanner = load_module("03", "scanner.py")

# A slightly rotated/perspective "page" (TL, TR, BR, BL) on an 800x600 image.
PAGE = np.array([[180, 90], [600, 120], [640, 520], [140, 490]], dtype=np.float32)


def synthetic_photo(corners=PAGE, size=(600, 800)):
    """Dark, slightly colored table with a white quadrilateral page on it."""
    img = np.full((*size, 3), (40, 70, 90), np.uint8)
    cv2.fillPoly(img, [corners.astype(np.int32)], (245, 245, 245))
    return img


def test_order_corners_from_any_order():
    shuffled = PAGE[[2, 0, 3, 1]]
    np.testing.assert_array_equal(scanner.order_corners(shuffled), PAGE)


def test_order_corners_accepts_contour_shape():
    as_contour = PAGE[[1, 3, 0, 2]].reshape(4, 1, 2).astype(np.int32)
    np.testing.assert_array_equal(scanner.order_corners(as_contour), PAGE)


def test_plausible_page_accepts_normal_page():
    assert scanner.plausible_page(PAGE, (600, 800))


def test_plausible_page_rejects_quad_touching_border():
    touching = PAGE.copy()
    touching[0] = [0, 0]
    assert not scanner.plausible_page(touching, (600, 800))


def test_plausible_page_rejects_sliver():
    sliver = np.array([[100, 100], [700, 100], [700, 130], [100, 130]], np.float32)
    assert not scanner.plausible_page(sliver, (600, 800))


def test_plausible_page_rejects_sharp_corners():
    skewed = np.array([[100, 100], [400, 100], [700, 400], [400, 400]], np.float32)  # 45° parallelogram
    assert not scanner.plausible_page(skewed, (600, 800))


def test_find_document_recovers_corners():
    corners = scanner.find_document(synthetic_photo())
    assert corners is not None
    err = np.linalg.norm(scanner.order_corners(corners) - PAGE, axis=1)
    assert err.max() < 5, f"corner errors {err}"


def test_find_document_scales_back_to_original_pixels():
    big = cv2.resize(synthetic_photo(), (1600, 1200))  # find_document works at 600 px high internally
    corners = scanner.find_document(big)
    assert corners is not None
    assert np.linalg.norm(scanner.order_corners(corners) - 2 * PAGE, axis=1).max() < 10


def test_find_document_returns_none_without_page():
    assert scanner.find_document(np.full((600, 800, 3), (40, 70, 90), np.uint8)) is None


def test_warp_produces_flat_white_page():
    img = synthetic_photo()
    flat = scanner.warp(img, PAGE)
    h, w = flat.shape[:2]
    # Output size = longest opposite sides of the quadrilateral.
    assert w == int(max(np.linalg.norm(PAGE[2] - PAGE[3]), np.linalg.norm(PAGE[1] - PAGE[0])))
    assert h == int(max(np.linalg.norm(PAGE[1] - PAGE[2]), np.linalg.norm(PAGE[0] - PAGE[3])))
    inner = flat[5:-5, 5:-5]
    assert inner.mean() > 240  # only the page is left


def test_scan_without_page_returns_none_corners():
    out, corners = scanner.scan(np.full((300, 400, 3), 40, np.uint8))
    assert corners is None
    assert out.shape == (300, 400, 3)
