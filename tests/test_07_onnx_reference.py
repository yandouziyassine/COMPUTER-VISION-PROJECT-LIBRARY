"""Project 07: reference decode + NMS for the browser ONNX model (web/ppe_onnx_reference.py)."""

import pytest

np = pytest.importorskip("numpy")

from conftest import load_module  # noqa: E402

ref = load_module("07", "web/ppe_onnx_reference.py")


def make_output(rows):
    """rows: list of (cx, cy, w, h, s0, s1) -> (1, 6, 8400) tensor, rest zero."""
    out = np.zeros((1, 6, 8400), dtype=np.float32)
    for i, r in enumerate(rows):
        out[0, :, i] = r
    return out


META = {"orig_w": 640, "orig_h": 640, "ratio": 1.0, "pad_left": 0, "pad_top": 0}


def test_decode_threshold_and_class():
    out = make_output([(100, 100, 20, 20, 0.9, 0.1), (300, 300, 20, 20, 0.1, 0.8), (500, 500, 20, 20, 0.2, 0.24)])
    dets = ref.decode(out, META)
    assert [d["label"] for d in dets] == ["hardhat", "no-hardhat"]
    assert dets[0]["box_xyxy"] == [90.0, 90.0, 110.0, 110.0]


def test_nms_is_class_aware():
    rows = [(100, 100, 40, 40, 0.9, 0.0), (102, 100, 40, 40, 0.8, 0.0), (101, 100, 40, 40, 0.0, 0.7)]
    dets = ref.decode(make_output(rows), META)
    assert [(d["class_id"], d["score"]) for d in dets] == [(0, 0.9), (1, 0.7)]


def test_letterbox_unmapping_and_clip():
    meta = {"orig_w": 1280, "orig_h": 720, "ratio": 0.5, "pad_left": 0, "pad_top": 140}
    dets = ref.decode(make_output([(5, 320, 20, 40, 0.9, 0.0)]), meta)
    assert dets[0]["box_xyxy"] == [0.0, 320.0, 30.0, 400.0]


def test_letterbox_padding():
    pytest.importorskip("cv2")
    x, meta = ref.letterbox(np.zeros((540, 960, 3), dtype=np.uint8))
    assert x.shape == (1, 3, 640, 640)
    assert (meta["pad_left"], meta["pad_top"]) == (0, 140)
    assert x[0, 0, 0, 0] == pytest.approx(114 / 255) and x[0, 0, 320, 320] == 0
