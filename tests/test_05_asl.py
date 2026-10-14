"""Project 05: letter mapping, webcam-crop preprocessing and the CNN's output shape.
asl.py creates data folders at import time, so only the needed definitions are loaded."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from conftest import load_defs  # noqa: E402

asl = load_defs("05", "asl.py", ["LABELS", "LETTERS", "hand_crop_to_input"], {"np": np})


def test_24_static_letters_without_j_and_z():
    assert len(asl.LETTERS) == 24
    assert "J" not in asl.LETTERS and "Z" not in asl.LETTERS
    assert asl.LETTERS[0] == "A" and asl.LETTERS[-1] == "Y"
    assert asl.LETTERS[asl.LABELS.index(10)] == "K"  # dataset label 10 is K: indices skip J, letters don't


def test_crop_becomes_square_28x28_and_keeps_aspect_ratio():
    crop = np.full((60, 20), 50, np.uint8)       # tall, narrow hand on a dark background
    crop[:, 8:12] = 250                            # bright vertical finger in the middle
    out = asl.hand_crop_to_input(crop)
    assert out.shape == (28, 28) and out.dtype == np.uint8
    assert out[14, 14] > 200                       # finger stays centered
    assert out[14, 2] < 100 and out[14, 25] < 100  # padding uses the background tone, not white


def test_cnn_outputs_one_score_per_letter():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    cnn = load_defs("05", "asl.py", ["SignCNN"], {"nn": nn, "LABELS": asl.LABELS}).SignCNN()
    assert cnn(torch.zeros(2, 1, 28, 28)).shape == (2, 24)
