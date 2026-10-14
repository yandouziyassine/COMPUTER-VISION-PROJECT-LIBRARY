"""Local vision-language model (SmolVLM2-500M) for questions about images and videos, on CPU.

A video is handled as a handful of frames sampled evenly over time and given to the model as a
sequence of images, followed by the question.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402

import cv2  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DATA = data_dir("vqa")
OUT = output_dir("vqa")
DOCS = Path(__file__).resolve().parent / "docs"


class VLM:
    def __init__(self, threads=4):
        torch.set_num_threads(threads)  # performance cores (see project 19)
        torch.set_grad_enabled(False)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        # By default SmolVLM splits each image into many 512 px tiles (thousands of tokens per question,
        # ~40 s on CPU). One 512 px view per image keeps it to ~64 image tokens: fast enough on a laptop.
        self.processor.image_processor.do_image_splitting = False
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32).eval()

    def ask(self, images, question, max_new_tokens=24):
        """images: one PIL image or a list (e.g. video frames, in time order)."""
        images = images if isinstance(images, list) else [images]
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": question}]
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()


def sample_frames(path, n=8, crop=None):
    """n frames evenly spaced over the video. crop: function applied to each BGR frame."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            f = crop(f) if crop else f
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    return frames
