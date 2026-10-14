"""Zero-shot auto-labeling: text prompt → boxes (Grounding DINO) → masks (SAM).

Give it a folder of images and a list of class names; it writes YOLO-format box labels (and
optionally PNG instance masks) without any training. Useful to bootstrap a dataset that you
then review and correct instead of drawing every box by hand.

Usage:
    python autolabel.py --images my_photos/ --classes "car, person, traffic light" --out labels/
    python autolabel.py --images my_photos/ --classes "dog, frisbee" --out labels/ --masks
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import output_dir  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,  # noqa: E402
    SamModel,
    SamProcessor,
)

OUT = output_dir("autolabel")
DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"


class GroundingDino:
    def __init__(self):
        torch.set_grad_enabled(False)
        self.processor = AutoProcessor.from_pretrained(DINO_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).eval()

    def detect(self, image, classes, box_threshold=0.30, text_threshold=0.25):
        """Returns a list of (class_name, score, [x1, y1, x2, y2]) in pixels."""
        # Grounding DINO expects lowercase phrases separated by " . " and ending with "."
        prompt = " . ".join(c.lower() for c in classes) + " ."
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        dets = []
        for label, score, box in zip(res["text_labels"], res["scores"], res["boxes"]):
            name = match_class(label, classes)
            if name is not None:
                dets.append((name, float(score), [float(v) for v in box]))
        return dets


def match_class(phrase, classes):
    """Map the model's output phrase back to one of the requested class names."""
    phrase = phrase.strip().lower()
    if phrase in (c.lower() for c in classes):
        return next(c for c in classes if c.lower() == phrase)
    hits = [c for c in classes if c.lower() in phrase or phrase in c.lower()]
    return max(hits, key=len) if hits and phrase else None


class Segmenter:
    """SAM turns each box into a pixel mask (box prompt → mask)."""

    def __init__(self):
        self.processor = SamProcessor.from_pretrained(SAM_ID)
        self.model = SamModel.from_pretrained(SAM_ID).eval()

    def masks(self, image, boxes):
        if not boxes:
            return []
        inputs = self.processor(image, input_boxes=[boxes], return_tensors="pt")
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=False)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        )[0]
        return [m[0].numpy() for m in masks]


def to_yolo(dets, classes, size):
    """YOLO label lines: class_id cx cy w h (normalized)."""
    w, h = size
    lines = []
    for name, _, (x1, y1, x2, y2) in dets:
        lines.append(
            f"{classes.index(name)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
        )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--classes", required=True, help='comma-separated, e.g. "car, person, dog"')
    ap.add_argument("--out", type=Path, default=OUT / "labels")
    ap.add_argument("--masks", action="store_true", help="also save SAM instance masks as PNG")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    dino = GroundingDino()
    sam = Segmenter() if args.masks else None
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    counts = {c: 0 for c in classes}
    for p in images:
        img = Image.open(p).convert("RGB")
        dets = dino.detect(img, classes, box_threshold=args.box_threshold)
        (args.out / f"{p.stem}.txt").write_text("\n".join(to_yolo(dets, classes, img.size)))
        for name, *_ in dets:
            counts[name] += 1
        if sam:
            for k, m in enumerate(sam.masks(img, [d[2] for d in dets])):
                name = dets[k][0].replace(" ", "_")
                Image.fromarray((m * 255).astype(np.uint8)).save(args.out / f"{p.stem}_{k}_{name}.png")
        print(f"{p.name}: {len(dets)} objects", flush=True)

    (args.out / "classes.txt").write_text("\n".join(classes))
    print(json.dumps({"images": len(images), "objects_per_class": counts}))


if __name__ == "__main__":
    main()
