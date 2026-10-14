"""README figure: VQA examples (model answer vs most common human answer), right and wrong ones.

Usage (after evaluate.py):
    python make_figures.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from vlm import DATA, DOCS, OUT

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir  # noqa: E402

COCO = data_dir("clip-search") / "val2017"


def main():
    std = OUT / "vqa_answers_standard.json"  # answers with the standard VQA prompt, if available
    rows = json.loads((std if std.exists() else OUT / "vqa_answers.json").read_text())
    anns = {
        a["question_id"]: a
        for a in json.loads(next(DATA.glob("v2_mscoco_val2014_annotations.json")).read_text())["annotations"]
    }
    rng = random.Random(1)
    right = rng.sample([r for r in rows if r["acc"] == 1.0], 3)
    wrong = rng.sample([r for r in rows if r["acc"] == 0.0], 3)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, r in zip(axes.ravel(), right + wrong):
        ax.imshow(Image.open(COCO / f"{r['image_id']:012d}.jpg").convert("RGB"))
        ax.axis("off")
        human = Counter(x["answer"] for x in anns[r["question_id"]]["answers"]).most_common(1)[0][0]
        ok = r["acc"] == 1.0
        ax.set_title(
            f"Q: {r['q']}\nmodel: {r['pred']}   |   humans: {human}", fontsize=9, color="green" if ok else "red"
        )
    fig.suptitle("SmolVLM2-500M on VQAv2 (CPU). Top: correct · bottom: wrong", fontsize=12)
    fig.tight_layout()
    fig.savefig(DOCS / "vqa_examples.jpg", dpi=70)
    plt.close(fig)
    print("saved docs/vqa_examples.jpg")


if __name__ == "__main__":
    main()
