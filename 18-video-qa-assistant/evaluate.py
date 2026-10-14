"""Two measurable checks of a small local vision-language model (no API, CPU only).

1. Image questions: VQAv2 validation questions on COCO images we already have (project 16).
   Simplified VQA accuracy: an answer counts min(#humans who gave it / 3, 1) — 10 humans answered each
   (the official metric also averages over leave-one-annotator-out subsets).
2. Video question: "Does the person fall?" on the 70 UR Fall videos (project 15), zero-shot,
   compared with the specialized pose-based detector from project 15.

Usage:
    python evaluate.py --n-vqa 300
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from vlm import DATA, DOCS, OUT, VLM, sample_frames

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir  # noqa: E402

COCO_IMAGES = data_dir("clip-search") / "val2017"
FALL_VIDEOS = data_dir("fall") / "videos"
NUMBERS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def normalize(ans):
    """Simplified VQA answer normalization: lowercase, strip punctuation/articles, number words → digits."""
    a = re.sub(r"[^\w\s]", "", ans.lower()).strip()
    words = [NUMBERS.get(w, w) for w in a.split() if w not in ("a", "an", "the")]
    return " ".join(words)


VQA_PROMPTS = {
    # First version: the small model often answered "1" even to yes/no questions (30/300 answers).
    "word_or_number": " Answer with a single word or number.",
    # The phrasing commonly used in VQAv2 evaluations of vision-language models.
    "standard": " Answer the question using a single word or phrase.",
}


def vqa_eval(vlm, n, seed=0, prompt="word_or_number"):
    from PIL import Image

    qs = json.loads(next(DATA.glob("v2_OpenEnded_mscoco_val2014_questions.json")).read_text())["questions"]
    anns = {
        a["question_id"]: a
        for a in json.loads(next(DATA.glob("v2_mscoco_val2014_annotations.json")).read_text())["annotations"]
    }
    have = {int(p.stem) for p in COCO_IMAGES.glob("*.jpg")}
    pool = [q for q in qs if q["image_id"] in have]
    random.Random(seed).shuffle(pool)
    # Checkpointing: the USB key has disconnected mid-run before, so progress is saved as we go
    # and a restart skips questions already answered.
    ckpt = OUT / ("vqa_checkpoint.json" if prompt == "word_or_number" else f"vqa_checkpoint_{prompt}.json")
    rows = json.loads(ckpt.read_text()) if ckpt.exists() else []
    done = {r["question_id"] for r in rows}
    t = [r["seconds"] for r in rows]
    for q in pool[:n]:
        if q["question_id"] in done:
            continue
        a = anns[q["question_id"]]
        img = Image.open(COCO_IMAGES / f"{q['image_id']:012d}.jpg").convert("RGB")
        t0 = time.time()
        pred = vlm.ask(img, q["question"] + VQA_PROMPTS[prompt], max_new_tokens=8)
        t.append(time.time() - t0)
        humans = Counter(normalize(x["answer"]) for x in a["answers"])
        p = normalize(pred).split(" ")[0] if a["answer_type"] in ("yes/no", "number") else normalize(pred)
        rows.append(
            {
                "question_id": q["question_id"],
                "q": q["question"],
                "pred": pred,
                "type": a["answer_type"],
                "acc": min(humans[p] / 3, 1.0),
                "image_id": q["image_id"],
                "seconds": t[-1],
            }
        )
        if len(rows) % 5 == 0:
            ckpt.write_text(json.dumps(rows))
            print(f"vqa {len(rows)}/{n}", flush=True)
    ckpt.write_text(json.dumps(rows))
    res = {
        "prompt": VQA_PROMPTS[prompt].strip(),
        "n": len(rows),
        "pool_questions_on_our_images": len(pool),
        "accuracy": round(float(np.mean([r["acc"] for r in rows])), 3),
        "s_per_question": round(float(np.mean(t)), 2),
    }
    for typ in ("yes/no", "number", "other"):
        sub = [r["acc"] for r in rows if r["type"] == typ]
        res[f"accuracy_{typ}"] = round(float(np.mean(sub)), 3) if sub else None
        res[f"n_{typ}"] = len(sub)
    return res, rows


FALL_QUESTIONS = {
    "fall": "These frames are from one video, in time order. Does the person fall down during the video? Answer yes or no.",
    # Concrete visual variant: asks about the end state instead of the event.
    "lying": "These frames are from one video, in time order. In the last frames, is the person lying on the floor? "
    "Answer yes or no.",
}


def fall_eval(vlm, n_frames=8, variant="fall"):
    question = FALL_QUESTIONS[variant]
    ckpt = OUT / f"fall_checkpoint_{variant}.json"
    rows = json.loads(ckpt.read_text()) if ckpt.exists() else []
    done = {r["video"] for r in rows}
    t = [r["seconds"] for r in rows]
    for p in sorted(FALL_VIDEOS.glob("*-cam0.mp4")):
        if p.stem in done:
            continue
        frames = sample_frames(p, n_frames, crop=lambda f: f[:, f.shape[1] // 2 :])  # RGB half of depth|RGB
        t0 = time.time()
        ans = vlm.ask(frames, question, max_new_tokens=4).lower()
        t.append(time.time() - t0)
        rows.append(
            {
                "video": p.stem,
                "fall": p.name.startswith("fall"),
                "said_yes": ans.startswith("yes"),
                "answer": ans,
                "seconds": t[-1],
            }
        )
        ckpt.write_text(json.dumps(rows))
    falls = [r for r in rows if r["fall"]]
    adls = [r for r in rows if not r["fall"]]
    sens = np.mean([r["said_yes"] for r in falls])
    spec = np.mean([not r["said_yes"] for r in adls])
    return {
        "question": question,
        "videos": len(rows),
        "frames_per_video": n_frames,
        "sensitivity": round(float(sens), 3),
        "specificity": round(float(spec), 3),
        "accuracy": round(float(np.mean([r["said_yes"] == r["fall"] for r in rows])), 3),
        "s_per_video": round(float(np.mean(t)), 1),
    }, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vqa", type=int, default=300)
    ap.add_argument(
        "--only-fall-variant",
        choices=list(FALL_QUESTIONS),
        default=None,
        help="run just one fall-question variant and merge it into docs/results.json",
    )
    ap.add_argument(
        "--only-vqa-prompt",
        choices=list(VQA_PROMPTS),
        default=None,
        help="run just the VQA subset with another prompt and merge it into docs/results.json",
    )
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    vlm = VLM()
    res_path = DOCS / "results.json"
    if args.only_vqa_prompt:
        res = json.loads(res_path.read_text())
        vqa, rows = vqa_eval(vlm, args.n_vqa, prompt=args.only_vqa_prompt)
        print("VQA", json.dumps(vqa), flush=True)
        res[f"vqav2_val_subset_{args.only_vqa_prompt}"] = vqa
        res_path.write_text(json.dumps(res, indent=1))
        (OUT / f"vqa_answers_{args.only_vqa_prompt}.json").write_text(json.dumps(rows, indent=1))
        return
    if args.only_fall_variant:
        res = json.loads(res_path.read_text())
        fall, fall_rows = fall_eval(vlm, variant=args.only_fall_variant)
        print("FALL", json.dumps(fall), flush=True)
        res[f"ur_fall_zero_shot_{args.only_fall_variant}"] = fall
        res_path.write_text(json.dumps(res, indent=1))
        (OUT / f"fall_answers_{args.only_fall_variant}.json").write_text(json.dumps(fall_rows, indent=1))
        return
    vqa, vqa_rows = vqa_eval(vlm, args.n_vqa)
    print("VQA", json.dumps(vqa), flush=True)
    fall, fall_rows = fall_eval(vlm)
    print("FALL", json.dumps(fall), flush=True)
    # Merge, so results from the --only-* variant runs are kept.
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res.update({"vqav2_val_subset": vqa, "ur_fall_zero_shot": fall})
    res_path.write_text(json.dumps(res, indent=1))
    (OUT / "vqa_answers.json").write_text(json.dumps(vqa_rows, indent=1))
    (OUT / "fall_answers.json").write_text(json.dumps(fall_rows, indent=1))


if __name__ == "__main__":
    main()
