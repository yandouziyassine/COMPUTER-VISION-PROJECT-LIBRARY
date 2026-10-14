"""Text-to-image (and image-to-image) search with CLIP embeddings + a FAISS index.

CLIP maps images and text into the same vector space, so "a dog on a beach" lands near photos
of dogs on beaches. We embed every image once, store the vectors in FAISS, and a search is a
single text embedding + nearest-neighbor lookup.

Usage:
    python clip_search.py index                       # embed all images (once)
    python clip_search.py search "a red double-decker bus"
    python clip_search.py evaluate                    # Recall@K with COCO captions
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402

import faiss  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

DATA = data_dir("clip-search")
IMAGES = DATA / "val2017"
CAPTIONS = DATA / "annotations" / "captions_val2017.json"
OUT = output_dir("clip-search")
DOCS = Path(__file__).resolve().parent / "docs"
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"


class ClipSearch:
    def __init__(self):
        torch.set_grad_enabled(False)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL, pretrained=PRETRAINED, cache_dir=str(OUT / "weights")
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL)
        self.index, self.paths = None, None

    def embed_text(self, texts):
        z = self.model.encode_text(self.tokenizer(texts))
        return torch.nn.functional.normalize(z, dim=-1).numpy().astype(np.float32)

    def embed_images(self, images):
        x = torch.stack([self.preprocess(im) for im in images])
        z = self.model.encode_image(x)
        return torch.nn.functional.normalize(z, dim=-1).numpy().astype(np.float32)

    def build(self, paths, batch=64):
        vecs = []
        t0 = time.time()
        for i in range(0, len(paths), batch):
            vecs.append(self.embed_images([Image.open(p).convert("RGB") for p in paths[i : i + batch]]))
            if (i // batch) % 10 == 0:
                print(f"embedded {i + batch}/{len(paths)} | {(i + batch) / (time.time() - t0):.1f} img/s", flush=True)
        vecs = np.concatenate(vecs)
        # Vectors are L2-normalized, so inner product == cosine similarity.
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.paths = [str(p) for p in paths]
        faiss.write_index(self.index, str(OUT / "images.faiss"))
        (OUT / "paths.json").write_text(json.dumps(self.paths))
        return time.time() - t0

    def load(self):
        self.index = faiss.read_index(str(OUT / "images.faiss"))
        self.paths = json.loads((OUT / "paths.json").read_text())

    def search_text(self, query, k=8):
        scores, ids = self.index.search(self.embed_text([query]), k)
        return [(self.paths[i], float(s)) for i, s in zip(ids[0], scores[0])]

    def search_image(self, image, k=8):
        scores, ids = self.index.search(self.embed_images([image]), k)
        return [(self.paths[i], float(s)) for i, s in zip(ids[0], scores[0])]


def save_grid(query, hits, path):
    fig, axes = plt.subplots(1, len(hits), figsize=(2.6 * len(hits), 3))
    for ax, (p, s) in zip(axes, hits):
        im = Image.open(p).convert("RGB")
        im.thumbnail((320, 320))
        ax.imshow(im)
        ax.set_title(f"{s:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f'"{query}"', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def evaluate(cs, max_captions=5000):
    """Text→image retrieval on COCO val2017: for each caption, is its image in the top K?"""
    ann = json.loads(CAPTIONS.read_text())
    pos = {Path(p).name: i for i, p in enumerate(cs.paths)}
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    pairs = [
        (a["caption"].strip(), pos[id2file[a["image_id"]]]) for a in ann["annotations"] if id2file[a["image_id"]] in pos
    ]
    rng = np.random.default_rng(0)
    pairs = [pairs[i] for i in rng.permutation(len(pairs))[:max_captions]]
    hits = {1: 0, 5: 0, 10: 0}
    for i in range(0, len(pairs), 256):
        chunk = pairs[i : i + 256]
        _, ids = cs.index.search(cs.embed_text([c for c, _ in chunk]), 10)
        for (_, target), row in zip(chunk, ids):
            for k in hits:
                hits[k] += int(target in row[:k])
    return {f"R@{k}": round(v / len(pairs), 4) for k, v in hits.items()} | {
        "n_captions": len(pairs),
        "n_images": len(cs.paths),
    }


DEMO_QUERIES = [
    "a red double-decker bus",
    "a dog catching a frisbee",
    "people surfing a big wave",
    "a plate of pizza with vegetables",
    "a giraffe next to a tree",
    "a snowy mountain with skiers",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["index", "search", "evaluate", "demo"])
    ap.add_argument("query", nargs="?")
    ap.add_argument("-k", type=int, default=6)
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    cs = ClipSearch()

    if args.command == "index":
        paths = sorted(IMAGES.glob("*.jpg"))
        secs = cs.build(paths)
        print(f"indexed {len(paths)} images in {secs:.0f}s ({len(paths) / secs:.1f} img/s)")
        return

    cs.load()
    if args.command == "search":
        for p, s in cs.search_text(args.query, args.k):
            print(f"{s:.3f}  {p}")
    elif args.command == "evaluate":
        t0 = time.time()
        r = evaluate(cs)
        r["query_ms"] = round((time.time() - t0) / r["n_captions"] * 1000, 2)
        r["model"] = f"{MODEL} ({PRETRAINED})"
        print(r)
        (DOCS / "results.json").write_text(json.dumps(r, indent=1))
    elif args.command == "demo":
        for i, q in enumerate(DEMO_QUERIES):
            save_grid(q, cs.search_text(q, args.k), DOCS / f"query_{i + 1}.jpg")
        print("saved", len(DEMO_QUERIES), "query grids to docs/")


if __name__ == "__main__":
    main()
