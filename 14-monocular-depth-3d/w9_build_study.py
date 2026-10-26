"""W9 · Human vs machine: build the "which point is closer?" study (pairs + images + static web page).

~300 point pairs, stratified by category, each with a physically justified answer:
  normal       NYU test images, both points on ordinary surfaces, raw sensor depth differs by >= 15 %
  textureless  both points in flat, texture-free regions (local intensity std < 4/255), depth differs >= 15 %
  mirror       one point on a mirror: its true depth is the mirror *surface*, a plane fitted to the sensor depth of
               the ring around the mirror (the sensor itself measures the reflected room inside it); differs >= 25 %
  glass        same with windows / glass (the sensor sees through or misses glass)
  poster       a flat framed picture pasted on a flat NYU wall with the wall's perspective (v1 poster test): both
               points lie on the wall plane, inside or beside the poster; their order is fixed by the wall's tilt
               (>= 6 %), while the picture's painted depth may suggest the opposite
  illusion     hand-annotated pairs on open-licensed illusion photos (configs/w9_illusion_pairs.json)

Outputs:
  docs/w9_pairs.json                         pair spec (ids, coordinates, truth, why): committed, no image data
  <cv-data>/depth/outputs/w9_study/          index.html + pairs.js + img/*.jpg, ready to host or zip for volunteers
                                             (NYU frames are not committed to the public repo, licence: research use)
Usage:
    python w9_build_study.py
"""

import json
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from metrics import backproject, eval_mask, fit_plane, plane_depth
from nyu import K, NYU
from poster_test import composite, poster_quad, wall_plane

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = output_dir("depth") / "w9_study"
ILLUSIONS = data_dir("depth") / "w9_illusions"
POSTER_PICS = data_dir("depth") / "w9_posters"
QUOTA = {"normal": 70, "textureless": 50, "mirror": 50, "glass": 40, "poster": 82}
UA = {"User-Agent": "cv-portfolio-research/1.0 (github.com/yandouziyassine)"}
# COCO val2017 photos used as poster content (CC BY 4.0 annotations; images keep their Flickr licences)
COCO_IDS = [139, 285, 632, 724, 776, 785, 802, 872, 885, 1000, 1268, 1296, 1353, 1425, 1490, 1503, 1532, 1584]
MIN_SEP = 60  # pixels between the two points of a pair


# ---------- pure helpers (unit-tested) ----------


def ring(mask, width=10):
    """Pixels within `width` px outside a region: the frame / wall around a mirror or window."""
    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    return (cv2.dilate(mask.astype(np.uint8), k) > 0) & ~mask


def surface_depth_from_ring(mask, raw, Kc, width=10, max_resid=0.03):
    """Plane through the valid sensor depth around `mask`; returns per-pixel plane depth (H, W) or None."""
    r = ring(mask, width) & (raw > 0)
    if r.sum() < 200:
        return None
    v, u = np.nonzero(r)
    pts = backproject(u, v, raw[v, u], Kc)
    n, d = fit_plane(pts)
    if np.median(np.abs(pts @ n + d)) > max_resid:
        return None
    vv, uu = np.mgrid[0 : raw.shape[0], 0 : raw.shape[1]]
    pd = plane_depth(uu, vv, n, d, Kc)
    return np.where(pd > 0, pd, np.nan)


def local_std(gray, k=15):
    g = gray.astype(np.float32)
    m = cv2.blur(g, (k, k))
    return np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - m * m, 0))


def pick_pair(rng, cand_a, cand_b, depth_a, depth_b, min_ratio, tries=200):
    """Two points (one from each candidate set) whose depths differ by at least `min_ratio`.
    Returns ((ya, xa), (yb, xb), za, zb) or None."""
    if len(cand_a[0]) == 0 or len(cand_b[0]) == 0:
        return None
    for _ in range(tries):
        i, j = rng.integers(len(cand_a[0])), rng.integers(len(cand_b[0]))
        a, b = (cand_a[0][i], cand_a[1][i]), (cand_b[0][j], cand_b[1][j])
        if np.hypot(a[0] - b[0], a[1] - b[1]) < MIN_SEP:
            continue
        za, zb = float(depth_a[a]), float(depth_b[b])
        if np.isfinite(za) and np.isfinite(zb) and za > 0 and zb > 0 and max(za, zb) / min(za, zb) >= min_ratio:
            return a, b, za, zb
    return None


def make_pair(pid, category, image, a, b, za, zb, why, rng, **extra):
    """Store a pair with A/B order randomised; coordinates as (x, y) pixels; truth = label of the nearer point."""
    pts = [(a, za), (b, zb)]
    swap = rng.random() < 0.5
    if swap:
        pts.reverse()
    (pa, z1), (pb, z2) = pts
    return {
        "id": pid,
        "category": category,
        "image": image,
        "a": [int(pa[1]), int(pa[0])],
        "b": [int(pb[1]), int(pb[0])],
        "depth_a": round(z1, 3),
        "depth_b": round(z2, 3),
        "truth": "a" if z1 < z2 else "b",
        "first_point": "b" if swap else "a",  # label of the first point given (mirror/glass surface, poster point)
        "why": why,
        **extra,
    }


# ---------- builders ----------


def nyu_pairs(data, rng):
    test = data.test_indices()
    train = [i for i in range(len(data.f["images"])) if i not in set(test)]
    ids = {n: data.class_id(n) for n in ("mirror", "window", "glass", "television", "monitor") if n in data.names}
    special = [ids[n] for n in ids]
    pairs, used = {k: [] for k in ("normal", "textureless", "mirror", "glass")}, {}
    # test images first; mirrors and glass are rare, so the labeled training images are used for them as well
    # (all models here are zero-shot on NYU)
    for i in [*rng.permutation(test), *rng.permutation(train)]:
        if all(len(pairs[c]) >= QUOTA[c] for c in pairs):
            break
        in_test = i in set(test)
        rgb, raw, lab = data.rgb(i), data.raw(i), data.labels(i)
        crop = eval_mask(np.full(raw.shape, 5.0))  # Eigen crop region
        valid = (raw > 0) & crop
        img = f"nyu_{i:04d}.jpg"

        def add(cat, pair, why, img=img, i=i, **extra):
            if pair is None or len(pairs[cat]) >= QUOTA[cat]:
                return
            a, b, za, zb = pair
            pairs[cat].append(make_pair(f"{cat}_{len(pairs[cat]):03d}", cat, img, a, b, za, zb, why, rng, **extra))
            used[img] = i

        ordinary = valid & ~np.isin(lab, special) & (lab > 0)
        if in_test and rng.random() < 0.35:
            c = np.nonzero(ordinary)
            add("normal", pick_pair(rng, c, c, raw, raw, 1.15), "Kinect depth, ordinary surfaces")
        flat = valid & (local_std(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)) < 4)
        flat = cv2.erode(flat.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if in_test and flat.sum() > 500:
            c = np.nonzero(flat)
            add("textureless", pick_pair(rng, c, c, raw, raw, 1.15), "Kinect depth, texture-free surfaces")
        for cat, names in (("mirror", ["mirror"]), ("glass", ["window", "glass"])):
            surf = np.isin(lab, [ids[n] for n in names if n in ids]) & crop
            if surf.sum() < 1500:
                continue
            nl, comp = cv2.connectedComponents(surf.astype(np.uint8))
            big = max(range(1, nl), key=lambda k: (comp == k).sum())
            region = comp == big
            plane = surface_depth_from_ring(region, raw, K)
            if plane is None:
                continue
            inner = cv2.erode(region.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
            outside = ordinary & ~ring(region, 15)
            for p in [pick_pair(rng, np.nonzero(inner), np.nonzero(outside), plane, raw, 1.25) for _ in range(3)]:
                if p is None:
                    continue
                a = p[0]
                sensor = float(raw[a]) if raw[a] > 0 else None
                why = f"{cat} surface depth = plane fitted to the sensor depth around it (frame / wall)"
                add(cat, p, why, sensor_depth_on_surface=sensor, nyu_split="test" if in_test else "train")
    return pairs, used


def fetch_posters():
    POSTER_PICS.mkdir(parents=True, exist_ok=True)
    for k in COCO_IDS:
        f = POSTER_PICS / f"{k:012d}.jpg"
        if not f.exists():
            try:
                f.write_bytes(urllib.request.urlopen(f"http://images.cocodataset.org/val2017/{k:012d}.jpg").read())
            except OSError:
                pass
    return sorted(POSTER_PICS.glob("*.jpg"))


def poster_pairs(data, rng, done_images):
    pics = fetch_posters()
    wall_id, pairs, images = data.class_id("wall"), [], {}
    for i in rng.permutation(data.test_indices()):
        if len(pairs) >= QUOTA["poster"]:
            break
        img = f"poster_{i:04d}.jpg"
        lab, gt = data.labels(i), data.depth(i)
        wall = (lab == wall_id) & eval_mask(gt)
        if wall.mean() < 0.15:
            continue
        n, d, resid = wall_plane(wall, gt, K)
        quad = poster_quad(wall, gt, K) if resid < 0.02 else None
        if quad is None:
            continue
        pic = cv2.cvtColor(cv2.imread(str(pics[len(images) % len(pics)])), cv2.COLOR_BGR2RGB)
        comp, poster = composite(data.rgb(i), quad, pic)
        vv, uu = np.mgrid[0:480, 0:640]
        plane = plane_depth(uu, vv, n, d, K)
        inner = cv2.erode(poster.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        beside = wall & ~ring(poster, 12) & ~poster
        added = 0
        for src_b, why in ((inner, "both points on the poster"), (beside, "poster point vs wall point beside it")):
            for _ in range(2):
                p = pick_pair(rng, np.nonzero(inner), np.nonzero(src_b), plane, plane, 1.06)
                if p is not None and len(pairs) < QUOTA["poster"]:
                    a, b, za, zb = p
                    why_full = f"flat poster on a flat wall: depth = wall plane fitted to the sensor; {why}"
                    pairs.append(make_pair(f"poster_{len(pairs):03d}", "poster", img, a, b, za, zb, why_full, rng))
                    added += 1
        if added:
            images[img] = comp
    return pairs, images


def illusion_pairs(rng):
    spec = json.loads((HERE / "configs" / "w9_illusion_pairs.json").read_text(encoding="utf-8"))
    ILLUSIONS.mkdir(parents=True, exist_ok=True)
    credits, images, pairs = {}, {}, []
    q = {
        "action": "query",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 1024,
        "titles": "|".join(spec["images"].values()),
        "format": "json",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(q)
    pages = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA)))["query"]["pages"].values()
    by_title = {p["title"]: p["imageinfo"][0] for p in pages}
    for key, title in spec["images"].items():
        info = by_title[title.replace("_", " ")]
        f = ILLUSIONS / f"{key}.jpg"
        if not f.exists():
            f.write_bytes(urllib.request.urlopen(urllib.request.Request(info["thumburl"], headers=UA)).read())
        m = info["extmetadata"]
        credits[key] = {
            "title": title,
            "page": info["descriptionurl"],
            "license": m.get("LicenseShortName", {}).get("value"),
            "artist": m.get("Artist", {}).get("value", ""),
        }
        images[f"ill_{key}.jpg"] = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
    for k, p in enumerate(spec["pairs"]):
        h, w = images[f"ill_{p['image']}.jpg"].shape[:2]
        a = (int(p["a"][1] * h), int(p["a"][0] * w))
        b = (int(p["b"][1] * h), int(p["b"][0] * w))
        za, zb = (1.0, 2.0) if p["truth"] == "a" else (2.0, 1.0)  # ordinal truth only
        pr = make_pair(f"illusion_{k:03d}", "illusion", f"ill_{p['image']}.jpg", a, b, za, zb, p["why"], rng)
        pr["kind"], pr["depth_a"], pr["depth_b"] = p["kind"], None, None
        pairs.append(pr)
    return pairs, images, credits


def main():
    rng = np.random.default_rng(14)
    data = NYU()
    nyu, used = nyu_pairs(data, rng)
    poster, poster_imgs = poster_pairs(data, rng, used)
    ill, ill_imgs, credits = illusion_pairs(rng)
    pairs = [p for c in nyu.values() for p in c] + poster + ill
    order = rng.permutation(len(pairs))  # presentation order (shuffled across categories)
    pairs = [pairs[k] for k in order]
    counts = {c: sum(p["category"] == c for p in pairs) for c in [*QUOTA, "illusion"]}
    spec = {"version": 1, "n_pairs": len(pairs), "counts": counts, "illusion_credits": credits, "pairs": pairs}
    (HERE / "docs" / "w9_pairs.json").write_text(json.dumps(spec, indent=1))

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "img").mkdir(parents=True)
    for img, i in used.items():
        cv2.imwrite(
            str(OUT / "img" / img), cv2.cvtColor(data.rgb(i), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
    for img, arr in {**poster_imgs, **ill_imgs}.items():
        cv2.imwrite(str(OUT / "img" / img), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    public = [{k: p[k] for k in ("id", "image", "a", "b")} for p in pairs]  # no answers sent to the browser
    (OUT / "pairs.js").write_text("window.PAIRS = " + json.dumps(public) + ";\n")  # works from file:// too
    shutil.copy(HERE / "web_study" / "index.html", OUT / "index.html")
    print(counts, "->", OUT)


if __name__ == "__main__":
    main()
