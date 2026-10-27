"""W9 public pair set for the website game: open-licensed images only (iBims-1 CC BY 4.0, Wikimedia Commons
CC0 / public domain / CC BY / CC BY-SA). No NYU, KITTI, ETH3D or Booster frames.

Categories and where the answer comes from:
  normal, textureless  iBims-1 laser-scan depth (>= 15 % apart)
  glass                iBims-1 windows / glass (mask_transp): surface depth = plane fitted to the laser depth of the
                       frame around it (the scanner sees through glass); >= 25 % apart
  poster               a Commons illusion photo pasted with the wall's perspective on a flat iBims-1 wall; both points
                       on the wall plane fitted to the laser depth, ordered by the wall's tilt (>= 6 %)
  mirror               hand-annotated Commons photos (configs/w9_public_mirror_pairs.json), geometry-certain answers
  illusion             hand-annotated Commons illusion photos (configs/w9_illusion_pairs.json)

Step 1  python w9_public.py --build      -> <cv-data>/depth/outputs/w9_public/ (images + internal spec with truth)
Step 2  Kaggle model answers (w9_kaggle_models.py on that folder, uploaded as a private Kaggle dataset)
Step 3  python w9_public.py --assemble <w9_model_answers.json>
        -> docs/w9_public_pairs.json + docs/w9_public_img/*.jpg (the website schema)
"""

import argparse
import io
import json
import re
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from metrics import backproject, fit_plane, plane_depth
from poster_test import composite, poster_quad
from w9_build_study import local_std, make_pair, pick_pair, ring, surface_depth_from_ring

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = output_dir("depth") / "w9_public"
IBIMS = data_dir("depth") / "ibims" / "ibims1_core_mat.zip"
QUOTA = {"normal": 20, "textureless": 16, "glass": 16, "poster": 16}
UA = {"User-Agent": "cv-portfolio-research/1.0 (github.com/yandouziyassine)"}
MODEL_IDS = {
    "da2-large": ("dav2l", "Depth Anything V2 Large"),
    "da2-base": ("dav2b", "Depth Anything V2 Base"),
    "da2-small": ("dav2s", "Depth Anything V2 Small"),
    "marigold-lcm": ("marigold", "Marigold LCM"),
    "dpt-hybrid": ("dpt", "DPT-Hybrid (MiDaS 3.0)"),
    "depth-pro": ("depthpro", "Depth Pro"),
    "moge2-large": ("moge2", "MoGe-2"),
    "da2-metric-indoor": ("dav2metric", "Depth Anything V2 Metric Indoor"),
}
# Pairs dropped after review because their answer is not geometry-certain (ids kept stable for the website).
EXCLUDE = {
    "p003": "B may lie on the shower glass rather than the side wall: answer not certain",
    "p016": "the brick wall is oblique, so the mirror may be nearer than the basin: answer not certain",
}
POSTER_CONTENT = ["colonnade", "film_crew", "street_boots", "street_parrots"]
POSTER_TITLES = {
    "colonnade": "File:Colonnade, trompe l'oeil mural by John Pugh.jpg",
    "film_crew": "File:Film crew in trompe l'oeil mural, Cinéma Vox, Chamonix-Mont-Blanc, 2007.jpg",
}


ALLOWED_LICENCES = re.compile(r"^(CC0|Public domain|CC BY \d\.\d|CC BY-SA \d\.\d|MIT|Apache-2\.0)$")


def normalise_licence(lic):
    """Licence string in the website's accepted form (CC0, Public domain, CC BY x.y, CC BY-SA x.y, MIT, Apache-2.0)."""
    s = lic.strip()
    if s.upper().startswith("CC0"):
        s = "CC0"
    elif s.lower() in ("public domain", "pd", "pdm"):
        s = "Public domain"
    if not ALLOWED_LICENCES.match(s):
        raise ValueError(f"licence {lic!r} is not accepted for public display")
    return s


def commons(titles, folder):
    """Download Commons files (1024 px thumbnails) with licence metadata. {key: (rgb, credit)}."""
    folder.mkdir(parents=True, exist_ok=True)
    q = {
        "action": "query",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 1024,
        "titles": "|".join(titles.values()),
        "format": "json",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(q)
    pages = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA)))["query"]["pages"].values()
    by_title = {p["title"]: p["imageinfo"][0] for p in pages}
    out = {}
    for key, title in titles.items():
        info = by_title[title.replace("_", " ")]
        f = folder / f"{key}.jpg"
        if not f.exists():
            f.write_bytes(urllib.request.urlopen(urllib.request.Request(info["thumburl"], headers=UA)).read())
        m = info["extmetadata"]
        lic = m.get("LicenseShortName", {}).get("value", "")
        assert "NC" not in lic and "ND" not in lic, f"{title}: {lic} is not permissive"
        artist = m.get("Artist", {}).get("value", "")
        credit = {
            "source": f"Wikimedia Commons, {re.sub('<[^>]+>', '', artist).strip()}",
            "license": lic,
            "url": info["descriptionurl"],
        }
        rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        s = min(1.0, 1280 / max(rgb.shape[:2]))  # website limit, applied before any coordinate is computed
        if s < 1:
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
        out[key] = (rgb, credit)
    return out


def ibims_items():
    with zipfile.ZipFile(IBIMS) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".mat")):
            d = loadmat(io.BytesIO(z.read(name)))["data"]
            C = d["calib"][0, 0]
            yield (
                Path(name).stem,
                d["rgb"][0, 0],
                d["depth"][0, 0].astype(np.float32),
                d["mask_invalid"][0, 0] > 0,
                d["mask_transp"][0, 0] == 0,
                d["mask_wall"][0, 0],
                (C[0, 0], C[1, 1], C[0, 2], C[1, 2]),
            )


IBIMS_CREDIT = {
    "source": "iBims-1, Koch et al. 2018, TUM (mediaTUM 1455541)",
    "license": "CC BY 4.0",
    "url": "https://mediatum.ub.tum.de/1455541",
}


def build():
    rng = np.random.default_rng(141)
    pairs, images, credits = {c: [] for c in [*QUOTA, "mirror", "illusion"]}, {}, {}
    posters = commons(POSTER_TITLES, data_dir("depth") / "w9_illusions")
    ill_spec = json.loads((HERE / "configs" / "w9_illusion_pairs.json").read_text(encoding="utf-8"))
    ills = commons(ill_spec["images"], data_dir("depth") / "w9_illusions")
    for k in ("street_boots", "street_parrots"):
        posters[k] = ills[k]
    vv, uu = np.mgrid[0:480, 0:640]
    items = list(ibims_items())
    for idx in rng.permutation(len(items)):
        name, rgb, depth, valid, transp, wall_lab, K = items[idx]
        margin = np.zeros(depth.shape, bool)
        margin[20:-20, 20:-20] = True  # no points at the image border
        ok = valid & ~transp & (depth > 0) & margin
        img = f"ibims_{name}.jpg"

        def add(cat, p, why, img=img, image=rgb, **extra):
            if p is None or len(pairs[cat]) >= QUOTA[cat]:
                return False
            a, b, za, zb = p
            pairs[cat].append(make_pair(f"{cat}_{len(pairs[cat]):02d}", cat, img, a, b, za, zb, why, rng, **extra))
            images[img], credits[img] = image, IBIMS_CREDIT
            return True

        if rng.random() < 0.3:
            c = np.nonzero(cv2.erode(ok.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
            add("normal", pick_pair(rng, c, c, depth, depth, 1.15), "Laser-scan depth (iBims-1).")
        flat = ok & (local_std(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)) < 4)
        flat = cv2.erode(flat.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if flat.sum() > 500 and rng.random() < 0.5:
            c = np.nonzero(flat)
            add(
                "textureless",
                pick_pair(rng, c, c, depth, depth, 1.15),
                "Laser-scan depth on texture-free surfaces (iBims-1).",
            )
        if transp.sum() > 1500:
            nl, comp = cv2.connectedComponents(transp.astype(np.uint8))
            region = comp == max(range(1, nl), key=lambda k: (comp == k).sum())
            ys, xs = np.nonzero(region)
            fill = region.sum() / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1))
            # panes only: large and nearly rectangular (a glass vase or bottle is not a plane)
            plane = (
                surface_depth_from_ring(region, np.where(ok, depth, 0), K)
                if region.sum() > 10000 and fill > 0.7
                else None
            )
            if plane is not None:
                inner = (cv2.erode(region.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0) & margin
                outside = ok & ~ring(region, 15)
                for _ in range(4):
                    add(
                        "glass",
                        pick_pair(rng, np.nonzero(inner), np.nonzero(outside), plane, depth, 1.25),
                        "Glass depth = plane fitted to the laser depth of its frame (the scanner sees through glass).",
                    )
        if len(pairs["poster"]) < QUOTA["poster"] and wall_lab.max() > 0:
            labs, counts = np.unique(wall_lab[wall_lab > 0], return_counts=True)
            wall = (wall_lab == labs[counts.argmax()]) & ok
            if wall.mean() > 0.15:
                v, u = np.nonzero(wall)
                pts = backproject(u, v, depth[v, u], K)
                n, d = fit_plane(pts)
                quad = poster_quad(wall, depth, K) if np.median(np.abs(pts @ n + d)) < 0.02 else None
                if quad is not None:
                    key = POSTER_CONTENT[len(pairs["poster"]) // 2 % len(POSTER_CONTENT)]
                    comp_img, mask = composite(rgb, quad, posters[key][0])
                    plane = plane_depth(uu, vv, n, d, K)
                    inner = (cv2.erode(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & margin
                    beside = wall & ~ring(mask, 12) & ~mask
                    pimg = f"poster_{name}.jpg"
                    for src_b in (inner, beside):
                        if add(
                            "poster",
                            pick_pair(rng, np.nonzero(inner), np.nonzero(src_b), plane, plane, 1.06),
                            "A flat picture on a flat wall: depth = wall plane fitted to the laser scan; the picture adds no depth.",
                            img=pimg,
                            image=comp_img,
                        ):
                            credits[pimg] = {**IBIMS_CREDIT, "also": posters[key][1]}
    for spec_file, cat, imgs in (
        ("w9_public_mirror_pairs.json", "mirror", None),
        ("w9_illusion_pairs.json", "illusion", ills),
    ):
        spec = json.loads((HERE / "configs" / spec_file).read_text(encoding="utf-8"))
        got = imgs or commons(spec["images"], data_dir("depth") / "w9_mirrors")
        for k, p in enumerate(spec["pairs"]):
            arr, credit = got[p["image"]]
            h, w = arr.shape[:2]
            a, b = (int(p["a"][1] * h), int(p["a"][0] * w)), (int(p["b"][1] * h), int(p["b"][0] * w))
            za, zb = (1.0, 2.0) if p["truth"] == "a" else (2.0, 1.0)  # ordinal truth only
            img = f"{cat}_{p['image']}.jpg"
            pairs[cat].append(make_pair(f"{cat}_{k:02d}", cat, img, a, b, za, zb, p["why"], rng))
            images[img], credits[img] = arr, credit
    allp = [p for c in pairs.values() for p in c]
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "img").mkdir(parents=True)
    for img, arr in images.items():
        if any(p["image"] == img for p in allp):
            save_jpeg(arr, OUT / "img" / img)
    (OUT / "w9_pairs.json").write_text(json.dumps({"pairs": allp, "credits": credits}, indent=1))
    print({c: len(v) for c, v in pairs.items()}, "->", OUT)


def save_jpeg(rgb, path, max_side=1280, max_bytes=150_000):
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    assert s == 1.0, "images are resized before pairs are built"
    for q in (90, 85, 80, 75, 70, 60, 50, 40, 30):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        if len(buf) <= max_bytes:
            break
    Path(path).write_bytes(buf.tobytes())
    return s


def assemble(answers_path):
    spec = json.loads((OUT / "w9_pairs.json").read_text())
    ans = json.loads(Path(answers_path).read_text())["models"]
    dst = HERE / "docs" / "w9_public_img"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(spec["pairs"]))
    out_pairs, credits, names = [], [], {}
    for n, k in enumerate(order, 1):
        p = spec["pairs"][k]
        pid = f"p{n:03d}"
        if pid in EXCLUDE:
            continue
        src = OUT / "img" / p["image"]
        im = cv2.imread(str(src))
        h, w = im.shape[:2]
        names.setdefault(p["image"], pid)
        img_name = f"{names[p['image']]}.jpg"
        if not (dst / img_name).exists():
            shutil.copy(src, dst / img_name)
            c = spec["credits"][p["image"]]
            credits.append({"img": f"w9_public_img/{img_name}", **{k2: c[k2] for k2 in ("source", "license", "url")}})
            if "also" in c:
                credits[-1]["includes"] = c["also"]
        out_pairs.append(
            {
                "id": pid,
                "img": f"w9_public_img/{img_name}",
                "w": w,
                "h": h,
                "a": p["a"],
                "b": p["b"],
                "closer": p["truth"],
                "category": p["category"],
                "why": p["why"],
                "models": {MODEL_IDS[m][0]: ans[m]["answers"][p["id"]]["answer"] for m in MODEL_IDS if m in ans},
            }
        )
    doc = {
        "version": 1,
        "excluded": EXCLUDE,
        "models": [{"id": i, "name": nm} for m, (i, nm) in MODEL_IDS.items() if m in ans],
        "pairs": out_pairs,
        "credits": credits,
    }
    (HERE / "docs" / "w9_public_pairs.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    # the volunteer study uses exactly these pairs and ids: spec with truth + model answers keyed by public id
    idmap = {spec["pairs"][k]["id"]: f"p{n:03d}" for n, k in enumerate(order, 1) if f"p{n:03d}" not in EXCLUDE}
    study_spec = [
        {
            "id": q["id"],
            "category": q["category"],
            "image": Path(q["img"]).name,
            "a": q["a"],
            "b": q["b"],
            "truth": q["closer"],
            "why": q["why"],
        }
        for q in out_pairs
    ]
    (HERE / "docs" / "w9_pairs.json").write_text(
        json.dumps({"version": 2, "set": "public", "pairs": study_spec}, indent=1)
    )
    remapped = {}
    for m, mm in ans.items():
        r = {k2: v for k2, v in mm.items() if k2 not in ("answers", "by_transform")}
        r["answers"] = {idmap[k2]: v for k2, v in mm["answers"].items() if k2 in idmap}
        if "by_transform" in mm:
            r["by_transform"] = {
                tr: {idmap[k2]: v for k2, v in a2.items() if k2 in idmap} for tr, a2 in mm["by_transform"].items()
            }
        remapped[m] = r
    (HERE / "docs" / "w9_model_answers.json").write_text(json.dumps({"set": "public", "models": remapped}))
    # volunteer study bundle over the same open-licensed pairs (static: index.html + pairs.js + img/)
    study = OUT / "study"
    if study.exists():
        shutil.rmtree(study)
    shutil.copytree(dst, study / "img")
    public = [{"id": p["id"], "image": Path(p["img"]).name, "a": p["a"], "b": p["b"]} for p in out_pairs]
    (study / "pairs.js").write_text("window.PAIRS = " + json.dumps(public) + ";\n")
    shutil.copy(HERE / "web_study" / "index.html", study / "index.html")
    truth = [{"id": p["id"], "truth": p["closer"], "category": p["category"]} for p in out_pairs]
    (OUT / "study_truth.json").write_text(json.dumps({"pairs": truth}))
    for m, (i, _) in MODEL_IDS.items():
        if m in ans:
            acc = np.mean([p["models"][i] == p["closer"] for p in out_pairs])
            print(f"{i:10s} {acc:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--assemble")
    a = ap.parse_args()
    if a.build:
        build()
    if a.assemble:
        assemble(a.assemble)
