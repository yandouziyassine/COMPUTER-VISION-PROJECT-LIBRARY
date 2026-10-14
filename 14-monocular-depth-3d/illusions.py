"""Depth models on real optical illusions (Wikimedia Commons, open licenses; credits written to docs/).

Usage:
    python illusions.py
"""

import json
import re
import urllib.parse
import urllib.request

import matplotlib
import numpy as np
from PIL import Image

import models as M
from benchmark import DOCS
from depth3d import backproject, render_view
from nyu import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIR = ROOT.parent / "illusions"
UA = {"User-Agent": "cv-portfolio-research/1.0 (github.com/yandouziyassine)"}
TITLES = {
    "ames_room": ("File:Ames room forced perspective.jpg", "Two people the same height look giant and tiny: the room is not a box"),
    "street_art": ("File:3D street painting.jpg", "A flat painting on the ground that looks like a hole"),
    "den_helder": ("File:Den Helder - Julianaplein, 3D street art 2020.jpg", "3D street art seen from its intended viewpoint"),
    "colonnade": ("File:Colonnade, trompe l'oeil mural by John Pugh.jpg", "A flat wall painted as a colonnade"),
    "cinema_vox": ("File:Film crew in trompe l'oeil mural, Cinéma Vox, Chamonix-Mont-Blanc, 2007.jpg", "Painted film crew on a flat facade"),
    "mirrors": ("File:Man photographing himself in hotel bathroom with mirrors all around to generate illusion (cropped twice).jpg",
                "Facing mirrors: an endless corridor that does not exist"),
}


def fetch():
    DIR.mkdir(parents=True, exist_ok=True)
    q = {"action": "query", "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 1280,
         "titles": "|".join(t for t, _ in TITLES.values()), "format": "json"}
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(q)
    pages = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA)))["query"]["pages"].values()
    by_title = {p["title"]: p["imageinfo"][0] for p in pages}
    credits = []
    for key, (title, _) in TITLES.items():
        info = by_title[title]
        path = DIR / f"{key}.jpg"
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(urllib.request.Request(info["thumburl"], headers=UA)).read())
        md = info["extmetadata"]
        artist = re.sub(r"<[^>]+>", "", md.get("Artist", {}).get("value", "unknown")).strip()
        credits.append(f"- **{key}**: [{title}]({info['descriptionurl']}), {md.get('LicenseShortName', {}).get('value')}, "
                       f"by {artist}")
    (DOCS / "illusion_credits.md").write_text("# Illusion photo credits\n\n" + "\n".join(credits) + "\n", encoding="utf-8")


def main():
    fetch()
    est = M.load("da2-small")
    fig, axes = plt.subplots(len(TITLES), 3, figsize=(12, 3.3 * len(TITLES)))
    for row, (key, (_, caption)) in enumerate(TITLES.items()):
        img = Image.open(DIR / f"{key}.jpg").convert("RGB")
        img.thumbnail((960, 960))
        disp = M.predict(est, np.asarray(img))
        pts, cols, f = backproject(img, disp)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(caption, fontsize=9, loc="left")
        axes[row, 1].imshow(disp, cmap="inferno")
        axes[row, 1].set_title("what the model infers (bright = near)", fontsize=9, loc="left")
        axes[row, 2].imshow(render_view(pts, cols, f, img.size, 35))
        axes[row, 2].set_title("its 3D, seen from the side", fontsize=9, loc="left")
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(DOCS / "illusions.jpg", dpi=75)
    plt.close(fig)


if __name__ == "__main__":
    main()
