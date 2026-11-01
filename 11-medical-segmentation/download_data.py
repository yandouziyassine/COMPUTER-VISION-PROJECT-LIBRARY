"""Download Kvasir-SEG (training + same-center test) and CVC-ClinicDB (second-center test).

Both are free for research/education, need no login, and land in ../cv-data/kvasir/ (outside the repo).
Archives are deleted after extraction to save disk space.

Usage:
    python download_data.py            # both datasets
    python download_data.py --only cvc
"""

import argparse
import urllib.request
import zipfile

from polyp import CVC_ROOT, ROOT

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"  # official (Simula), 44 MB
# The official CVC-ClinicDB page (polyp.grand-challenge.org/CVCClinicDB/) no longer shows a working
# link, so we use a public Hugging Face file mirror of the same 612 PNG frames + masks (~20 MB).
CVC_REPO = "berkaytrhn/cvc-clinicdb"


def get_kvasir():
    if (ROOT / "images").exists():
        print("Kvasir-SEG already present:", ROOT)
        return
    zpath = ROOT.parent / "kvasir-seg.zip"
    print("downloading", KVASIR_URL)
    urllib.request.urlretrieve(KVASIR_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(ROOT.parent)
    zpath.unlink()
    print("images:", len(list((ROOT / "images").glob("*.jpg"))))


def get_cvc():
    if (CVC_ROOT / "Original").exists():
        print("CVC-ClinicDB already present:", CVC_ROOT)
        return
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=CVC_REPO,
        repo_type="dataset",
        local_dir=str(CVC_ROOT.parent),
        allow_patterns=["CVC-ClinicDB/Original/*.png", "CVC-ClinicDB/Ground Truth/*.png", "*.csv", "README.md"],
    )
    print("frames:", len(list((CVC_ROOT / "Original").glob("*.png"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["kvasir", "cvc"])
    args = ap.parse_args()
    if args.only in (None, "kvasir"):
        get_kvasir()
    if args.only in (None, "cvc"):
        get_cvc()


if __name__ == "__main__":
    main()
