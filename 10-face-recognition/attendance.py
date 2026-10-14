"""Minimal attendance check-in: enroll people from a folder of photos, then identify faces.

    enroll/
      alice/  photo1.jpg photo2.jpg
      bob/    photo1.jpg

Only embeddings (128 numbers per person) are stored, not photos. Use only with the consent of
the people enrolled.

Usage:
    python attendance.py enroll enroll/
    python attendance.py identify group_photo.jpg           # labels every face in the photo
    python attendance.py identify 0                         # webcam (q to quit)
"""

import argparse
import datetime as dt
import json
import sys

import cv2
import numpy as np

from faces import OUT, FaceEmbedder

GALLERY = OUT / "gallery.npz"
THRESHOLD = 0.31  # mean 10-fold LFW threshold for the FP32 model (docs/results.json: 0.313)


def enroll(folder, emb):
    names, vecs = [], []
    for person in sorted(p for p in folder.iterdir() if p.is_dir()):
        photos = [f for f in person.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        es = [e for e in (emb.embed(cv2.imread(str(f))) for f in photos) if e is not None]
        if es:
            m = np.mean(es, axis=0)
            names.append(person.name)
            vecs.append(m / np.linalg.norm(m))
            print(f"enrolled {person.name} from {len(es)} photo(s)")
    np.savez(GALLERY, names=np.array(names), vecs=np.array(vecs))


def label_faces(img, emb, names, vecs, log):
    for face in emb.detect(img):
        e = emb.embed(img, face)
        sims = vecs @ e
        k = int(np.argmax(sims))
        who = names[k] if sims[k] > THRESHOLD else "unknown"
        x, y, w, h = face[:4].astype(int)
        color = (0, 200, 0) if who != "unknown" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{who} {sims[k]:.2f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if who != "unknown" and who not in log:
            log[who] = dt.datetime.now().isoformat(timespec="seconds")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["enroll", "identify"])
    ap.add_argument("path")
    args = ap.parse_args()
    emb = FaceEmbedder()
    if args.command == "enroll":
        from pathlib import Path

        enroll(Path(args.path), emb)
        return
    if not GALLERY.exists():
        sys.exit("run 'enroll' first")
    g = np.load(GALLERY)
    names, vecs, log = list(g["names"]), g["vecs"], {}
    if args.path == "0":
        cap = cv2.VideoCapture(0)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            cv2.imshow("attendance", label_faces(f, emb, names, vecs, log))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        img = label_faces(cv2.imread(args.path), emb, names, vecs, log)
        cv2.imwrite(str(OUT / "identified.jpg"), img)
    print(json.dumps({"checked_in": log}, indent=1))


if __name__ == "__main__":
    main()
