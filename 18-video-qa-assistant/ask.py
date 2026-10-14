"""Ask a question about an image or a video, fully offline on CPU.

Usage:
    python ask.py photo.jpg "How many people are in this picture?"
    python ask.py clip.mp4 "What happens in this video?" --frames 8
"""

import argparse
from pathlib import Path

from PIL import Image

from vlm import VLM, sample_frames

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media", type=Path)
    ap.add_argument("question")
    ap.add_argument("--frames", type=int, default=8, help="frames sampled from a video")
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()
    vlm = VLM()
    if args.media.suffix.lower() in VIDEO_EXT:
        media = sample_frames(args.media, args.frames)
        question = f"These {len(media)} frames are from one video, in time order. {args.question}"
    else:
        media = Image.open(args.media).convert("RGB")
        question = args.question
    print(vlm.ask(media, question, max_new_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
