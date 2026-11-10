"""Rerun every results script of the coach, in dependency order, then stamp each results JSON with generated_by.

python -m coach.rebuild_results            # ~1 h on CPU; the A-League steps need the raw store on the USB key
python -m coach.rebuild_results --idsse    # only the steps that need the 7 IDSSE matches (~30 min)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .stamp import generated_by

DOCS = Path(__file__).resolve().parents[1] / "docs"
IDSSE = [
    ["coach.evaluate"],
    ["coach.report"],
    ["coach.benchmark", "--set", "idsse"],
]
ALEAGUE = [
    ["coach.cross_league"],
    ["coach.shot_pooled"],
    ["coach.profiles_eval"],
    ["coach.benchmark", "--set", "aleague"],
    ["coach.benchmark", "--set", "pooled"],
    ["coach.benchmark", "--cutbacks"],
    ["coach.cutback"],
]


def stamp_all(stamp: dict) -> list:
    done = []
    for f in sorted(DOCS.glob("*results*.json")) + sorted(DOCS.glob("coach_benchmark_cutbacks.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data["generated_by"] = stamp
            f.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            done.append(f.name)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idsse", action="store_true")
    args = ap.parse_args()
    steps = IDSSE if args.idsse else IDSSE + ALEAGUE
    for step in steps:
        print(">>", " ".join(step), flush=True)
        subprocess.run([sys.executable, "-m", *step], check=True, cwd=Path(__file__).resolve().parents[1])
    print("stamped", stamp_all(generated_by()))


if __name__ == "__main__":
    main()
