"""Provenance stamp for generated results: the commit of the library and the date (UTC)."""

import datetime as dt
import subprocess
from pathlib import Path


def generated_by() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"commit": commit, "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")}
