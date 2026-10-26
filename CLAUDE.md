# CLAUDE.md — Computer Vision Project Library

Portfolio of 20 computer vision projects built to land a CV/ML job. Recruiters skim:
every project must run, show real results, and explain itself in its README.

## Layout

- `NN-short-name/` — one folder per project (see root `README.md` for the list and status).
- `_template/README.md` — README skeleton + checklist for new projects. Use it.
- `common/` — shared utilities. Only add code here once 2+ projects need it.
- `.venv/` — shared CPU deep-learning env (projects 02+). Rebuild with `setup_env.ps1`.
- `01-virtual-painter/.venv/` — separate env (mediapipe 0.10.14 needs numpy<2).
- Data lives **outside the repo** in `../cv-data/<project>/` (sibling folder: `C:\Users\yando\Code\cv-data`).
  Caches (pip, Hugging Face, torch hub) go to `../cv-data/` too, never to the default locations on `C:`.

## Machine constraints

- No NVIDIA GPU. Intel Core Ultra 7 258V, 32 GB RAM, Arc iGPU + NPU. **CPU training only**:
  fine-tune small pretrained models, small image sizes, few epochs. OpenVINO for acceleration.
- The repo lives at `C:\Users\yando\Code\COMPUTER-VISION-PROJECT-LIBRARY` (the USB key was lost on 2026-10-23).
  `C:` is nearly full (~27 GB free on 2026-10-26): keep big datasets on Kaggle and download only what a run needs.
- GPU work runs on **free Kaggle GPUs** (T4/P100) through Kaggle kernels; the laptop stays CPU-only.

## Rules

1. **Commit and push finished, tested work to `main`** (the repo is public on GitHub since 2026-10-14):
   one focused commit per project or stage, CI green (ruff format, ruff, pytest) before pushing.
   Never commit data, weights, videos or secrets. Unfinished experiments stay uncommitted.
2. **Data, weights and videos stay in `../cv-data/`**, never inside the repo folder (`.gitignore`
   covers them anyway). Small result images (`docs/*.png`, `docs/*.gif`, < 2 MB each) belong in the repo.
3. **Finish a project fully before starting the next**: code runs → real metrics measured →
   README filled from `_template` with real numbers and result images → root README status
   updated → `PROGRESS.md` updated. A complete project beats two half-done ones.
8. **`PROGRESS.md` is the overnight log**: what's done, what's in progress, what's next, and
   anything the owner must know. Read it first when resuming; update it after every milestone.
9. **Obsidian vault** (`C:\Users\yando\Documents\SuperVault\Projets\CV-Project-Library\`, notes in
   **French**, prefixed `cvlib-`): after each finished project, update `cvlib-00-accueil.md`
   (finished-projects table), `cvlib-01-feuille-de-route.md` (status),
   `cvlib-02-journal-de-construction.md` (what was done + results + lessons) and
   `cvlib-04-preparation-entrevue.md` (STAR talking points). Follow the vault's `AGENTS.md`. No git commits there either.
4. **Report results honestly.** Real numbers only. Say what didn't work and why.
5. **Code style**: plain Python scripts (`train.py`, `predict.py`, `app.py`…), `argparse`
   for options, paths built from `common/paths.py`, no notebooks as the main deliverable.
6. **Projects needing a webcam** must also accept a video file, so they can be tested without one.
7. **Dataset licenses**: note source + license in each README. Only use datasets downloadable
   without login; if one needs an account, document the manual step instead.

## Token discipline

- Long jobs (training, installs, downloads) run in the background and log to a file;
  read only the tail/summary lines. Never dump full training logs or progress bars.
- Prefer `--quiet`/`verbose=False` flags; print one summary line per epoch at most.
- Check results via numbers and saved files, not screenshots.
- Use well-trodden libraries (timm, ultralytics, supervision, openvino) to avoid long fix loops.
- Extract archives with `tar -m -xf` on the FAT32 key: without `-m`, tar prints a
  "Can't restore time" warning for every file (thousands of lines).
- Background-job output files: read only their tail or grep them, never the whole file.

## Long jobs survive disconnects (lessons from 2026-10-12/13)

- The USB key disconnected mid-run twice (2026-10-12/13), and harness background tasks have been killed
  (exit -1) after long runs or laptop sleep. For any job > 30 min: launch it as a **detached process**
  (`Start-Process ... -WindowStyle Hidden -RedirectStandardOutput/-Error ... -PassThru`, save the PID to
  a file), then wait with a separate background loop on that PID.
- Make long jobs **resumable**: YOLO `train.py --resume` (last.pt every epoch), checkpoint files for
  evaluation loops (project 18 saves every 5 questions), history written every epoch.
- Before resuming, check that the env imports and that the run's checkpoint and log files are where you expect.

## Skills & agents

- `superpowers:using-superpowers` — check for a relevant skill before starting a task.
- `engineering:system-design` — use when designing a project's pipeline (input → model → output,
  trade-offs) and write the result into the README's "How it works" section.
- `superpowers:systematic-debugging` — on any error, find the root cause before patching.
- `superpowers:verification-before-completion` — run it and see the output before saying "done".
- `caveman:cavecrew` — delegate small read-only lookups (`cavecrew-investigator`), 1–2 file edits
  (`cavecrew-builder`) and diff reviews (`cavecrew-reviewer`) to keep main context small.
- `superpowers:dispatching-parallel-agents` — only for truly independent work (e.g. writing docs
  while a training run is going). Never run two CPU training jobs at once: they fight for the CPU.
- Communication style: caveman (terse) for chat; code, commits and READMEs in normal English.
