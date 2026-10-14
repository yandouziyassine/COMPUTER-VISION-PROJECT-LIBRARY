# Capstone: smart site camera (project 21)

Date: 2026-10-14 · Status: proposed, waiting for the owner's approval

## Goal

One application that combines several library projects into the system a construction company would actually
deploy: a camera watching a work site that raises **safety alerts**, running in real time on the laptop's
**NPU** (no GPU). It shows integration skills: several models, one video loop, alert logic, deployment and a
measured latency budget.

## Footage (openly licensed, downloaded 2026-10-14)

| Clip | Content | License |
|---|---|---|
| *Building construction Moira Close, Broadwater Farm, Haringey 2025 (21)*, Wikimedia Commons, 333 s, 1080p | workers with hard hats and hi-vis on decks, a telehandler, a crane load | CC BY-SA 4.0 |
| *Workers dismantling scaffolding*, Wikimedia Commons, 79 s, 1080p | scaffolders working at height | CC BY-SA 3.0 |

Clips stay in `cv-data/` (never committed); figures and a short GIF derived from them are shared under the same
CC BY-SA license with attribution.

## Pipeline (`site_camera.py`)

1. **People detection + tracking**: YOLO11n (COCO person class) + ByteTrack, as in project 08.
2. **Hard-hat check**: project 07's hard-hat detector on the same frame; each tracked person is matched to the
   head box inside the top third of their body box. A violation must persist for 2 s on the same track before it
   becomes an alert (no flicker alerts).
3. **Exclusion zones**: polygons drawn once per camera (e.g. under the crane load, around the telehandler);
   alert when a tracked person's feet enter one.
4. **Person-down alert**: project 15's pose features (YOLO11n-pose) on tracked people, using the fall classifier
   when the camera angle allows it, otherwise the "lying + not moving for N seconds" rule.
5. **Alert log**: one JSON line per alert (time, track id, type, snapshot path) and an annotated video/GIF.
6. **Deployment**: all three networks exported to OpenVINO; run the loop on CPU, iGPU and NPU and report the
   per-stage latency budget and end-to-end FPS (reusing project 19's tooling).

## Evaluation (honest about the lack of ground truth)

- **Hand-checked alert precision**: every alert raised on the two clips is saved as a snapshot; I review each one
  visually and label it correct / wrong (expected: tens of alerts). Reported as precision with the count.
- **Recall on a sampled audit**: 60 frames sampled uniformly; I mark people without a visible hard hat and
  people inside zones; report how many the system flagged.
- **Detector-level numbers** already measured in projects 07 (hard hats, mAP50 0.828) and 15 (falls) are cited,
  not re-claimed.
- **Latency**: per stage and end to end, CPU vs iGPU vs NPU, idle machine, median of 3 runs.

## Deliverables

`21-smart-site-camera/` with `site_camera.py`, `zones.json` per clip, `export_openvino.py`, `evaluate.py`,
README (results, latency table, alert gallery, limitations), unit tests for track ↔ head matching, zone
entry and alert debouncing. Website card + root README row.

## Out of scope

Training new models; multi-camera; a web dashboard (possible follow-up reusing project 01's FastAPI stream).

## Risks

- Hard-hat model was trained on close-up photos; far, small heads in these clips may be missed. The audit will
  show it; if recall is poor, run the hard-hat model on upscaled person crops instead of the full frame.
- Falls are rare in real footage (none expected here): the person-down path is demonstrated on project 15's
  UR Fall clips locally (not redistributed), and labelled as such.
