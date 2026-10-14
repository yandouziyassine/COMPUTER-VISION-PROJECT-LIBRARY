"""Traffic analytics on a highway video: detect → track → count → estimate speed → heatmap.

Pipeline
    YOLO11n (COCO-pretrained) detects vehicles in each frame
    ByteTrack links detections across frames into tracks with stable IDs
    A counting line records each track crossing it (both directions)
    A perspective transform maps the road from pixels to meters → speed in km/h
    A heatmap accumulates where vehicles drive

Usage:
    python traffic.py                         # downloads the sample video on first run
    python traffic.py --video my_clip.mp4 --max-frames 300

The road calibration (SOURCE polygon, real-world size) is for the sample video. For another
camera, mark 4 road points in the image and measure the real width/length they span.
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402  (must come before ultralytics)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import supervision as sv  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DATA = data_dir("traffic")
OUT = output_dir("traffic")
DOCS = Path(__file__).resolve().parent / "docs"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO ids

# Calibration for the sample video (3840x2160): four road corners in pixels, and the
# real-world size of that patch of road (approx. 25 m wide x 250 m long).
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]], dtype=np.float32)
TARGET_W, TARGET_H = 25, 250
TARGET = np.array([[0, 0], [TARGET_W, 0], [TARGET_W, TARGET_H], [0, TARGET_H]], dtype=np.float32)
LINE = (sv.Point(446, 1400), sv.Point(3504, 1400))


class ViewTransformer:
    """Maps image points to road coordinates in meters (bird's-eye view)."""

    def __init__(self, source, target):
        self.m = cv2.getPerspectiveTransform(source, target)

    def __call__(self, points):
        if len(points) == 0:
            return points
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.m).reshape(-1, 2)


def get_sample_video():
    path = DATA / "vehicles.mp4"
    if not path.exists():
        from supervision.assets import VideoAssets, download_assets

        download_assets(VideoAssets.VEHICLES)
        Path(VideoAssets.VEHICLES.value).replace(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out-width", type=int, default=1280, help="width of the saved video")
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)

    video = args.video or get_sample_video()
    info = sv.VideoInfo.from_video_path(str(video))
    fps = int(round(info.fps))
    model = YOLO(str(OUT / args.model))  # weights are downloaded into cv-data, not the repo

    tracker = sv.ByteTrack(frame_rate=fps)
    line_zone = sv.LineZone(start=LINE[0], end=LINE[1])
    road_zone = sv.PolygonZone(polygon=SOURCE.astype(int))
    to_road = ViewTransformer(SOURCE, TARGET)

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=info.resolution_wh)
    box_ann = sv.BoxAnnotator(thickness=thickness)
    label_ann = sv.LabelAnnotator(
        text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER
    )
    trace_ann = sv.TraceAnnotator(thickness=thickness, trace_length=fps * 2, position=sv.Position.BOTTOM_CENTER)
    heat_ann = sv.HeatMapAnnotator(position=sv.Position.BOTTOM_CENTER, opacity=0.5, radius=40)
    line_ann = sv.LineZoneAnnotator(thickness=thickness, text_scale=text_scale)

    road_y = defaultdict(lambda: deque(maxlen=fps))  # track id → last second of road positions (m)
    speeds = {}  # track id → latest speed km/h
    track_class = {}
    heat_frame = None
    n_frames = info.total_frames if not args.max_frames else min(args.max_frames, info.total_frames)
    out_h = int(info.height * args.out_width / info.width)
    out_info = sv.VideoInfo(width=args.out_width, height=out_h, fps=fps, total_frames=n_frames)
    gif_frames = []

    t0 = time.time()
    with sv.VideoSink(str(OUT / "traffic_annotated.mp4"), out_info) as sink:
        for i, frame in enumerate(sv.get_video_frames_generator(str(video))):
            if i >= n_frames:
                break
            result = model(frame, imgsz=960, conf=args.conf, verbose=False)[0]
            det = sv.Detections.from_ultralytics(result)
            det = det[np.isin(det.class_id, list(VEHICLE_CLASSES))]
            det = det[road_zone.trigger(det)]
            det = det.with_nms(threshold=0.6)
            det = tracker.update_with_detections(det)
            line_zone.trigger(det)

            anchors = det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            road_pts = to_road(anchors)
            labels = []
            for tid, cid, (_, y) in zip(det.tracker_id, det.class_id, road_pts):
                track_class[tid] = VEHICLE_CLASSES[cid]
                road_y[tid].append(y)
                if len(road_y[tid]) >= fps // 2:
                    dist = abs(road_y[tid][-1] - road_y[tid][0])
                    speeds[tid] = dist / ((len(road_y[tid]) - 1) / fps) * 3.6
                    labels.append(f"#{tid} {speeds[tid]:.0f} km/h")
                else:
                    labels.append(f"#{tid}")

            annotated = trace_ann.annotate(frame.copy(), det)
            annotated = box_ann.annotate(annotated, det)
            annotated = label_ann.annotate(annotated, det, labels=labels)
            annotated = line_ann.annotate(annotated, line_counter=line_zone)
            heat_frame = heat_ann.annotate(frame.copy(), det)  # accumulates internally

            small = cv2.resize(annotated, (args.out_width, out_h))
            sink.write_frame(small)
            if i % 6 == 0 and len(gif_frames) < 45:  # small GIF for the README (< 2 MB)
                gif_frames.append(cv2.resize(annotated, (400, int(info.height * 400 / info.width))))
            if i % 50 == 0:
                print(
                    f"frame {i}/{n_frames} | tracks {len(track_class)} | "
                    f"in {line_zone.in_count} out {line_zone.out_count} | {(i + 1) / (time.time() - t0):.1f} FPS",
                    flush=True,
                )

    elapsed = time.time() - t0
    cv2.imwrite(
        str(DOCS / "heatmap.jpg"),
        cv2.resize(heat_frame, (960, int(info.height * 960 / info.width))),
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )
    cv2.imwrite(
        str(DOCS / "annotated_frame.jpg"),
        cv2.resize(annotated, (960, int(info.height * 960 / info.width))),
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )
    save_gif(gif_frames, DOCS / "demo.gif", fps=max(1, fps // 6))

    valid_speeds = [s for s in speeds.values() if 5 < s < 200]
    summary = {
        "video": video.name,
        "resolution": list(info.resolution_wh),
        "frames_processed": n_frames,
        "processing_fps": round(n_frames / elapsed, 2),
        "line_in": line_zone.in_count,
        "line_out": line_zone.out_count,
        "unique_tracks": len(track_class),
        "tracks_by_class": dict(Counter(track_class.values())),
        "mean_speed_kmh": round(float(np.mean(valid_speeds)), 1) if valid_speeds else None,
        "median_speed_kmh": round(float(np.median(valid_speeds)), 1) if valid_speeds else None,
    }
    with open(DOCS / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))


def save_gif(frames, path, fps):
    from PIL import Image

    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=64) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)


if __name__ == "__main__":
    main()
