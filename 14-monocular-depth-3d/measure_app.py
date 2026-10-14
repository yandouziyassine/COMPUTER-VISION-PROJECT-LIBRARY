"""Click two points on a photo, get the distance in meters (with the error measured on NYU).

Usage:
    python measure_app.py           # opens http://127.0.0.1:7860
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from measure import Measurer

RESULTS = json.loads((Path(__file__).resolve().parent / "docs" / "results.json").read_text())["measure"]
P90 = RESULTS["metric_model_only"]["p90_rel_error"]
P90_REF = RESULTS["with_one_known_length"]["p90_rel_error"]
measurer = Measurer()


def on_upload(img):
    return img, [], None, "Click two points."


def on_click(state_img, points, depth, evt: gr.SelectData):
    if state_img is None:
        return None, points, depth, "Upload a photo first."
    img = Image.fromarray(state_img)
    if depth is None:
        depth = measurer.depth(img)
    points = (points + [tuple(evt.index)])[-2:]
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 91, 31), width=3)
    if len(points) < 2:
        return np.asarray(canvas), points, depth, "Click a second point."
    draw.line([points[0], points[1]], fill=(255, 91, 31), width=3)
    d, source = measurer.measure(img, points[0], points[1], depth=depth)
    msg = (f"**{d:.2f} m** (±{P90 * 100:.0f} % for 9 measurements out of 10 on NYU indoor scenes).  \n"
           f"Camera: {source}. Indoor model: outdoor scenes will be less accurate.")
    return np.asarray(canvas), points, depth, msg


with gr.Blocks(title="Measure from one photo") as demo:
    gr.Markdown("# Measure from one photo\nUpload an indoor photo, click two points. Metric depth from Depth Anything V2 "
                f"(indoor); with one known length in the photo the 90th-percentile error drops to ±{P90_REF * 100:.0f} %.")
    img_state, pts_state, depth_state = gr.State(), gr.State([]), gr.State()
    with gr.Row():
        inp = gr.Image(type="numpy", label="Photo (click two points)")
        out = gr.Markdown()
    inp.upload(on_upload, inp, [img_state, pts_state, depth_state, out])
    inp.select(on_click, [img_state, pts_state, depth_state], [inp, pts_state, depth_state, out])

if __name__ == "__main__":
    demo.launch()
