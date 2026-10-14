"""Gradio demo: photo of a receipt → flattened scan → extracted fields as JSON.

Usage:
    python app.py        # then open http://127.0.0.1:7860
"""

import cv2
import gradio as gr
import numpy as np

from receipt import read_receipt
from scanner import scan


def process(image_rgb):
    bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    scanned, corners = scan(bgr)
    shown = bgr.copy()
    if corners is not None:
        cv2.polylines(shown, [corners.astype(np.int32)], True, (0, 255, 0), max(3, bgr.shape[1] // 200))
    fields, lines = read_receipt(scanned)
    fields["page_found"] = corners is not None
    return (cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), cv2.cvtColor(scanned, cv2.COLOR_BGR2RGB), fields, "\n".join(lines))


demo = gr.Interface(
    fn=process,
    inputs=gr.Image(type="numpy", label="Photo of a receipt (on a contrasting surface)"),
    outputs=[
        gr.Image(label="Detected page"),
        gr.Image(label="Scan"),
        gr.JSON(label="Extracted fields"),
        gr.Textbox(label="OCR text", lines=12),
    ],
    title="Receipt Scanner + OCR",
    description="Classical OpenCV page detection and perspective correction, RapidOCR (OpenVINO) "
    "text recognition, rule-based extraction of company, date and total.",
)

if __name__ == "__main__":
    demo.launch()
