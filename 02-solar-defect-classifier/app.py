"""Gradio demo: upload a solar panel photo → predicted condition + Grad-CAM heatmap.

Usage:
    python app.py        # then open http://127.0.0.1:7860
"""

import gradio as gr
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from solar import CLASSES, IMG_SIZE, eval_tf, load_trained

model = load_trained()
cam = GradCAM(model=model, target_layers=[model.conv_head])


def predict(image):
    img = image.convert("RGB")
    x = eval_tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = model(x).softmax(1)[0].numpy()
    heat = cam(input_tensor=x, targets=None)[0]  # None → explain the predicted class
    rgb = np.asarray(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)
    return {c: float(p) for c, p in zip(CLASSES, probs)}, overlay


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Solar panel photo"),
    outputs=[gr.Label(num_top_classes=3, label="Prediction"), gr.Image(label="Grad-CAM: where the model looked")],
    title="Solar Panel Defect Classifier",
    description="EfficientNet-B0 fine-tuned on 6 panel conditions: bird droppings, clean, dusty, "
    "electrical damage, physical damage, snow.",
)

if __name__ == "__main__":
    demo.launch()
