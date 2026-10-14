"""Gradio search UI: type a description or drop in a photo, get the closest images.

Usage:
    python clip_search.py index   # once
    python app.py                 # then open http://127.0.0.1:7860
"""

import gradio as gr

from clip_search import ClipSearch

cs = ClipSearch()
cs.load()


def by_text(query, k):
    return [(p, f"{s:.3f}") for p, s in cs.search_text(query, int(k))] if query.strip() else []


def by_image(image, k):
    return [(p, f"{s:.3f}") for p, s in cs.search_image(image.convert("RGB"), int(k))] if image is not None else []


with gr.Blocks(title="CLIP Image Search") as demo:
    gr.Markdown(f"# CLIP Image Search\nSearch {len(cs.paths):,} COCO images by description or by example photo.")
    k = gr.Slider(4, 24, value=12, step=4, label="Results")
    with gr.Tab("Text → image"):
        q = gr.Textbox(label="Describe what you're looking for", placeholder="a cat sleeping on a laptop")
        g1 = gr.Gallery(columns=4, height="auto")
        q.submit(by_text, [q, k], g1)
    with gr.Tab("Image → image"):
        img = gr.Image(type="pil", label="Example photo")
        g2 = gr.Gallery(columns=4, height="auto")
        img.change(by_image, [img, k], g2)

if __name__ == "__main__":
    demo.launch()
