# 14 · Monocular Depth → 3D

One ordinary photo in, 3D out. **Depth Anything V2** predicts a depth map from a single RGB image;
every pixel is back-projected through a pinhole camera model into a colored **3D point cloud**,
which is then re-rendered from new camera angles, a "3D photo" effect created from a 2D image.

![3D photo sweep](docs/demo.gif)

*Generated from one still photo: the camera "moves" around a scene that was only seen from one viewpoint.*

## Results

![depth maps](docs/depth_maps.jpg)

![new viewpoints](docs/000000000724_views.jpg)

*Rotating the virtual camera ±15°: the stop sign shifts against the background (parallax).
Dark regions are areas hidden behind foreground objects in the original photo, which one image can't reveal.*

| Metric | Value |
|---|---|
| Model | Depth Anything V2 Small (25 M parameters) |
| Depth inference, CPU | ~650 ms per image (Intel Core Ultra 7 258V, ~640×480 input) |
| Point cloud | up to 200k colored points per image, saved as `.ply` |

More views: [living room](docs/000000000139_views.jpg) · [bedroom](docs/000000000632_views.jpg) · [skier](docs/000000000785_views.jpg)

## How it works

1. **Depth estimation**: Depth Anything V2 (a ViT trained on a mix of labeled data and millions of
   pseudo-labeled images) outputs *relative inverse depth* (disparity) for every pixel.
2. **Disparity → depth**: normalized disparity `d ∈ [0, 1]` becomes depth `z = 1 / (0.15 + 0.85·d)`.
   The offset keeps the sky and far background at a finite distance instead of infinity.
3. **Back-projection**: with an assumed 60° horizontal field of view, focal length `f = (W/2) / tan(30°)`,
   and each pixel `(u, v)` becomes the point `((u − W/2)·z/f, (v − H/2)·z/f, z)`.
4. **Point cloud**: colored points are saved with Open3D (`.ply`), keeping up to 200k points and
   dropping the farthest 2 % (background noise).
5. **New views**: the cloud is rotated around the scene center and projected back into an image
   with a **z-buffer** (far points drawn first, near points overwrite them) and 2×2 splats to close
   small gaps.

## Data

- **Images:** 4 photos from [COCO](https://cocodataset.org) val2017 (shared with project 16), or any photo you pass in.
- **Model:** [`depth-anything/Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
  (Apache-2.0), downloaded automatically.

## Run it

```bash
.venv\Scripts\activate            # shared environment, from the repo root
cd 14-monocular-depth-3d
python depth3d.py --demo                    # the 4 COCO examples
python depth3d.py my_photo.jpg --fov 70     # your own photos (set your camera's field of view)
python view.py ../../cv-data/depth/outputs/000000000139.ply   # interactive 3D viewer
```

## What I learned / what didn't work

- **Monocular depth is relative, not metric.** The model knows the stop sign is in front of the
  trees, not that it's 4.2 m away. Scale and shift are unknown, so the clouds have the right
  *shape* but arbitrary size. Metric depth needs a metric-trained model, a known object size, or a second view.
- **The field of view matters.** A wrong FOV stretches or squashes the scene in depth. COCO
  photos don't come with camera parameters, so 60° is a guess.
- **Matplotlib 3D scatter plots were unreadable** for 200k points (my first attempt). Re-rendering
  from new camera angles with a z-buffer shows the 3D structure far more clearly.
- **Disocclusions are unavoidable** from one image: rotating the camera exposes regions that were
  never seen. Filling them would need inpainting (e.g. a diffusion model).

## Next steps

- Quantitative evaluation on NYU Depth V2 (AbsRel, δ<1.25) against ground-truth depth.
- Metric depth (Depth Anything V2 Metric or Depth Pro) and a mesh instead of points.
- Inpaint disoccluded regions for a full "3D photo" effect.
- OpenVINO export (project 19) for real-time depth from a webcam.

## Tech stack

Python · Hugging Face Transformers (Depth Anything V2) · PyTorch · Open3D · NumPy · Matplotlib
