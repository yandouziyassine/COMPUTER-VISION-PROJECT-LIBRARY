# Computer Vision, from Pixels to Foundation Models

*A course built from the 20 projects in this repository.*

This document teaches the theory behind every project, **one concept at a time, in the order a beginner
should learn it**. It is written like a class: each chapter builds on the previous ones, so read it from top to
bottom the first time. The numbers in parentheses after each heading, like **(03, 08)**, are the projects
where you can see that concept in real code; open the project folder after reading a chapter to see it in action.

**What you need before starting:** basic Python, and high-school math (functions, vectors, a little probability).
Everything else is explained here. Formulas are kept short and always come with the intuition first.

---

## Contents

**Part 1 · Images as data**
1. [What an image is](#1-what-an-image-is-all-projects)
2. [Color spaces](#2-color-spaces-01-03-20)
3. [Histograms, brightness and contrast](#3-histograms-brightness-and-contrast-03)
4. [Thresholding: turning an image into a mask](#4-thresholding-turning-an-image-into-a-mask-03)
5. [Convolution and filters](#5-convolution-and-filters-03-13)
6. [Edges](#6-edges-03)
7. [Morphology](#7-morphology-03)
8. [Contours and shapes](#8-contours-and-shapes-03)

**Part 2 · Geometry: images and the 3D world**
9. [Coordinates and transformations](#9-coordinates-and-transformations-03-08)
10. [Homography: undoing perspective](#10-homography-undoing-perspective-03-08)
11. [The pinhole camera model](#11-the-pinhole-camera-model-14-08)
12. [Depth, point clouds and novel views](#12-depth-point-clouds-and-novel-views-14)

**Part 3 · Machine learning foundations**
13. [Learning from examples](#13-learning-from-examples-all-learned-projects)
14. [Train, validation and test sets](#14-train-validation-and-test-sets-all-learned-projects)
15. [Classic machine learning: trees and gradient boosting](#15-classic-machine-learning-trees-and-gradient-boosting-15)
16. [Measuring a classifier](#16-measuring-a-classifier-02-07-10-15)
17. [Threshold-free metrics: ROC and AUROC](#17-threshold-free-metrics-roc-and-auroc-13-15)
18. [Data leakage and grouped cross-validation](#18-data-leakage-and-grouped-cross-validation-05-10-15)
19. [Unsupervised learning: clustering](#19-unsupervised-learning-clustering-20)

**Part 4 · Deep learning**
20. [Neural networks and gradient descent](#20-neural-networks-and-gradient-descent-02-11-12)
21. [Convolutional neural networks (CNNs)](#21-convolutional-neural-networks-cnns-02-05-11-12-13)
22. [Training in practice](#22-training-in-practice-02-05-11-12)
23. [Transfer learning and fine-tuning](#23-transfer-learning-and-fine-tuning-02-06-07-09-11-12-20)
24. [Looking inside: Grad-CAM](#24-looking-inside-grad-cam-02)
25. [Domain shift](#25-domain-shift-02-12-17)

**Part 5 · Finding objects: detection**
26. [The four basic vision tasks](#26-the-four-basic-vision-tasks-all-projects)
27. [Bounding boxes and IoU](#27-bounding-boxes-and-iou-06-07-09-17-20)
28. [How a one-stage detector (YOLO) works](#28-how-a-one-stage-detector-yolo-works-06-07-09-19-20)
29. [Measuring a detector: AP and mAP](#29-measuring-a-detector-ap-and-map-06-07-09-17-19-20)
30. [Hard cases: small, dense and rare objects](#30-hard-cases-small-dense-and-rare-objects-09-20-07)

**Part 6 · Pixel-level understanding: segmentation**
31. [Semantic segmentation and U-Net](#31-semantic-segmentation-and-u-net-11)
32. [Measuring masks: Dice and IoU](#32-measuring-masks-dice-and-iou-11)

**Part 7 · People and motion**
33. [Keypoints and pose estimation](#33-keypoints-and-pose-estimation-01-15)
34. [Video: frames, time and events](#34-video-frames-time-and-events-01-08-15-18)
35. [Multi-object tracking](#35-multi-object-tracking-08-20)
36. [From tracks to measurements](#36-from-tracks-to-measurements-08-20)

**Part 8 · Embeddings and similarity**
37. [Embeddings: images as vectors](#37-embeddings-images-as-vectors-10-13-16)
38. [Face recognition: detect, align, embed, compare](#38-face-recognition-detect-align-embed-compare-10)
39. [Nearest-neighbor search at scale](#39-nearest-neighbor-search-at-scale-13-16)
40. [Anomaly detection without defect examples: PatchCore](#40-anomaly-detection-without-defect-examples-patchcore-13)

**Part 9 · Reading text: OCR**
41. [The OCR pipeline](#41-the-ocr-pipeline-03-06)

**Part 10 · Foundation models**
42. [Transformers and attention for images](#42-transformers-and-attention-for-images-14-16-17-18)
43. [Contrastive learning and CLIP](#43-contrastive-learning-and-clip-12-16)
44. [Open-vocabulary detection and promptable segmentation](#44-open-vocabulary-detection-and-promptable-segmentation-17)
45. [Monocular depth estimation](#45-monocular-depth-estimation-14)
46. [Vision-language models](#46-vision-language-models-18)
47. [Auto-labeling: models that label data for other models](#47-auto-labeling-models-that-label-data-for-other-models-17)

**Part 11 · Deployment**
48. [What makes a model fast or slow](#48-what-makes-a-model-fast-or-slow-19)
49. [Model formats and runtimes: ONNX and OpenVINO](#49-model-formats-and-runtimes-onnx-and-openvino-03-06-13-19)
50. [Quantization: FP32, FP16, INT8](#50-quantization-fp32-fp16-int8-13-19)
51. [CPU, GPU and NPU](#51-cpu-gpu-and-npu-19)

**Part 12 · Doing it responsibly**
52. [Honest evaluation](#52-honest-evaluation-all-projects)
53. [Privacy, bias, consent and licenses](#53-privacy-bias-consent-and-licenses-10-15-all)

[Project → concept index](#project--concept-index) · [Glossary](#glossary)

---

# Part 1 · Images as data

## 1. What an image is (all projects)

To a computer, an image is **a grid of numbers**. Each cell of the grid is a **pixel**, and each pixel stores how
bright it is.

- A **grayscale** image is a 2D array of shape `(height, width)`. Each value is usually a whole number from
  **0 (black) to 255 (white)**: one byte, the type called `uint8`.
- A **color** image stores three numbers per pixel, one per **channel** (red, green, blue). Its shape is
  `(height, width, 3)`. A 1920×1080 photo is therefore about 6.2 million numbers.

```python
import cv2
img = cv2.imread("photo.jpg")   # numpy array, shape (H, W, 3), dtype uint8
print(img.shape, img[0, 0])      # e.g. (1080, 1920, 3) [ 34  51  60 ]  ← top-left pixel
```

Two conventions trip up every beginner:
- **Rows first.** `img[y, x]`: the first index is the row (vertical), the second the column (horizontal).
  The origin `(0, 0)` is the **top-left** corner and `y` grows **downwards**.
- **Channel order.** OpenCV loads images as **BGR**, most other libraries (PIL, PyTorch, matplotlib) use **RGB**.
  Forgetting to convert gives blue faces and orange skies.

Neural networks prefer small floating-point numbers, so images are usually converted to `float32`, divided by
255 (range 0–1) and **normalized**: subtract the dataset mean and divide by the standard deviation of each
channel. You will see `MEAN = (0.485, 0.456, 0.406)` and `STD = (0.229, 0.224, 0.225)` everywhere: these are
the statistics of ImageNet, the dataset most pretrained models learned from (**13**, **02**).

**Resolution** matters: a model sees only the pixels you give it. Shrinking a 4K frame to 640 pixels wide makes a
distant football almost disappear (**20**), and medical or satellite images often have their own fixed sizes
(**11**, **12**).

## 2. Color spaces (01, 03, 20)

RGB mixes color and brightness together: a shadow changes all three channels at once. Other **color spaces**
separate them, which makes many rules simpler.

- **Grayscale**: one channel, a weighted average `0.299 R + 0.587 G + 0.114 B` (the eye is most sensitive
  to green).
- **HSV**: **H**ue (which color, an angle around a color wheel), **S**aturation (how vivid; 0 = gray) and
  **V**alue (how bright). "White paper" in HSV is simply *high V and low S*, whatever the exact lighting:
  that is how the document scanner finds the page (**03**).
- **Lab** is designed so that distances between colors match how different they *look* to humans; useful when
  you compare colors, such as jersey colors for team assignment (**20**).

Rule of thumb: pick the color space where the thing you care about is **one simple number**.

## 3. Histograms, brightness and contrast (03)

A **histogram** counts how many pixels have each brightness value (0 to 255). It summarizes an image in 256
numbers:

- A dark photo has its histogram piled up on the left; an overexposed one on the right.
- A **low-contrast** image uses only a narrow band of values; **contrast stretching** or **histogram
  equalization** spreads them over the full range.
- Two well-separated "hills" in the histogram mean the image contains two kinds of regions (for example dark
  table and bright paper). The valley between them is a natural threshold (next chapter).

Uneven lighting (a shadow across a receipt) is a problem histograms cannot fix globally. A classic trick is to
**estimate the background** by blurring the image heavily, then **divide** the image by that estimate: the
shadow cancels out and the text stays (**03**, the lighting-correction step).

## 4. Thresholding: turning an image into a mask (03)

**Thresholding** turns a grayscale image into a **binary mask**: every pixel above a value `T` becomes 1 (white),
everything else 0 (black). Masks are how classical vision says "these pixels belong to the object".

- **Global threshold**: one `T` for the whole image. Simple, but fails when lighting varies.
- **Otsu's method** chooses `T` automatically: it tries every value and keeps the one that makes the two groups
  of pixels as internally uniform as possible (it finds the valley between the two histogram hills).
- **Adaptive (local) threshold**: compute a different `T` for every pixel from its neighborhood (for example
  "brighter than the local average by 5"). This survives gradients and soft shadows, which is why the scanner
  combines several global *and* adaptive thresholds and lets later steps pick the best candidate (**03**).

## 5. Convolution and filters (03, 13)

**Convolution** is the single most important operation in computer vision, both classical and deep.

Take a small grid of weights called a **kernel** (say 3×3). Slide it over the image; at each position, multiply
the kernel by the pixels under it and add everything up. The sum becomes the output pixel.

```
kernel (box blur)        image patch          output pixel
 1/9 1/9 1/9            10 10 10
 1/9 1/9 1/9     ×      10 90 10      →      (8·10 + 90)/9 ≈ 18.9
 1/9 1/9 1/9            10 10 10
```

Different kernels do different jobs:
- **Box / Gaussian blur**: averaging kernels. A **Gaussian** kernel weights the center more, with a width set
  by σ (sigma). Blurring removes noise and fine detail. PatchCore blurs its anomaly map with a Gaussian of σ = 4
  so that the heatmap is smooth (**13**).
- **Sharpening**: center weight positive, neighbors negative; it exaggerates differences.
- **Edge detectors**: see the next chapter.

Two details you will meet in code: **padding** (what to assume beyond the image border: zeros, or a mirror
"reflect" of the image) and **stride** (how many pixels the kernel jumps each step; stride 2 halves the output
size).

Keep convolution in mind: in Part 4 we will let a neural network **learn** its own kernels instead of designing
them by hand.

## 6. Edges (03)

An **edge** is where brightness changes sharply: the border of an object, a line of text, a shadow.

- The **Sobel** kernels estimate the horizontal and vertical **gradient** (rate of change of brightness). Its
  magnitude is large on edges.
- **Canny** is the classic full edge detector: smooth, compute gradients, keep only the thin ridge of maximum
  gradient (*non-maximum suppression*), then keep strong edges plus the weak edges connected to them
  (*hysteresis*). The result is a clean one-pixel-wide edge map.

Edges are precise but fragile: a shadow or a textured table creates as many edges as the object you want. The
document scanner uses Canny edges only as a **fallback**, after color-based masks (**03**).

## 7. Morphology (03)

**Morphological operations** clean up binary masks by sliding a small shape (the *structuring element*) over
them:

- **Erosion** shrinks white regions (a pixel stays white only if its whole neighborhood is white): removes
  specks of noise.
- **Dilation** grows white regions: fills small holes and gaps.
- **Opening** = erosion then dilation: removes small white noise, keeps big shapes.
- **Closing** = dilation then erosion: fills small black holes, such as text inside a white page mask.

## 8. Contours and shapes (03)

Once you have a clean mask, you want the **outline** of each white region: its **contour**, a list of boundary
points. From a contour you can measure:

- **Area** and **perimeter**.
- The **convex hull**: the smallest convex shape containing it (like a rubber band stretched around it).
- A **polygon approximation** (Douglas–Peucker): simplify the outline to a few corners. A sheet of paper should
  simplify to **4 corners**.
- **Shape checks** that real pages satisfy: the blob fills its quadrilateral almost completely, corner angles
  are near 90° (the scanner accepts 55–125° to allow perspective), it is not a thin sliver, and it does not touch
  the image border (**03**).

This is the heart of classical vision: **hand-written rules about shapes**. They are fast and explainable, but
every rule is an assumption that some photo will break (snow behind a white receipt, in **03**). Learned
models (Part 4) replace rules with examples.

---

# Part 2 · Geometry: images and the 3D world

## 9. Coordinates and transformations (03, 08)

Many operations move pixels around. Writing a pixel as a point `(x, y)`, the common transformations are:

| Transformation | What it can do | Degrees of freedom | Points needed to define it |
|---|---|---|---|
| Translation | shift | 2 | 1 |
| Rigid / similarity | shift, rotate (+ uniform scale) | 3 / 4 | 2 |
| Affine | + shear, non-uniform scale; parallel lines stay parallel | 6 | 3 |
| **Perspective (homography)** | anything a camera viewing a flat surface can do; parallel lines may meet | **8** | **4** |

They are written as matrices acting on **homogeneous coordinates** `(x, y, 1)`: adding a 1 lets a single matrix
multiplication express shifts as well as rotations. Face alignment uses a similarity transform from 5 facial
landmarks (**10**); the scanner and the traffic project use full homographies (**03**, **08**).

## 10. Homography: undoing perspective (03, 08)

Photograph a flat sheet of paper at an angle and its rectangle becomes a general quadrilateral. A **homography**
is the 3×3 matrix `H` that maps one plane to another:

```
[x']       [x]
[y'] ~ H · [y]      then divide by the third coordinate: (x'/w, y'/w)
[w ]       [1]
```

It has 8 unknowns, so **4 point correspondences** determine it (`cv2.getPerspectiveTransform`). Two uses in this
repo:

- **Document scanning (03)**: map the 4 detected page corners to the corners of a straight rectangle and warp
  the image: the page now looks scanned from directly above.
- **Measuring speed (08)**: the road is (almost) a flat plane. Pick 4 points on the road in the video and say
  where they are in **meters** (a 25 m × 250 m rectangle of lanes). The homography then converts any pixel on
  the road into real-world meters, and distance ÷ time gives speed.

The key assumption is **flatness**: a homography is exact only for points on that plane. That is why the traffic
project tracks each car's **bottom-center** point (where it touches the road), not its roof.

## 11. The pinhole camera model (14, 08)

How does a 3D point in the world end up as a pixel? The **pinhole model** is the standard answer. Light from a
point `(X, Y, Z)` (in camera coordinates, `Z` = distance in front of the camera) passes through a single point
and lands on the sensor at:

```
u = fx · X / Z + cx
v = fy · Y / Z + cy
```

- `fx, fy` are the **focal length** in pixels; `cx, cy` the **principal point** (usually the image center).
  Together they form the **intrinsic matrix K**.
- Dividing by `Z` is **perspective**: far objects look smaller. It is also why a single photo loses depth:
  every point along the same ray from the camera lands on the same pixel.

The camera's position and orientation in the world (rotation `R` and translation `t`) are the **extrinsics**.
Calibrating a real camera means measuring `K` (and lens distortion), usually with a checkerboard. When you don't
know `K`, a reasonable guess comes from the field of view: `f = (W / 2) / tan(FOV / 2)` (**14**).

## 12. Depth, point clouds and novel views (14)

A **depth map** is an image where each pixel stores a distance instead of a color. If you have one, you can
run the pinhole equations **backwards** (*back-projection*):

```
X = (u − cx) · Z / fx
Y = (v − cy) · Z / fy
```

Doing this for every pixel produces a **point cloud**: a set of 3D points, each with the color of its pixel.
To show it from a **new viewpoint**, rotate/translate the points (new extrinsics) and project them again with the
pinhole model. When several points land on the same pixel, keep the nearest one: that is a **z-buffer**, the same
visibility trick every 3D game uses. Regions hidden behind objects in the original photo have no points, so they
appear as holes (*disocclusions*): a single photo cannot show what it never saw (**14**).

Where does the depth map come from with only one photo? A neural network (chapter 45) that has learned what
depth usually looks like.

---

# Part 3 · Machine learning foundations

## 13. Learning from examples (all learned projects)

Writing rules by hand (Part 1) breaks on the cases you didn't imagine. **Machine learning** replaces the rules
with **examples**: you show a model many inputs with the correct answer (**labels**), and an algorithm adjusts the
model's parameters until its answers match.

Vocabulary:
- **Features**: the numbers the model receives (pixels, or measurements computed from them, like the body angle
  of a person in **15**).
- **Label / target**: the answer to learn ("fall" / "no fall", a class name, a box).
- **Classification**: predict a category (**02**, **12**). **Regression**: predict a number (depth in **14**,
  box coordinates in detection).
- **Supervised learning**: learning from labeled examples. **Unsupervised**: finding structure without labels
  (chapters 19 and 40). **Self-supervised**: inventing labels from the data itself (chapter 43).
- **Loss function**: a number measuring how wrong the model is on an example. Training = **minimizing the
  loss**.
- **Overfitting**: the model memorizes the training examples (including their noise) and fails on new ones.
  **Underfitting**: the model is too simple to capture the pattern at all.

The whole game of machine learning is **generalization**: doing well on data the model has never seen.

## 14. Train, validation and test sets (all learned projects)

To measure generalization honestly, split your data into three parts **before** doing anything:

- **Training set**: the model learns from it.
- **Validation set**: you use it to make decisions: which model, how many epochs, which threshold.
- **Test set**: touched **once**, at the end, to report the final number.

Why three? Every decision you make by looking at a set leaks information about it into your model. If you tune
the threshold on the test set, your test score is optimistic. This repo is explicit about such cases: PatchCore's
F1 uses a threshold picked on the test set, and the write-up says so; AUROC (chapter 17) does not depend on that
choice (**13**).

**Cross-validation** (k-fold) is the alternative when data is scarce: split into k parts, train on k−1, test on the
remaining one, repeat k times and average. LFW face verification uses an official 10-fold protocol (**10**),
fall detection uses grouped 5-fold cross-validation (**15**, chapter 18).

## 15. Classic machine learning: trees and gradient boosting (15)

Before deep learning, and still today for **tabular** data (rows of a few dozen features), the strongest models
are tree ensembles.

- A **decision tree** asks a sequence of yes/no questions on features ("is the torso angle > 60°?", "did the head
  drop fast?") and ends in a leaf with a prediction. Easy to read, but a single tree overfits.
- **Gradient boosting** builds **hundreds of small trees one after another**, each new tree trained to correct
  the errors (the *residuals*, more precisely the gradient of the loss) of all the previous ones. The final
  prediction is the sum.
- **Histogram-based gradient boosting** (`HistGradientBoostingClassifier`) first bins every feature into ~256
  buckets, which makes training very fast.

Fall detection turns each frame into posture and motion features from the skeleton (torso angle, box aspect ratio, head and hip height, how fast they move down), then classifies
them with gradient boosting, and compares it with a hand-written rule baseline (**15**). This "deep model for
perception, simple model for the decision" split is common in practice: it is fast, needs little data and stays
explainable.

## 16. Measuring a classifier (02, 07, 10, 15)

**Accuracy** (fraction of correct answers) is misleading when classes are unbalanced: if 1 % of frames contain a
fall, a model that always says "no fall" is 99 % accurate and useless. Use the **confusion matrix** instead:

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | True positive (TP) | False negative (FN), a *miss* |
| **Actually negative** | False positive (FP), a *false alarm* | True negative (TN) |

- **Precision** = TP / (TP + FP): when the model raises an alarm, how often is it right?
- **Recall** (sensitivity) = TP / (TP + FN): of all real cases, how many did it catch?
- **F1** = 2 · P · R / (P + R): the harmonic mean, high only if both are high.
- **Macro-F1**: compute F1 per class and average, so rare classes count as much as common ones (**02**).

Precision and recall trade off through the **decision threshold**: lower it and you catch more (recall ↑) but
raise more false alarms (precision ↓). Which side matters depends on the application, and good projects report
the metric the *user* feels. The hard-hat monitor reports "90.8 % of violations flagged at 87.8 % precision",
the alert a safety manager actually receives, on top of detector metrics (**07**). Face verification reports
**TAR @ FAR** (true accept rate at a fixed false accept rate) because a security system first fixes how many
impostors it can tolerate (**10**).

## 17. Threshold-free metrics: ROC and AUROC (13, 15)

Many models output a **score**, and you choose the threshold later. To compare models independently of that
choice, sweep the threshold over all values:

- The **ROC curve** plots the true positive rate (recall) against the false positive rate (FP / (FP + TN)).
- **AUROC** (area under the ROC curve) summarizes it in one number: the probability that a random positive
  example gets a higher score than a random negative one. **1.0** = perfect ranking, **0.5** = random guessing.

PatchCore reaches image AUROC **1.000 / 0.999 / 0.977** on three MVTec AD categories: it ranks defective parts
above good ones almost perfectly, before any threshold is chosen (**13**). When positives are rare, the
**precision–recall curve** and its area (average precision) are more informative; they reappear in detection
(chapter 29).

## 18. Data leakage and grouped cross-validation (05, 10, 15)

**Data leakage** is when information from the test set sneaks into training, making results look better than they
will be in reality. The most common form in vision is **near-duplicates across the split**:

- Consecutive video frames are almost identical. If frame 100 of a fall is in training and frame 101 in testing,
  the model is tested on something it has essentially seen.
- Several photos of the same person, the same patient or the same scene behave the same way.

The fix is to split by **group** (video, person, patient), not by image. Fall detection uses
**StratifiedGroupKFold**: each video sequence lives entirely in one fold, and every fold keeps a similar share of
falls (**15**). LFW's official protocol likewise keeps identities separated between folds (**10**). A random
seed fixed for every split makes results reproducible (**11**).

When the groups are not recorded, you cannot split by them, and the test score becomes optimistic. Sign Language
MNIST was generated from only 1,704 photos expanded into 34,627 images; a third of its test images are near
copies of training images, and even a linear model on raw pixels scores ~99 % on those. A model can reach 100 %
there and still fail on a new person's hand, so the project measures robustness instead (**05**).

## 19. Unsupervised learning: clustering (20)

Sometimes you have no labels at all. **Clustering** groups similar examples automatically.

**k-means** with k clusters:
1. Pick k starting centers.
2. Assign every point to its nearest center.
3. Move each center to the mean of its points.
4. Repeat 2–3 until nothing changes.

Football analytics never sees a "team" label: it takes the jersey color of each detected player and runs k-means
with **k = 2**; the two clusters are the two teams. Players far from both centers (referees, goalkeepers) are set
aside (**20**).

How do you know the clusters are real? The **silhouette score** compares, for each point, the distance to its own
cluster with the distance to the nearest other cluster: from −1 (wrong cluster) to +1 (clearly separated). The
team split reaches **0.80** (**20**).

---

# Part 4 · Deep learning

## 20. Neural networks and gradient descent (02, 11, 12)

A **neuron** computes a weighted sum of its inputs plus a bias, then applies a non-linear **activation function**
(most commonly **ReLU**: `max(0, x)`). A **layer** is many neurons side by side; a **deep network** stacks many
layers. Without the non-linearity, any stack of layers would collapse into a single linear function; with it,
deep networks can approximate almost any function.

Training repeats four steps on small **batches** of examples:
1. **Forward pass**: compute the predictions.
2. **Loss**: measure the error. For classification, **cross-entropy**: `−log(probability given to the correct
   class)`. Confident and right → near 0; confident and wrong → very large.
3. **Backward pass (backpropagation)**: compute the **gradient** of the loss with respect to every weight, i.e.
   which direction each weight should move to reduce the loss. This is the chain rule of calculus, applied
   automatically by PyTorch.
4. **Update**: move each weight a small step against its gradient: `w ← w − learning_rate · gradient`. This is
   **gradient descent**; practical variants like **Adam** and **AdamW** adapt the step size per weight.

One pass over the whole training set is an **epoch**. The final layer outputs **logits** (raw scores), turned
into probabilities that sum to 1 by the **softmax** function.

## 21. Convolutional neural networks (CNNs) (02, 05, 11, 12, 13)

A fully connected layer on a 224×224×3 image would need 150,528 weights *per neuron* and would not know that
nearby pixels are related. **Convolutional layers** fix both: each layer learns a set of small kernels
(chapter 5) and slides them over the whole image.

- **Weight sharing**: the same kernel is applied everywhere, so a "vertical edge" detector learned in one corner
  works in all corners.
- **Feature maps**: each kernel produces one output channel. A layer with 64 kernels turns an image into 64
  feature maps.
- **Pooling / stride**: halve the resolution so the next layers see a larger area. The **receptive field**, the
  part of the input that influences one output value, grows with depth.
- **Hierarchy**: early layers learn edges and colors, middle layers textures and parts, late layers whole
  objects. PatchCore deliberately uses **middle layers** (layers 2 and 3 of a WideResNet-50): local enough to
  point at a small scratch, abstract enough to ignore pixel noise (**13**).

A small CNN trained **from scratch** is enough for tiny images: three conv blocks (28 → 14 → 7 → 3 pixels) read
28×28 hand signs with ~0.4 M parameters (**05**).

Famous architectures you will meet:
- **ResNet** adds **skip connections** (each block outputs `input + f(input)`), which let gradients flow through
  very deep networks. ResNet-18 is used in **11** and **12**; WideResNet-50 in **13**.
- **EfficientNet** scales depth, width and resolution together for a good accuracy/compute balance (**02**).
- **Batch normalization** layers re-center and re-scale activations during training, which stabilizes it.

## 22. Training in practice (02, 05, 11, 12)

Getting a network to train well is a craft. The main tools:

- **Data augmentation**: random flips, crops, rotations, color jitter applied at training time. The model sees
  "new" images every epoch and learns that a polyp is still a polyp when mirrored (**11**). Augmentations must
  be label-preserving: flipping a satellite tile is fine (**12**); flipping text is not.
- **Learning rate** is the most important setting: too high and training diverges, too low and it crawls. A
  **schedule** (warm-up, then cosine decay) usually helps.
- **Batch size**: larger batches give smoother gradients but need more memory; on a laptop CPU, small batches
  are the norm.
- **Early stopping / checkpoint selection**: watch the validation metric each epoch and keep the best weights,
  not the last.
- **Class imbalance**: weight the loss or resample so rare classes are not ignored.

Augmentation is a **specification** of what the model should ignore, and it protects only against what it
includes. In the sign-language project, training with rotations and shifts doubled accuracy on a 30° tilt, but
the augmented model became *worse* under blur, which it had never seen (**05**).

On a CPU, every one of these choices also has a **time** cost, which is why this repo leans on the next idea,
transfer learning.

## 23. Transfer learning and fine-tuning (02, 06, 07, 09, 11, 12, 20)

Training a CNN from scratch needs millions of images. But a network trained on **ImageNet** (1.2 million photos,
1000 classes) has already learned generic features (edges, textures, parts) that are useful for almost any
image. **Transfer learning** reuses them:

1. Take a **pretrained** network (the *backbone*).
2. Replace its last layer with a new **head** for your classes.
3. **Fine-tune**: train on your small dataset, often first only the head, then the whole network with a small
   learning rate so the pretrained features are adjusted, not destroyed.

This is how a solar-panel classifier reaches 82.9 % with only 869 images in total (**02**), how ResNet-18 reaches 98 % on
EuroSAT (**12**), and how every detector in this repo starts from COCO-pretrained YOLO weights: the shelf
monitor was fine-tuned with only **219** images (**09**). PatchCore pushes the idea to the limit: it does **no
training at all**, it only reads features from the pretrained backbone (**13**).

## 24. Looking inside: Grad-CAM (02)

A classifier that says "dusty" is more trustworthy if you can see *where* it looked. **Grad-CAM** produces a
heatmap of the evidence:

1. Take the feature maps of the last convolutional layer (each one says "my pattern is here").
2. Compute the gradient of the chosen class score with respect to each map: how much each pattern pushes the
   decision.
3. Average each gradient map into one weight, form the weighted sum of the feature maps, keep the positive part
   (ReLU) and upsample to the image size.

Explanations are also a **debugging tool**: Grad-CAM revealed that the solar classifier partly looked at the
**roof around the panel**, not only at the dust (**02**), a shortcut that would fail on a different roof.

## 25. Domain shift (02, 12, 17)

A model learns the statistics of its training data. When the data it meets in use looks different, performance
drops: that is **domain shift**. Different camera, lighting, country, season, or viewpoint all count.

The clearest example in this repo: CLIP, which beat a supervised detector in zero-shot labeling of everyday
photos (**17**), reaches only **47.4 %** on 64×64-pixel satellite tiles, while a small ResNet fine-tuned on those
tiles reaches **98.1 %** (**12**). Satellite images seen from above at 10 m per pixel are far from the internet
photos CLIP learned from. Practical consequences:
- Always evaluate on data that looks like the deployment data.
- A small, fine-tuned specialist often beats a huge generalist **in its domain**.

---

# Part 5 · Finding objects: detection

## 26. The four basic vision tasks (all projects)

| Task | Output | Example in this repo |
|---|---|---|
| **Classification** | one label for the whole image | solar panel condition (**02**), land cover (**12**) |
| **Object detection** | a box + label (+ confidence) per object | plates (**06**), hard hats (**07**), vehicles (**08**), products (**09**), players and ball (**20**) |
| **Segmentation** | a label for every pixel | polyps (**11**), SAM masks (**17**) |
| **Keypoints / pose** | named points per object | hand landmarks (**01**), body skeletons (**15**) |

Real systems **chain** these tasks: detect a plate then read it (**06**), detect people then track them (**08**),
detect a face then embed it (**10**), detect a person then estimate their pose (**15**).

## 27. Bounding boxes and IoU (06, 07, 09, 17, 20)

A **bounding box** is the smallest axis-aligned rectangle around an object, stored as `(x1, y1, x2, y2)` corners,
or as `(center x, center y, width, height)`. YOLO label files use the second form, **normalized** to 0–1 by the
image size, one line per object: `class cx cy w h`.

To compare a predicted box with the true one, use **Intersection over Union**:

```
IoU = area of overlap / area of union        (0 = no overlap, 1 = identical)
```

A prediction usually counts as correct if IoU ≥ 0.5 with a true box of the same class. IoU is also the glue of
tracking (chapter 35) and of non-maximum suppression (next chapter).

## 28. How a one-stage detector (YOLO) works (06, 07, 09, 19, 20)

**YOLO** ("You Only Look Once") detects all objects in a single forward pass:

1. A CNN **backbone** extracts feature maps at several scales (for example 1/8, 1/16 and 1/32 of the input
   size), so small objects are found on the fine maps and large ones on the coarse maps. A **neck** mixes
   information between scales.
2. A **head** predicts, at **every location** of every feature map, a box (distances to its four sides) and a
   score per class. Modern YOLO versions (like YOLO11 used here) are **anchor-free**: no predefined box shapes.
3. Thousands of candidate boxes come out; most overlap. **Non-maximum suppression (NMS)** keeps the
   highest-scoring box and deletes the others that overlap it with IoU above a threshold, then repeats.
4. A **confidence threshold** drops weak boxes.

The **"n"** in YOLO11**n** means *nano*: the smallest variant (~2.6 M parameters), the right size for a laptop
CPU or an NPU (**19**). Training loss combines a box-regression part, a classification part, and a term that
sharpens box edges.

Two-stage detectors (the R-CNN family) first propose regions, then classify each; they are slower but were long
more accurate. Transformer detectors (DETR, and Grounding DINO in chapter 44) predict a fixed set of boxes
directly and need no NMS.

## 29. Measuring a detector: AP and mAP (06, 07, 09, 17, 19, 20)

Detection mixes two questions: *did you find the objects* and *are your boxes accurate*. The standard metric:

1. Sort all predictions of a class by confidence.
2. Walk down the list: each prediction is a TP if it matches an unmatched true box with IoU ≥ threshold,
   otherwise an FP. Compute precision and recall after each one.
3. **Average precision (AP)** = area under this precision–recall curve.
4. **mAP** = mean of AP over classes.

- **mAP50** uses IoU ≥ 0.5: "found it, roughly in the right place".
- **mAP50-95** averages over IoU thresholds 0.50, 0.55, … 0.95 (the COCO metric): rewards precise boxes.

Results in this repo: plates mAP50 **0.983** (**06**), hard hats **0.828** (**07**), shelf products **0.838**
(**09**), players AP50 **0.980** but ball only **0.438** (**20**). The official **pycocotools** implementation is
used when comparing models fairly, as in the zero-shot vs supervised comparison (**17**) and the INT8 accuracy
check (**19**).

## 30. Hard cases: small, dense and rare objects (09, 20, 07)

- **Small objects**: a football can be 10 pixels wide in a broadcast frame; after resizing to the model's input
  size it is a few pixels, and the coarse feature maps cannot see it. Remedies: larger input size, tiling the
  image into crops, higher-resolution feature maps. Hence ball AP50 0.44 vs players 0.98 (**20**).
- **Dense scenes**: a shelf photo holds 100–700 products. Default limits (maximum detections per image, NMS IoU)
  must be raised, and overlapping neighbors get suppressed by mistake (**09**).
- **Counting** from detections: count = number of boxes above a threshold. Report it with its own error metric
  (**mean absolute error**: 22.1 products, 13 % on shelves in **09**), not just mAP.
- **Rare classes and class balance**: if one class (heads *without* hard hats) is rare, the detector sees few
  examples; per-class AP and recall reveal the weakness that the mean hides (**07**).

---

# Part 6 · Pixel-level understanding: segmentation

## 31. Semantic segmentation and U-Net (11)

**Semantic segmentation** assigns a class to **every pixel**. The output is a mask the size of the image.

The classic architecture for it is **U-Net**, shaped like a U:
- The **encoder** (left side) is a normal CNN (here a pretrained ResNet-18): it shrinks the image step by step
  while building more abstract features ("this region looks like a polyp").
- The **decoder** (right side) upsamples back to full resolution.
- **Skip connections** copy the encoder's fine-resolution feature maps to the decoder at the same scale. Deep
  layers know *what*, shallow layers know precisely *where*; combining them gives sharp boundaries.

The network outputs, for each pixel, the probability of "polyp". A threshold (typically 0.5) gives the final mask (**11**).
Related tasks: **instance segmentation** separates individual objects (each SAM mask in **17** is one instance);
**panoptic** segmentation does both.

## 32. Measuring masks: Dice and IoU (11)

Pixel accuracy is misleading: if the polyp covers 5 % of the image, predicting "no polyp" everywhere is 95 %
accurate. Mask overlap metrics fix this:

```
IoU  = |A ∩ B| / |A ∪ B|
Dice = 2 |A ∩ B| / (|A| + |B|)
```

where A is the predicted mask and B the true one. Dice is always ≥ IoU and weighs the overlap more generously. The
polyp U-Net reaches **Dice 0.897, IoU 0.833** (**11**). **Dice loss** (1 − Dice) is also a popular training loss for
segmentation precisely because it ignores the huge, easy background.

---

# Part 7 · People and motion

## 33. Keypoints and pose estimation (01, 15)

**Keypoint detection** predicts the position of specific named points: the 21 joints of a hand, or the 17 body
joints of the COCO skeleton (nose, shoulders, elbows, wrists, hips, knees, ankles…). A **pose** is the set of
keypoints of one person.

- **Top-down** methods detect each person first, then find keypoints inside each box. **Bottom-up** methods find
  all keypoints first and group them into people. YOLO11n-**pose** predicts box and keypoints together in one
  pass (**15**).
- MediaPipe Hands tracks **21 hand landmarks** per frame in real time on a CPU (**01**).

Keypoints are a powerful **compression**: millions of pixels become ~50 numbers that describe the body,
independent of clothing, background and lighting. Features built on them are easy to reason about: the virtual
painter checks which fingers are raised to switch between drawing and erasing (**01**); fall detection computes
torso angle, hip height, aspect ratio and their speeds (**15**). Normalize them (by body size, by frame size) so
that a person far from the camera looks like one close to it.

## 34. Video: frames, time and events (01, 08, 15, 18)

A video is a sequence of images (**frames**) shown at a fixed rate (**fps**, frames per second; typically 25–30).
Time adds three new ideas:

- **Temporal smoothing**: single-frame predictions are noisy. A rolling average of landmark positions stops the
  painter's brush from jittering (**01**).
- **Motion features**: speed and acceleration from differences between frames. A fall is not "lying down", it is
  "going down **fast**" (**15**). Measured in pixels per frame, they depend on fps, so convert to seconds using
  the real frame rate, a bug fixed in **15**.
- **Events and state machines**: an alarm should fire once per fall, not on every frame. Require the condition
  for N consecutive frames, then hold a "fallen" state; measure the **detection delay** (~0.7 s in **15**). The
  painter's gesture logic is a small state machine (idle / draw / erase / select color) for the same reason (**01**).

Video models can also process time directly (3D convolutions, video transformers), or sample a few frames and
reason over them, as the video Q&A assistant does (**18**).

## 35. Multi-object tracking (08, 20)

Detection answers "where are the cars in this frame". **Tracking** answers "which car is which across frames":
each object gets a persistent **ID**. Without IDs you cannot count (each car would be counted in every frame) or
measure speed.

**Tracking-by-detection**, the standard approach:
1. Run the detector on each frame.
2. **Predict** where each existing track should be now. A **Kalman filter** keeps a position and velocity per
   track and predicts the next position, then corrects it with the new measurement, smoothing noisy boxes.
3. **Associate** predicted tracks with new detections: build a cost matrix (1 − IoU) and solve the best
   one-to-one matching (the **Hungarian algorithm**).
4. Unmatched detections start new tracks; tracks unmatched for too long are deleted.

**ByteTrack** adds one idea: also try to match **low-confidence** detections to existing tracks in a second
round, instead of throwing them away. A car half hidden behind a truck gets a weak score, but it is still the same
car, so its track survives the occlusion (**08**).

## 36. From tracks to measurements (08, 20)

Once objects have stable IDs over time, analytics become simple geometry:

- **Line counting**: a vehicle is counted when its track's anchor point crosses a virtual line, once, in a given
  direction (**08**).
- **Speed**: map the bottom-center point to meters with the road homography (chapter 10), then speed =
  distance over a window of frames ÷ elapsed time. A median of 119 km/h on a 130 km/h motorway is a
  plausibility check when no ground truth exists (**08**).
- **Heatmaps**: accumulate track positions on a grid and blur it: where do vehicles drive, where do players run
  (**08**).
- **Possession**: the team of the player nearest the ball, with a distance limit and some smoothing over time
  (**20**).

Each step adds its own error on top of the detector's, so the projects measure each stage separately wherever
ground truth exists.

---

# Part 8 · Embeddings and similarity

## 37. Embeddings: images as vectors (10, 13, 16)

An **embedding** is a vector of numbers (for example 128 or 512 of them) produced by a network such that
**similar inputs get nearby vectors**. Once images are vectors, many problems become geometry:

- **Are these two faces the same person?** → are their vectors close? (**10**)
- **Which photos match this description?** → which vectors are nearest to the text's vector? (**16**)
- **Is this patch abnormal?** → is it far from every normal patch's vector? (**13**)

Distances:
- **Euclidean (L2)** distance: straight-line distance.
- **Cosine similarity**: the cosine of the angle between two vectors (1 = same direction, 0 = unrelated).
  If vectors are **L2-normalized** to length 1, ranking by cosine similarity and by Euclidean distance give the
  same order.

A network trained for classification already produces useful embeddings (its second-to-last layer). Networks
trained specifically for embeddings use losses that pull matching pairs together and push others apart
(**metric learning**, and **contrastive learning** in chapter 43).

## 38. Face recognition: detect, align, embed, compare (10)

Face recognition is a four-stage pipeline:

1. **Detect** faces (YuNet, a tiny CNN detector that also outputs 5 landmarks: eyes, nose, mouth corners).
2. **Align**: rotate and scale each face with a similarity transform (chapter 9) so the eyes are always at the
   same position. This removes pose variation the embedding model would otherwise have to learn.
3. **Embed**: SFace turns the aligned 112×112 face into a 128-number vector.
4. **Compare**: cosine similarity between two embeddings; above a threshold → same person.

Two different problems use this pipeline:
- **Verification (1:1)**: "is this the person they claim to be?" Measured on LFW with the 10-fold protocol:
  **99.35 %** (**10**).
- **Identification (1:N)**: "who is this, among N enrolled people?" Errors grow with N, because each extra
  person is another chance of a false match. The attendance tool only checks in people who **consented** and
  enrolled (**10**; see chapter 53).

The threshold is the policy. On LFW it is chosen on 9 folds and tested on the 10th, so the accuracy is never
measured on the data that set it; **TAR @ FAR** then reports how many genuine pairs are accepted when only 1 % or
0.1 % of impostors get through (chapter 16) (**10**).

## 39. Nearest-neighbor search at scale (13, 16)

Given a query vector, find the most similar vectors in a collection. The exact method compares the query with
**every** vector: fine for thousands, too slow for millions.

- **FAISS** (Facebook AI Similarity Search) does exact search with highly optimized matrix multiplication, and
  **approximate nearest neighbor (ANN)** search for big collections: indexes such as IVF (cluster the vectors,
  search only the closest clusters) or HNSW (a graph to hop through) trade a little recall for huge speed-ups.
- **Recall@K** measures retrieval quality: the fraction of queries whose correct answer appears in the top K
  results. Text-to-image search on 5,000 COCO images reaches **Recall@10 = 75.2 %** (**16**).
- The pattern **embed once, search many times** makes search instant: the expensive part (encoding every image)
  happens offline (**16**).

The same nearest-neighbor computation is the core of PatchCore (next chapter), which uses the identity
`‖q − b‖² = ‖q‖² + ‖b‖² − 2 q·b` to turn it into one big matrix multiplication, fast on any hardware (**13**).

## 40. Anomaly detection without defect examples: PatchCore (13)

In a factory, defects are rare and varied: you cannot collect thousands of examples of every possible scratch.
**Anomaly detection** learns only what **normal** looks like and flags everything else. PatchCore (Roth et al.,
CVPR 2022), reimplemented from scratch in project 13, does it in four steps:

1. **Patch features**: pass each defect-free training image through a pretrained WideResNet-50 and take the
   layer-2 and layer-3 feature maps (chapter 21). Average each position with its 3×3 neighborhood (*locally
   aware* features), upsample layer 3 to layer 2's size and concatenate: every image becomes 28 × 28 = 784 patch
   vectors of 1536 numbers.
2. **Memory bank**: store the patch vectors of all good images.
3. **Coreset subsampling**: the bank is huge and redundant. **Greedy coreset selection** keeps 10 % of the
   patches: repeatedly add the patch that is **farthest** from everything already selected (farthest-point
   sampling, done on a cheap random projection), so the subset still covers every kind of normal patch.
4. **Scoring**: for a test image, each patch's anomaly score is the distance to its **nearest neighbor** in the
   bank. The 28×28 score map, upsampled and smoothed with a Gaussian, is the **heatmap**; the **image score** is
   its maximum.

There is **no training loop and no defect image**, yet image AUROC reaches **1.000 / 0.999 / 0.977** (bottle /
metal nut / grid), within about one point of the paper (**13**). The method's assumption is also its limit: it
models one product at a time, so a photo of another object looks anomalous everywhere.

---

# Part 9 · Reading text: OCR

## 41. The OCR pipeline (03, 06)

**Optical character recognition** turns text in an image into a string. Modern OCR (like RapidOCR, which runs the
PaddleOCR models) has three stages:

1. **Text detection**: a segmentation-style network predicts which pixels belong to text and groups them into
   boxes around words or lines.
2. **(Angle classification)**: decide whether a box is upside down.
3. **Text recognition**: a network reads each cropped box as a sequence: a CNN turns the crop into a sequence of
   feature columns, and a sequence model predicts one character (or "blank") per column. **CTC decoding**
   (Connectionist Temporal Classification) merges repeated characters and removes blanks: `hh-e-ll-ll-oo` →
   `hello`. That is how a model reads text of any length without knowing where each letter is.

OCR quality depends heavily on what comes **before and after** the model:
- **Before**: a flat, well-lit, correctly cropped image. The document scanner exists to provide one (**03**).
  On tight crops of license plates, the text detector split plates letter by letter; feeding it the crop with a
  margin fixed it and took exact plate reading from **0 % to ~90 %** (**06**).
- **After**: **post-processing with domain knowledge**. Plates follow known formats, so letters and digits that
  look alike (O/0, I/1, S/5) can be corrected by position (**06**). Receipts are read by grouping words into
  lines (deskewed along the dominant text slope) and applying rules for dates and totals; the line grouping did
  more for accuracy than the vision part (**03**).

---

# Part 10 · Foundation models

## 42. Transformers and attention for images (14, 16, 17, 18)

**Transformers** began in language and now dominate vision's largest models. The **Vision Transformer (ViT)**:

1. Cuts the image into **patches** (for example 32×32 pixels in CLIP's ViT-B/32 in **16**).
2. Turns each patch into a vector (**token**) and adds a **position embedding** so the model knows where each
   patch was.
3. Passes the tokens through layers of **self-attention**: every token computes how relevant every other token
   is to it (attention weights from query·key products), then takes a weighted mix of their values. A patch of
   a dog's ear can directly "look at" the patch with its tail, however far apart they are, unlike a CNN whose
   receptive field grows slowly.

Transformers have fewer built-in assumptions than CNNs, so they need **much more training data**, but they scale
very well with data and size, and the same architecture handles images, text and both together. That is what
makes the **foundation models** below possible: models pretrained on enormous datasets that you use
**zero-shot** (no training at all) or with light fine-tuning.

## 43. Contrastive learning and CLIP (12, 16)

**CLIP** learns from hundreds of millions of image–caption pairs from the internet, without class labels. It has
two encoders, one for images and one for text, trained with a **contrastive** objective: in a batch of N pairs,
make each image's embedding close to **its own** caption's embedding and far from the N−1 others. After training,
images and text live in the **same embedding space**.

That shared space enables:
- **Text-to-image search**: embed the query "a dog catching a frisbee" and find the nearest image embeddings
  (**16**).
- **Zero-shot classification**: embed one sentence per class ("a satellite photo of a forest", "…of a
  river"…), and assign each image the class whose sentence is closest. No training images needed (**12**).

The catch is domain shift (chapter 25): zero-shot CLIP manages 47.4 % on EuroSAT satellite tiles where a small
fine-tuned CNN gets 98.1 % (**12**). Prompt wording also matters, so zero-shot results are always measured,
never assumed.

## 44. Open-vocabulary detection and promptable segmentation (17)

Classical detectors know a fixed list of classes. **Open-vocabulary** models take the class names **as text at
run time**:

- **Grounding DINO** is a transformer detector that fuses text features and image features, and outputs boxes
  for each phrase in the prompt ("ski pole . helmet . goggles"). A box threshold and a text threshold decide what
  counts as found.
- **SAM** (Segment Anything Model) is **promptable segmentation**: give it a point or a box, and it returns the
  mask of the object there. A heavy image encoder runs once per image; a light prompt encoder and mask decoder run
  once per prompt, so many masks per image are cheap.

Chaining them (text → boxes → masks) labels objects nobody trained it on. On 10 COCO classes, this zero-shot
pipeline reached mAP **0.558**, above a YOLO11n trained on those classes (**0.473**) (**17**), at the price of
being far slower: a typical trade-off between foundation models and small specialists.

## 45. Monocular depth estimation (14)

Recovering depth from **one** image is mathematically impossible (chapter 11: a whole ray maps to one pixel),
yet humans do it constantly using **cues**: perspective lines, relative size, texture getting finer with
distance, occlusion, shading, haze. Networks learn the same cues from data.

**Depth Anything V2** is a ViT-based model trained on very large numbers of images (many with depth labels
produced by a teacher model trained on precise synthetic data). Important properties to understand when using it:

- **Relative vs metric depth**: the general models predict depth **up to an unknown scale and shift**: the order
  of what is near and far is right, but not the distance in meters. Metric variants are fine-tuned for indoor or
  outdoor scenes.
- Errors concentrate where the cues lie: mirrors, glass, textureless walls, unusual viewpoints, optical
  illusions.

Project 14 turns the predicted depth into a point cloud and new views (chapter 12) (**14**).

## 46. Vision-language models (18)

A **vision-language model (VLM)** answers free-form questions about images. The typical recipe:

1. A **vision encoder** (a ViT, often from the CLIP family) turns the image into a sequence of feature vectors.
2. A **projector** maps them into the token space of a **language model**, and compresses them so there are
   fewer image tokens.
3. The language model reads the image tokens followed by the question, and **generates** the answer one token at
   a time.

Large images are often **split into tiles** to keep detail, at the cost of many more tokens and slower answers;
project 18 disables tiling to stay fast on a CPU. **SmolVLM2-500M** (500 million parameters) answers an image
question in ~2.5 s on a laptop CPU (**18**).

Measuring a VLM: **VQA accuracy** compares the answer with those of 10 human annotators (simplified: correct if
enough humans gave the same answer). The small model reaches **67 %** with the standard short-answer prompt, and
the **prompt format** changes the score noticeably: VLM results are only meaningful together with the exact prompt
(**18**). Small VLMs are good at "what is in this image" and weak at counting and fine spatial reasoning.

## 47. Auto-labeling: models that label data for other models (17)

Labeling data by hand is the most expensive part of most vision projects. **Auto-labeling** uses a large, slow,
general model to produce labels, which humans then **review** (much faster than drawing every box), and which
train a small, fast model for deployment. This is a form of **knowledge distillation**: the big "teacher"
transfers what it knows to a small "student".

Project 17 writes Grounding DINO + SAM outputs directly as YOLO-format labels (chapter 27) for a whole folder. The
measurement that justifies the workflow: the zero-shot teacher already beats a supervised nano detector on the
evaluation classes (**17**). Labels from a model inherit its mistakes and biases, which is why the review step
is part of the method, not an option.

---

# Part 11 · Deployment

## 48. What makes a model fast or slow (19)

- **Parameters**: the number of learned weights; sets the model's **memory** footprint (×4 bytes in FP32).
- **FLOPs / MACs**: how many multiply-add operations one inference needs; roughly sets the **compute** cost.
  It grows with input resolution (doubling the side → 4× the FLOPs for a CNN).
- **Latency**: time for one input (what a live camera feels). **Throughput**: inputs per second (what a batch
  job cares about). Batching raises throughput but not latency.
- **Measure properly**: warm up first (the first runs include memory allocation and compilation), repeat many
  times, report the median, and describe the machine's state: the same benchmark ran noticeably slower while
  other programs were busy, so the README reports idle and busy numbers (**19**).

Real pipelines also spend time **outside** the network: decoding video, resizing, NMS, tracking, drawing.
Measure the whole loop, not just the model.

## 49. Model formats and runtimes: ONNX and OpenVINO (03, 06, 13, 19)

Training happens in a framework (PyTorch), but deployment needs something lighter and faster:

- **ONNX** is an open file format describing the network as a graph of standard operations. Export once, run
  anywhere: **ONNX Runtime** on CPU/GPU, **ONNX Runtime Web** in a browser (the PatchCore demo runs fully on the
  visitor's device, with WebGPU or WebAssembly), mobile runtimes…
- **OpenVINO** is Intel's runtime. It converts a model to its own format and optimizes it for Intel CPUs, GPUs
  and NPUs: fusing layers, choosing fast kernels, picking precisions.
- **Execution providers / devices**: the same model file can be sent to different hardware by changing one
  setting.

Changing the runtime can matter more than changing the model: swapping RapidOCR's backend from ONNX Runtime to
OpenVINO made OCR **~11× faster** with identical text (**03**). After any export, **check the outputs match** the
original model: the PatchCore ONNX export was re-validated on the full test set with identical AUROC (**13**).

## 50. Quantization: FP32, FP16, INT8 (13, 19)

Networks are trained with 32-bit floating-point numbers (**FP32**), but they rarely need that much precision.

- **FP16** (half precision): half the memory, faster on GPUs and NPUs, usually no accuracy change. Storing
  PatchCore's weights and memory banks in FP16 halves the download for the browser demo (**13**).
- **INT8**: weights and activations as 8-bit integers, with a **scale** (and zero point) per tensor or channel:
  `real ≈ scale × (int8 − zero_point)`. Four times smaller than FP32 and much faster on hardware with integer
  units.
- **Post-training quantization** needs a small **calibration set** (a few hundred representative images) to
  measure the range of activations and pick the scales. Tools like **NNCF** automate it.

Quantization can hurt accuracy, so measure it **on held-out data**: YOLO11n INT8 lost only **0.1 mAP point** on
500 COCO validation images not used for calibration (**19**).

## 51. CPU, GPU and NPU (19)

- A **CPU** has a few powerful, flexible cores. Good at everything, best at nothing for big matrix math.
  On hybrid laptop chips, using only the performance cores can be faster than using all of them.
- A **GPU** has thousands of simple cores running the same operation on many numbers at once: ideal for the
  matrix multiplications and convolutions inside networks. An **integrated GPU** (iGPU, like the Arc 140V)
  shares memory with the CPU.
- An **NPU** (neural processing unit) is a dedicated accelerator for neural-network operations, especially
  low-precision matrix math, at very low power. It supports fewer operations and settings (it rejected a
  precision hint the CPU and GPU accept) but is excellent for always-on inference (**19**).

Project 19 runs the same YOLO11n on all three units of one laptop and reaches a **17× speed-up** over the
PyTorch CPU baseline, showing that **deployment choices can matter more than model choices** on edge hardware
(**19**).

---

# Part 12 · Doing it responsibly

## 52. Honest evaluation (all projects)

A number is only meaningful with its context. Habits used throughout this repo:

- **Held-out data, always.** COCO128 is a slice of the *training* set; benchmarking on it overstated accuracy,
  so the edge benchmark switched to 500 held-out validation images (**19**).
- **Baselines.** Compare with something simple: a rule-based fall detector (**15**), zero-shot vs fine-tuned
  (**12**, **17**), raw OCR vs scanned (**03**). A model is only "good" relative to an alternative.
- **Report failures.** The scanner's failure cases stay in the figure (**03**); the ball detector's weakness is in
  the headline numbers (**20**).
- **Measure what the user feels.** Alerts (**07**), alarm delay (**15**), count error (**09**), speed plausibility
  (**08**).
- **Separate the stages** of a pipeline and measure each one: detection vs reading (**06**), page finding vs OCR
  (**03**).
- **Write down what changed** and its measured effect: every README has a "debugging story" table.

## 53. Privacy, bias, consent and licenses (10, 15, all)

Computer vision watches people, so it carries real risks:

- **Privacy and consent**: faces and bodies are personal data. The attendance tool only recognizes people who
  explicitly enrolled, and the face project includes an ethics and limits section (**10**). Fall detection works on
  skeletons, which reveal less than raw video (**15**).
- **Bias**: a model is only as fair as its data. Face recognition error rates have been shown to differ across
  demographic groups; a single overall accuracy (even 99.35 %) can hide that (**10**).
- **Misuse**: the same technology that counts cars can track individuals. Design for the narrowest use that
  solves the problem.
- **Licenses**: datasets and models come with terms. MVTec AD is **non-commercial** (CC BY-NC-SA), some datasets
  forbid redistribution, some model licenses (AGPL) impose conditions on derived software. This repository
  records them in [`DATA_LICENSES.md`](DATA_LICENSES.md) and never commits third-party data.

---

## Project → concept index

| Project | Chapters to read first |
|---|---|
| 01 Virtual painter | 1, 2, 33, 34 |
| 02 Solar panel defects | 13, 14, 16, 20–25 |
| 03 Document scanner + receipt OCR | 1–10, 41, 49 |
| 05 Sign language translator | 18, 21, 22, 33, 52 |
| 06 License plates | 23, 27–29, 41 |
| 07 PPE safety monitor | 16, 23, 27–30 |
| 08 Traffic analytics | 10, 28, 34–36 |
| 09 Retail shelf monitor | 23, 27–30 |
| 10 Face recognition | 9, 16, 18, 37, 38, 53 |
| 11 Polyp segmentation | 18, 21–23, 31, 32 |
| 12 Satellite land cover | 21–23, 25, 43 |
| 13 PatchCore anomaly detection | 5, 17, 21, 37, 39, 40, 49, 50 |
| 14 Monocular depth → 3D | 11, 12, 42, 45 |
| 15 Fall detection | 15–18, 33, 34 |
| 16 Text-to-image search | 37, 39, 42, 43 |
| 17 Zero-shot auto-labeling | 25, 27, 29, 42, 44, 47 |
| 18 Image & video Q&A | 34, 42, 46 |
| 19 Edge deployment benchmark | 28, 29, 48–52 |
| 20 Football analytics | 19, 27–30, 35, 36 |

## Glossary

- **Activation**: the output of a neuron after its non-linear function; also the intermediate values of a network.
- **Anchor-free**: a detector that predicts boxes directly from each location, without predefined box shapes.
- **AUROC**: area under the ROC curve; probability a positive outranks a negative.
- **Backbone**: the pretrained feature-extracting part of a network; a **head** sits on top for the task.
- **Batch**: a group of examples processed together in one training step.
- **Embedding**: a vector representation in which similar inputs are close.
- **Epoch**: one pass over the full training set.
- **Fine-tuning**: continuing to train a pretrained model on new data.
- **fps**: frames per second.
- **Ground truth**: the correct answer used for evaluation.
- **Homography**: 3×3 matrix mapping one plane to another; 4 point pairs define it.
- **Inference**: running a trained model on new inputs.
- **IoU**: intersection over union of two boxes or masks.
- **Logits**: a network's raw output scores before softmax.
- **mAP**: mean average precision over classes; the standard detection metric.
- **NMS**: non-maximum suppression; removes duplicate overlapping boxes.
- **Overfitting**: memorizing the training data instead of learning the general pattern.
- **Quantization**: storing and computing with fewer bits (FP16, INT8).
- **Receptive field**: the input region that influences one output value.
- **Token**: one element of a transformer's input sequence (a word piece, or an image patch).
- **Zero-shot**: performing a task without any training example for it.
