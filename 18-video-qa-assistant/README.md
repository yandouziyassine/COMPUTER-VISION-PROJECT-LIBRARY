# 18 · Image & Video Q&A Assistant (local, CPU)

Ask questions in plain English about a photo or a video ("How many people are there?", "What
happens in this clip?") and get an answer **fully offline**, with no API key and no GPU. It runs
**SmolVLM2-500M**, a small vision-language model, on a laptop CPU at ~2.5 s per image question.
Two experiments measure what such a small model can and can't do.

![VQA examples](docs/vqa_examples.jpg)

*VQAv2 questions on COCO photos. Top: correct answers. Bottom: mistakes (model answer vs the most
common answer from 10 humans).*

## Results

### 1. Image questions: VQAv2 (300 validation questions on COCO images)

Simplified VQA accuracy: an answer scores `min(#humans who gave it / 3, 1)` (10 people answered each question).
The official metric also averages over the 10 leave-one-annotator-out subsets and uses a longer
normalization list, so these numbers are close to, but not exactly, official VQAv2 scores.

| Prompt appended to the question | Overall | Yes/no (118) | Number (34) | Other (148) | Time / question |
|---|---|---|---|---|---|
| "Answer with a single word or number." | 58.3 % | 69.5 % | 48.0 % | 51.8 % | 2.1 s |
| **"Answer the question using a single word or phrase."** | **67.0 %** | **84.7 %** | 46.1 % | **57.7 %** | 2.5 s |

**Prompt wording alone moved accuracy by almost 9 points.** With the first prompt the model answered
"1" to 30 of 300 questions (14 of them yes/no questions!); the standard VQA phrasing cut that to 7.
Small models follow the *form* of an instruction very literally.

### 2. Video question: "Did the person fall?" (UR Fall, 70 videos, 8 frames each)

Same 30 falls + 40 daily activities as project 15, asked zero-shot:

| Approach | Sensitivity (falls caught) | Specificity | Time / video |
|---|---|---|---|
| SmolVLM2: "Does the person fall down during the video?" | 3.3 % (1/30) | 100 % | 10.8 s |
| SmolVLM2: "In the last frames, is the person lying on the floor?" | 13.3 % (4/30) | 97.5 % | 14.7 s |
| **Project 15**: pose features + trained classifier | **~97–100 %** | 80–100 % | real time |

The small general model almost always answers "no". Rewording to a concrete visual question barely
helps, so it's a **perception** limit (small, low-resolution people in 8 sampled frames), not
just a prompt problem. A specialized pipeline beats a general model by a mile on a narrow task;
the general model is for open-ended questions no one trained a detector for.

## How it works

1. **Model**: SmolVLM2-500M-Video-Instruct (Hugging Face Transformers), float32 on CPU, 4 threads.
2. **Images**: each image is resized to one 512 px view. By default SmolVLM tiles every image into many
   crops (thousands of tokens, ~40 s per question on CPU); turning tiling off made it ~15× faster.
3. **Videos**: 8 frames sampled evenly over the clip are given as an image sequence, followed by the question.
4. **Decoding**: greedy (deterministic), a few tokens for short answers.
5. **Evaluation** ([`evaluate.py`](evaluate.py)): questions are restricted to COCO images we already
   have (26,333 VQAv2 questions qualify; 300 sampled with a fixed seed). Answers are normalized
   (lowercase, punctuation, articles, number words → digits) as in the VQA protocol (simplified).
   Runs are checkpointed every 5 questions so they survive interruptions.

## Run it

```bash
.venv\Scripts\activate            # shared environment, from the repo root
cd 18-video-qa-assistant
python ask.py photo.jpg "How many people are in this picture?"
python ask.py clip.mp4 "What happens in this video?" --frames 8
python evaluate.py --n-vqa 300                                   # VQA + fall test (~1 h on CPU)
python evaluate.py --only-vqa-prompt standard --n-vqa 300        # VQA with the standard prompt
python evaluate.py --only-fall-variant lying                     # the "lying on the floor" question
python make_figures.py
```

## Data and model

- **VQAv2** validation questions and answers (visualqa.org, CC BY 4.0), on the COCO val2017 images from project 16.
- **UR Fall Detection** videos (project 15).
- **Model:** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (Apache-2.0).
- **Location:** `../../cv-data/vqa/` (not committed).

## What I learned / what didn't work

- **Prompt format is part of the model.** 58 % → 67 % from one sentence; always try the phrasing
  the model was evaluated with.
- **Image tiling is the hidden cost** of modern vision-language models: it improves detail but
  multiplies tokens. On CPU, one view per image is the practical setting.
- **General ≠ good at everything.** The same model that answers 85 % of yes/no photo questions
  catches almost no falls. For a safety task, a specialized, measured pipeline (project 15) is the answer.
- **Counting is weak** (~46–48 %), a known limitation of small vision-language models.
- My first two evaluation runs died: the USB key disconnected, then the laptop slept. Hence the
  checkpointing and running long jobs as separate processes.

## Next steps

- Larger model (SmolVLM2-2.2B or Qwen2.5-VL-3B) through OpenVINO for speed (see project 19).
- More frames and higher resolution for the video question, to see where perception starts to work.
- Retrieval over long videos: caption every few seconds, embed the captions, answer from the relevant moments.

## Tech stack

Python · Hugging Face Transformers (SmolVLM2) · PyTorch · OpenCV · NumPy · Matplotlib
