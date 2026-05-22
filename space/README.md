---
title: EK100 MIR Demo
emoji: 🍳
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
hardware: cpu-basic
startup_duration_timeout: 1h
---

# EPIC-KITCHENS-100 Multi-Instance Retrieval Demo

Interactive demo for **AVION ViT-L + SMS** ([model card](https://huggingface.co/jsurrea/avion-vitl-ek100-sms)) — **69.68 nDCG AVG** on the [Codabench challenge](https://www.codabench.org/competitions/12008).

Runs on **CPU** (free tier). Preset search is instant. **Free-text** encodes your query on CPU (~2–4 min first time, ~5–20 s after) and ranks against precomputed `video_embeds.npy`.

## Assets

Precomputed files are loaded from the dataset [`jsurrea/ek100-mir-demo-assets`](https://huggingface.co/datasets/jsurrea/ek100-mir-demo-assets):

- `test.pkl` — full similarity matrix (9668 × 3842)
- CSV metadata
- `video_embeds.npy` — optional, enables free-text search
- `thumbnails/` — optional JPEG frames for preset results

## Source repo

Training and submission notebooks live under `src/` in [ISIS-4825-Assignments/Multi-Instance-Retrieval-EK-100](https://github.com/ISIS-4825-Assignments/Multi-Instance-Retrieval-EK-100). All demo runtime code lives in this `space/` folder (`app.py` plus sibling modules).
