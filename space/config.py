"""Configuration for the EK-100 MIR Gradio demo."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = Path(__file__).resolve().parent

# Hugging Face repos
HF_DATASET_REPO = os.environ.get("EK100_DEMO_DATASET", "jsurrea/ek100-mir-demo-assets")
HF_MODEL_REPO = os.environ.get("EK100_DEMO_MODEL", "jsurrea/avion-vitl-ek100-sms")
# AVION Python package (`avion/`) lives beside weights on the same model repo.
HF_SMS_SOURCE_REPO = os.environ.get("EK100_SMS_SOURCE_REPO", HF_MODEL_REPO)

def hf_thumbnail_url(vis_id: str, frame: int = 0) -> str:
    return (
        f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/"
        f"thumbnails/{vis_id}/frame_{frame}.jpg"
    )


def hf_clip_url(vis_id: str) -> str:
    return (
        f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/"
        f"clips/{vis_id}.mp4"
    )

# Local override (development / Colab before upload)
LOCAL_ASSETS_DIR = Path(os.environ.get("EK100_DEMO_ASSETS", SPACE_DIR / "assets"))

TOP_K = int(os.environ.get("EK100_DEMO_TOP_K", "10"))
EMBED_DIM = 256

# AVION ViT-L + SMS (matches train_and_test.ipynb / test_mir.py)
MODEL_NAME = "CLIP_VITL14"
PROJECT_EMBED_DIM = 256
CLIP_LENGTH = 32
CLIP_STRIDE = 4
VIDEO_CHUNK_LENGTH = 15

# Preset queries (narration_id, label) — paper-style examples
PRESET_QUERIES: list[tuple[str, str]] = [
    ("P08_09_74", "cut tomato"),
    ("P01_11_0", "take plate"),
    ("P01_11_100", "wash cloth"),
    ("P02_02_110", "open fridge"),
    ("P03_04_23", "stir soup"),
    ("P04_101_0", "wash cup"),
    ("P05_07_0", "turn on tap"),
    ("P06_01_10", "close drawer"),
    ("P07_07_0", "put down knife"),
    ("P08_16_63", "cut tomatoes"),
    ("P09_02_0", "open cupboard"),
    ("P10_04_0", "take bowl"),
    ("P11_16_0", "wipe table"),
    ("P12_03_0", "pour water"),
    ("P13_07_0", "open oven"),
    ("P14_06_0", "close fridge"),
    ("P15_02_0", "take pan"),
    ("P16_05_0", "stir pan"),
    ("P17_03_0", "wash hands"),
    ("P18_05_79", "cut tomatoes"),
    ("P22_01_0", "open microwave"),
    ("P23_05_0", "put down cup"),
    ("P24_09_236", "cut tomato"),
    ("P25_04_0", "take spoon"),
    ("P26_41_0", "close tap"),
    ("P27_02_0", "open bin"),
    ("P28_24_1", "cut tomato"),
    ("P30_01_0", "wash plate"),
    ("P31_104_0", "take lid"),
    ("P32_07_0", "open door"),
]
