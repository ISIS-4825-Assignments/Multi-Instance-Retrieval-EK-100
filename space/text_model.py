"""Load AVION text encoder from Hugging Face for free-text queries."""
from __future__ import annotations

import os
import time
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import CLIP_LENGTH, HF_MODEL_REPO, MODEL_NAME, PROJECT_EMBED_DIM
from sms_source import ensure_sms_on_path

_TEXT_MODEL = None
_TOKENIZER = None
_DEVICE = None
_LAST_ENCODE_SEC: float | None = None
_LAST_LOAD_SEC: float | None = None


@lru_cache(maxsize=1)
def _load_text_module(device: str):
    ensure_sms_on_path()
    from functools import partial as _partial

    from avion.data.tokenizer import tokenize
    import avion.models.model_clip as model_clip

    from huggingface_hub import hf_hub_download

    text_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="text_encoder.pt")
    ckpt = torch.load(text_path, map_location="cpu", weights_only=False)

    model = getattr(model_clip, MODEL_NAME)(
        freeze_temperature=True,
        use_grad_checkpointing=False,
        context_length=77,
        vocab_size=49408,
        patch_dropout=0.0,
        drop_path_rate=0.0,
        num_frames=CLIP_LENGTH,
        use_fast_conv1=False,
        use_flash_attn=False,
        project_embed_dim=PROJECT_EMBED_DIM,
        pretrain_zoo="openai",
        pretrain_path=None,
    )

    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        key = k.replace("module.", "")
        if key.startswith("textual.") or key.startswith("text_projection"):
            cleaned[key] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    # Only text branch required
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    tokenizer = _partial(tokenize, context_length=77)
    return model, tokenizer, dev


def hf_space_is_cpu_only() -> bool:
    """True on Hugging Face Spaces `cpu-basic` (no attached GPU)."""
    hw = (os.environ.get("SPACES_HARDWARE") or os.environ.get("SPACE_HARDWARE") or "").lower()
    if "zero" in hw or "gpu" in hw or "a10" in hw or "t4" in hw:
        return False
    if "cpu" in hw:
        return True
    return not torch.cuda.is_available()


def get_device() -> str:
    if hf_space_is_cpu_only():
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def free_text_device() -> str:
    """Device for encoding a user query on the Space (CPU on free tier)."""
    return get_device()


def last_encode_seconds() -> float | None:
    return _LAST_ENCODE_SEC


def last_load_seconds() -> float | None:
    return _LAST_LOAD_SEC


@torch.no_grad()
def encode_text(query: str, device: str | None = None) -> np.ndarray:
    """L2-normalized text embedding (256-d) for dot-product with video_embeds."""
    global _TEXT_MODEL, _TOKENIZER, _DEVICE, _LAST_ENCODE_SEC, _LAST_LOAD_SEC
    dev = device or free_text_device()
    if dev == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    t0 = time.perf_counter()
    if _TEXT_MODEL is None or str(_DEVICE) != str(dev):
        load_t0 = time.perf_counter()
        _TEXT_MODEL, _TOKENIZER, _DEVICE = _load_text_module(dev)
        _LAST_LOAD_SEC = time.perf_counter() - load_t0

    tokens = _TOKENIZER([query])
    if isinstance(tokens, list):
        tokens = torch.stack(tokens)
    tokens = tokens.to(_DEVICE)
    cast_dtype = next(_TEXT_MODEL.parameters()).dtype
    text_embed = _TEXT_MODEL.encode_text(tokens, cast_dtype=cast_dtype)
    text_embed = F.normalize(text_embed, dim=-1)
    vec = text_embed[0].float().cpu().numpy()
    _LAST_ENCODE_SEC = time.perf_counter() - t0
    return vec.astype(np.float32)


def warm_text_encoder(device: str | None = None) -> float:
    """Load text encoder once; returns total seconds."""
    encode_text("warmup query", device=device)
    return last_encode_seconds() or 0.0


def cpu_free_text_eta_note() -> str:
    """User-facing estimate for HF cpu-basic."""
    return (
        "Free-text runs on **CPU** on this Space (no GPU). "
        "First search after startup: typically **2–4 minutes** (loads the text encoder). "
        "Later searches: about **5–20 seconds**. Ranking against precomputed "
        "`video_embeds.npy` is fast."
    )
