"""Preset query helpers."""
from __future__ import annotations

import random

from config import PRESET_QUERIES
from data_loader import DemoAssets, load_assets


def preset_choices() -> list[str]:
    return [cap for _nid, cap in PRESET_QUERIES]


def parse_preset_choice(choice: str | None) -> tuple[str, str]:
    if not choice:
        return "", ""
    cap = choice.strip()
    for nid, c in PRESET_QUERIES:
        if c == cap:
            return nid, c
    return "", cap


def preset_narration_ids() -> list[str]:
    return [nid for nid, _ in PRESET_QUERIES]


def pick_random_v2t_clip(assets: DemoAssets | None = None, exclude: str | None = None) -> str:
    """Pick a test-set segment with thumbnails and a sim_mat row."""
    assets = assets or load_assets()
    pool = assets.v2t_clip_pool
    if not pool:
        raise RuntimeError("No video clips available for Video → Text (missing v2t_clip_pool.json).")
    exclude = (exclude or "").strip()
    candidates = [v for v in pool if v != exclude] if exclude else pool
    if not candidates:
        candidates = pool
    return random.choice(candidates)
