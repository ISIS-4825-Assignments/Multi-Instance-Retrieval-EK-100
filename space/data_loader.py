"""Load submission matrix, metadata, and optional precomputed embeddings."""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import HF_DATASET_REPO, LOCAL_ASSETS_DIR, TOP_K

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None  # type: ignore


@dataclass
class DemoAssets:
    sim_mat: np.ndarray
    vis_ids: list[str]
    txt_ids: list[str]
    test_df: pd.DataFrame
    sentence_df: pd.DataFrame
    video_embeds: np.ndarray | None
    vis_id_to_row: dict[str, int]
    txt_id_to_col: dict[str, int]
    narration_to_txt_id: dict[str, str]
    assets_dir: Path
    thumbnails_dir: Path | None
    thumbnail_vis_ids: frozenset[str]
    v2t_clip_pool: list[str]


_CACHE: DemoAssets | None = None


def _resolve(path: str, local_name: str) -> Path:
    local = LOCAL_ASSETS_DIR / local_name
    if local.is_file():
        return local
    assets_dir = Path(__file__).parent / "assets"
    for candidate in (
        assets_dir / local_name,
        assets_dir / Path(local_name).name,
        Path(__file__).parent / "metadata" / Path(local_name).name,
        assets_dir / "metadata" / Path(local_name).name,
    ):
        if candidate.is_file():
            return candidate
    if hf_hub_download is None:
        raise FileNotFoundError(
            f"Missing {local_name}. Set EK100_DEMO_ASSETS or install huggingface_hub."
        )
    return Path(
        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=path,
            repo_type="dataset",
        )
    )


def _resolve_dir(subdir: str) -> Path | None:
    """Local thumbnails when present; Space loads images via HF URLs in viz.image_for_gradio."""
    local = LOCAL_ASSETS_DIR / subdir
    if local.is_dir() and any(local.iterdir()):
        return local
    return None


def load_assets(force_reload: bool = False) -> DemoAssets:
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE

    pkl_path = _resolve("test.pkl", "test.pkl")
    with open(pkl_path, "rb") as f:
        sub = pickle.load(f)

    sim_mat = np.asarray(sub["sim_mat"], dtype=np.float32)
    vis_ids = list(sub["vis_ids"])
    txt_ids = list(sub["txt_ids"])

    test_csv = _resolve("EPIC_100_retrieval_test.csv", "metadata/EPIC_100_retrieval_test.csv")
    sent_csv = _resolve(
        "EPIC_100_retrieval_test_sentence.csv",
        "metadata/EPIC_100_retrieval_test_sentence.csv",
    )
    test_df = pd.read_csv(test_csv)
    sentence_df = pd.read_csv(sent_csv)

    vis_id_to_row = {vid: i for i, vid in enumerate(vis_ids)}
    txt_id_to_col = {tid: j for j, tid in enumerate(txt_ids)}
    # narration_id in sentence csv equals txt_id in submission
    narration_to_txt_id = {
        str(r["narration_id"]): str(r["narration_id"])
        for _, r in sentence_df.iterrows()
    }

    video_embeds = None
    emb_local = LOCAL_ASSETS_DIR / "video_embeds.npy"
    if emb_local.is_file():
        video_embeds = np.load(emb_local)
    elif hf_hub_download is not None:
        try:
            emb_path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename="video_embeds.npy",
                repo_type="dataset",
            )
            video_embeds = np.load(emb_path)
        except Exception:
            video_embeds = None

    vis_ids_json = LOCAL_ASSETS_DIR / "vis_ids.json"
    if video_embeds is not None and vis_ids_json.is_file():
        with open(vis_ids_json) as f:
            order = json.load(f)
        if order != vis_ids:
            idx = [vis_id_to_row[v] for v in order]
            video_embeds = video_embeds[idx]

    thumbs = _resolve_dir("thumbnails")
    v2t_clip_pool = _load_v2t_clip_pool(vis_id_to_row, thumbs)
    thumbnail_vis_ids = frozenset(v2t_clip_pool)

    _CACHE = DemoAssets(
        sim_mat=sim_mat,
        vis_ids=vis_ids,
        txt_ids=txt_ids,
        test_df=test_df,
        sentence_df=sentence_df,
        video_embeds=video_embeds,
        vis_id_to_row=vis_id_to_row,
        txt_id_to_col=txt_id_to_col,
        narration_to_txt_id=narration_to_txt_id,
        assets_dir=LOCAL_ASSETS_DIR,
        thumbnails_dir=thumbs,
        thumbnail_vis_ids=thumbnail_vis_ids,
        v2t_clip_pool=v2t_clip_pool,
    )
    return _CACHE


def _load_v2t_clip_pool(vis_id_to_row: dict[str, int], thumbs: Path | None) -> list[str]:
    """vis_ids that are in sim_mat and have preview frames on the demo dataset."""
    pool: list[str] = []
    try:
        pool_path = _resolve("v2t_clip_pool.json", "v2t_clip_pool.json")
        raw = json.loads(pool_path.read_text())
        if isinstance(raw, list):
            pool = [str(v) for v in raw]
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pool = []

    if not pool and thumbs is not None:
        pool = sorted(
            d.name
            for d in thumbs.iterdir()
            if d.is_dir() and (d / "frame_0.jpg").is_file() and d.name in vis_id_to_row
        )

    return [v for v in pool if v in vis_id_to_row]


def segment_row(assets: DemoAssets, vis_id: str) -> pd.Series | None:
    rows = assets.test_df.loc[assets.test_df["narration_id"] == vis_id]
    if rows.empty:
        return None
    return rows.iloc[0]


def caption_for_txt_id(assets: DemoAssets, txt_id: str) -> str:
    rows = assets.sentence_df.loc[assets.sentence_df["narration_id"] == txt_id]
    if rows.empty:
        return txt_id
    return str(rows.iloc[0]["narration"])


def rank_from_sim_column(assets: DemoAssets, txt_id: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    if txt_id not in assets.txt_id_to_col:
        return []
    col = assets.txt_id_to_col[txt_id]
    scores = assets.sim_mat[:, col]
    order = np.argsort(-scores)[:top_k]
    return [(assets.vis_ids[i], float(scores[i])) for i in order]


def rank_from_sim_row(assets: DemoAssets, vis_id: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    if vis_id not in assets.vis_id_to_row:
        return []
    row = assets.vis_id_to_row[vis_id]
    scores = assets.sim_mat[row, :]
    order = np.argsort(-scores)[:top_k]
    return [(assets.txt_ids[j], float(scores[j])) for j in order]
