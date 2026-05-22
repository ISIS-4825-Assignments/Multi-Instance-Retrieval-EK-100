"""Load submission matrix, metadata, and optional precomputed embeddings."""
from __future__ import annotations

import json
import os
import pickle
import time
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
    hf_files: frozenset[str]
    clips_on_dataset: frozenset[str]
    thumbs_on_dataset: frozenset[str]


_CACHE: DemoAssets | None = None
_HF_FILES_CACHE: frozenset[str] | None = None
_HF_FILES_CACHE_AT: float = 0.0
_HF_FILES_REFRESH_SEC = int(os.environ.get("EK100_DEMO_MEDIA_REFRESH_SEC", "90"))


def _list_hf_dataset_files(force: bool = False) -> frozenset[str]:
    """Paths on the demo dataset repo (refreshed periodically as Colab uploads)."""
    global _HF_FILES_CACHE, _HF_FILES_CACHE_AT
    now = time.time()
    if (
        not force
        and _HF_FILES_CACHE is not None
        and now - _HF_FILES_CACHE_AT < _HF_FILES_REFRESH_SEC
    ):
        return _HF_FILES_CACHE

    paths: set[str] = set()
    try:
        from huggingface_hub import HfApi

        for path in HfApi().list_repo_files(HF_DATASET_REPO, repo_type="dataset"):
            paths.add(path)
    except Exception:
        paths = set()

    _HF_FILES_CACHE = frozenset(paths)
    _HF_FILES_CACHE_AT = now
    return _HF_FILES_CACHE


def _media_sets_from_hf_files(
    hf_files: frozenset[str], vis_id_to_row: dict[str, int]
) -> tuple[frozenset[str], frozenset[str]]:
    clips: set[str] = set()
    thumbs: set[str] = set()
    for path in hf_files:
        if path.startswith("clips/") and path.endswith(".mp4"):
            vid = path[len("clips/") : -4]
            if vid in vis_id_to_row:
                clips.add(vid)
        elif path.startswith("thumbnails/") and path.endswith("/frame_0.jpg"):
            parts = path.split("/")
            if len(parts) >= 3:
                vid = parts[1]
                if vid in vis_id_to_row:
                    thumbs.add(vid)
    return frozenset(clips), frozenset(thumbs)


def has_hf_clip(assets: DemoAssets, vis_id: str) -> bool:
    return vis_id in assets.clips_on_dataset


def has_hf_thumbnail(assets: DemoAssets, vis_id: str) -> bool:
    return vis_id in assets.thumbs_on_dataset


def has_hf_media(assets: DemoAssets, vis_id: str) -> bool:
    return has_hf_clip(assets, vis_id) or has_hf_thumbnail(assets, vis_id)


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
        # Refresh HF file index so incremental Colab uploads show up without redeploy.
        hf_files = _list_hf_dataset_files()
        clips_on, thumbs_on = _media_sets_from_hf_files(hf_files, _CACHE.vis_id_to_row)
        if hf_files != _CACHE.hf_files:
            pool_json = set(_CACHE.v2t_clip_pool)
            media_ids = clips_on | thumbs_on | pool_json
            _CACHE = DemoAssets(
                sim_mat=_CACHE.sim_mat,
                vis_ids=_CACHE.vis_ids,
                txt_ids=_CACHE.txt_ids,
                test_df=_CACHE.test_df,
                sentence_df=_CACHE.sentence_df,
                video_embeds=_CACHE.video_embeds,
                vis_id_to_row=_CACHE.vis_id_to_row,
                txt_id_to_col=_CACHE.txt_id_to_col,
                narration_to_txt_id=_CACHE.narration_to_txt_id,
                assets_dir=_CACHE.assets_dir,
                thumbnails_dir=_CACHE.thumbnails_dir,
                thumbnail_vis_ids=media_ids,
                v2t_clip_pool=sorted(media_ids & set(_CACHE.vis_id_to_row)),
                hf_files=hf_files,
                clips_on_dataset=clips_on,
                thumbs_on_dataset=thumbs_on,
            )
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
    hf_files = _list_hf_dataset_files(force=force_reload)
    clips_on, thumbs_on = _media_sets_from_hf_files(hf_files, vis_id_to_row)
    pool_json = set(_load_v2t_clip_pool(vis_id_to_row, thumbs))
    media_ids = clips_on | thumbs_on | pool_json
    thumbnail_vis_ids = frozenset(media_ids)
    v2t_clip_pool = sorted(
        vid for vid in (thumbs_on | clips_on) if vid in vis_id_to_row
    ) or sorted(media_ids)

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
        hf_files=hf_files,
        clips_on_dataset=clips_on,
        thumbs_on_dataset=thumbs_on,
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
