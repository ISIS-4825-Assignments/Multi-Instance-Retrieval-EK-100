"""Ranking: preset captions via sim_mat, free text via video embeddings."""
from __future__ import annotations

import numpy as np

from config import TOP_K
from data_loader import DemoAssets, caption_for_txt_id, load_assets, rank_from_sim_column, rank_from_sim_row


def normalize_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def find_txt_id_for_query(assets: DemoAssets, query: str) -> str | None:
    """Match a test-set caption exactly (case-insensitive)."""
    q = normalize_query(query)
    for _, row in assets.sentence_df.iterrows():
        if normalize_query(str(row["narration"])) == q:
            return str(row["narration_id"])
    return None


def rank_text_to_video(
    query: str,
    *,
    text_vec: np.ndarray | None = None,
    top_k: int = TOP_K,
    assets: DemoAssets | None = None,
) -> tuple[list[dict], str]:
    """
    Return ranked video segments and a short mode label.
    Uses sim_mat when query matches a benchmark caption; else dot-product with video_embeds.
    """
    assets = assets or load_assets()
    txt_id = find_txt_id_for_query(assets, query)

    if txt_id is not None and txt_id in assets.txt_id_to_col:
        pairs = rank_from_sim_column(assets, txt_id, top_k)
        mode = "benchmark caption (submission sim_mat)"
    elif text_vec is not None and assets.video_embeds is not None:
        scores = assets.video_embeds @ text_vec.astype(np.float32)
        order = np.argsort(-scores)[:top_k]
        pairs = [(assets.vis_ids[i], float(scores[i])) for i in order]
        mode = "free text (live encoder + precomputed video embeddings)"
    else:
        return [], "free text unavailable (upload video_embeds.npy to the demo dataset)"

    results = []
    for vis_id, score in pairs:
        row = assets.test_df.loc[assets.test_df["narration_id"] == vis_id]
        narration = str(row.iloc[0]["narration"]) if not row.empty else vis_id
        results.append(
            {
                "vis_id": vis_id,
                "score": score,
                "narration": narration,
                "participant": str(row.iloc[0]["participant_id"]) if not row.empty else "",
                "video_id": str(row.iloc[0]["video_id"]) if not row.empty else "",
            }
        )
    return results, mode


def rank_video_to_text(
    vis_id: str,
    *,
    top_k: int = TOP_K,
    assets: DemoAssets | None = None,
) -> list[dict]:
    assets = assets or load_assets()
    pairs = rank_from_sim_row(assets, vis_id, top_k)
    out = []
    for txt_id, score in pairs:
        out.append(
            {
                "txt_id": txt_id,
                "score": score,
                "caption": caption_for_txt_id(assets, txt_id),
            }
        )
    return out
