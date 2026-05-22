"""Gradio UI helpers: thumbnails, labels, and structured search results."""
from __future__ import annotations

import html
import io
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from config import hf_thumbnail_url
from data_loader import DemoAssets, load_assets

_THUMB_MIN_BYTES = 500


def _local_frame_path(assets: DemoAssets, vis_id: str, frame: int) -> Path | None:
    if assets.thumbnails_dir is None:
        return None
    p = assets.thumbnails_dir / vis_id / f"frame_{frame}.jpg"
    if p.is_file() and p.stat().st_size > _THUMB_MIN_BYTES:
        return p
    return None


def _fetch_image_array(url: str) -> np.ndarray | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ek100-mir-demo/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        if len(data) < _THUMB_MIN_BYTES:
            return None
        from PIL import Image

        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def image_for_gradio(assets: DemoAssets, vis_id: str, frame: int = 0) -> str | np.ndarray | None:
    """
    Value for gr.Image: local path, or RGB numpy from HF CDN.
    Server-side fetch avoids broken previews when the Space cannot load resolve URLs directly.
    """
    local = _local_frame_path(assets, vis_id, frame)
    if local is not None:
        return str(local)
    return _fetch_image_array(hf_thumbnail_url(vis_id, frame))


def frame_images_for_gradio(
    assets: DemoAssets, vis_id: str, num_frames: int = 3
) -> tuple[str | np.ndarray | None, str | np.ndarray | None, str | np.ndarray | None]:
    return tuple(image_for_gradio(assets, vis_id, i) for i in range(num_frames))


def preview_image(assets: DemoAssets, vis_id: str) -> str | np.ndarray | None:
    for frame in (1, 0, 2):
        img = image_for_gradio(assets, vis_id, frame)
        if img is not None:
            return img
    return None


def v2t_clip_header_md(assets: DemoAssets, vis_id: str) -> str:
    row = assets.test_df.loc[assets.test_df["narration_id"] == vis_id]
    if row.empty:
        return f"**`{vis_id}`**"
    row = row.iloc[0]
    ref = html.escape(str(row["narration"]).strip())
    return (
        f"**`{vis_id}`** · `{row['participant_id']}` · `{row['video_id']}` · "
        f"`{row['start_timestamp']}` → `{row['stop_timestamp']}`\n\n"
        f"**Ground truth:** {ref}"
    )


def _v2t_norm_caption(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def v2t_predictions_html(
    results: list[dict],
    *,
    ground_truth: str | None = None,
) -> str:
    if not results:
        return (
            "<p class='v2t-empty' style='color:#52525b!important;"
            "font-size:0.875rem;'>No captions ranked for this clip.</p>"
        )

    gt_norm = _v2t_norm_caption(ground_truth) if ground_truth else ""

    def is_gt(caption: str) -> bool:
        return bool(gt_norm) and _v2t_norm_caption(caption) == gt_norm

    def card(rank: int, item: dict) -> str:
        cap = html.escape((item.get("caption") or "").strip())
        score = float(item["score"])
        lead = " v2t-card--lead" if rank == 1 else ""
        gt_badge = (
            "<span class='v2t-gt'>Ground truth</span>" if is_gt(item.get("caption") or "") else ""
        )
        return f"""
        <article class="v2t-card{lead}">
          <div class="v2t-card-meta">
            <span class="v2t-rank">#{rank}</span>
            <span class="v2t-score">{score:.4f}</span>
          </div>
          <p class="v2t-caption">{cap}</p>
          {gt_badge}
        </article>
        """

    top3 = results[:3]
    rest = results[3:]

    podium = "".join(card(i + 1, item) for i, item in enumerate(top3))

    table_rows = []
    for i, item in enumerate(rest, start=4):
        cap = html.escape((item.get("caption") or "").strip())
        score = float(item["score"])
        gt_cell = (
            "<span class='v2t-gt v2t-gt--inline'>GT</span>"
            if is_gt(item.get("caption") or "")
            else ""
        )
        table_rows.append(
            f"<tr><td class='v2t-td-rank'>{i}</td>"
            f"<td class='v2t-td-score'>{score:.4f}</td>"
            f"<td class='v2t-td-cap'>{cap}{gt_cell}</td></tr>"
        )

    table_block = ""
    if table_rows:
        table_block = f"""
        <details class="v2t-more" open>
          <summary>Ranks 4–{len(results)}</summary>
          <table class="v2t-table">
            <thead><tr><th>#</th><th>Score</th><th>Caption</th></tr></thead>
            <tbody>{"".join(table_rows)}</tbody>
          </table>
        </details>
        """

    return f"""
<style>
/* Self-contained light panel: Gradio dark theme inherits pale text onto our HTML. */
.v2t-wrap {{
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  margin-top: 10px;
}}
.v2t-panel {{
  color-scheme: light;
  background: #f4f4f5;
  border: 1px solid #d4d4d8;
  border-radius: 12px;
  padding: 18px 20px;
  color: #09090b !important;
}}
.v2t-panel,
.v2t-panel p,
.v2t-panel span,
.v2t-panel td,
.v2t-panel th,
.v2t-panel summary {{
  color: #09090b !important;
}}
.v2t-title {{
  font-size: 1rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  margin: 0 0 16px 0;
  color: #09090b !important;
}}
.v2t-podium {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}}
@media (max-width: 720px) {{
  .v2t-podium {{ grid-template-columns: 1fr; }}
}}
.v2t-card {{
  background: #ffffff !important;
  border: 1px solid #d4d4d8;
  border-radius: 10px;
  padding: 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 108px;
}}
.v2t-card--lead {{
  border-color: #71717a;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}}
.v2t-card-meta {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}}
.v2t-rank {{
  font-size: 0.75rem;
  font-weight: 600;
  color: #3f3f46 !important;
  background: #e4e4e7 !important;
  padding: 4px 8px;
  border-radius: 6px;
}}
.v2t-card--lead .v2t-rank {{
  color: #fafafa !important;
  background: #27272a !important;
}}
.v2t-score {{
  font-size: 0.8125rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: #18181b !important;
  background: #e4e4e7 !important;
  padding: 4px 10px;
  border-radius: 6px;
}}
.v2t-caption {{
  margin: 0;
  font-size: 1.05rem;
  font-weight: 600;
  line-height: 1.45;
  color: #09090b !important;
  flex: 1;
}}
.v2t-gt {{
  font-size: 0.6875rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #14532d !important;
  background: #dcfce7 !important;
  border: 1px solid #86efac;
  padding: 4px 8px;
  border-radius: 6px;
  align-self: flex-start;
}}
.v2t-gt--inline {{
  margin-left: 8px;
  font-size: 0.625rem;
  padding: 2px 6px;
  vertical-align: middle;
}}
.v2t-more summary {{
  cursor: pointer;
  font-weight: 600;
  font-size: 0.875rem;
  color: #3f3f46 !important;
  margin-bottom: 10px;
  list-style: none;
}}
.v2t-more summary::-webkit-details-marker {{ display: none; }}
.v2t-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
  border: 1px solid #d4d4d8;
  border-radius: 8px;
  overflow: hidden;
  background: #ffffff !important;
}}
.v2t-table th {{
  text-align: left;
  padding: 10px 12px;
  background: #e4e4e7 !important;
  color: #27272a !important;
  font-weight: 700;
  border-bottom: 1px solid #d4d4d8;
}}
.v2t-table td {{
  padding: 10px 12px;
  border-bottom: 1px solid #e4e4e7;
  vertical-align: top;
  background: #ffffff !important;
  color: #09090b !important;
}}
.v2t-table tbody tr:last-child td {{ border-bottom: none; }}
.v2t-table tbody tr:hover td {{ background: #f4f4f5 !important; }}
.v2t-td-rank {{
  width: 2.5rem;
  font-weight: 700;
  color: #52525b !important;
}}
.v2t-td-score {{
  width: 5.5rem;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  color: #18181b !important;
}}
.v2t-td-cap {{
  color: #09090b !important;
  line-height: 1.45;
  font-weight: 500;
}}
.v2t-empty {{
  color: #52525b !important;
  font-size: 0.875rem;
}}
@media (prefers-color-scheme: dark) {{
  .v2t-panel {{
    background: #fafafa;
    border-color: #a1a1aa;
  }}
}}
</style>
<div class="v2t-wrap">
  <div class="v2t-panel">
    <p class="v2t-title">Predicted captions</p>
    <div class="v2t-podium">{podium}</div>
    {table_block}
  </div>
</div>
"""


def v2t_predictions_md(results: list[dict], **kwargs) -> str:
    return v2t_predictions_html(results, **kwargs)


segment_summary_md = v2t_clip_header_md


def t2v_choice_label(rank: int, item: dict) -> str:
    narr = (item.get("narration") or "")[:72]
    if len(item.get("narration") or "") > 72:
        narr += "…"
    return f"#{rank}  ·  {item['score']:.4f}  ·  {narr}"


def t2v_detail_md(item: dict, rank: int, mode: str) -> str:
    return f"""### Rank {rank}
**Score:** `{item['score']:.4f}` · **Segment:** `{item['vis_id']}`

> {item.get('narration', '')}

`{item.get('participant', '')}` / `{item.get('video_id', '')}` — *{mode}*
"""


def t2v_choices(results: list[dict]) -> list[str]:
    return [t2v_choice_label(i, r) for i, r in enumerate(results, 1)]


def parse_rank_choice(choice: str | None) -> int:
    if not choice or not str(choice).startswith("#"):
        return 0
    try:
        return int(str(choice).split("#", 1)[1].split("·", 1)[0].strip()) - 1
    except ValueError:
        return 0


parse_t2v_choice = parse_rank_choice
parse_v2t_choice = parse_rank_choice


def t2v_frames_for_item(assets: DemoAssets, item: dict) -> tuple:
    return frame_images_for_gradio(assets, item["vis_id"], 3)


def v2t_choice_label(rank: int, item: dict) -> str:
    cap = (item.get("caption") or "")[:80]
    if len(item.get("caption") or "") > 80:
        cap += "…"
    return f"#{rank}  ·  {item['score']:.4f}  ·  {cap}"


def v2t_detail_md(item: dict, rank: int) -> str:
    return f"""### Rank {rank} matching caption
**Score:** `{item['score']:.4f}`

> {item.get('caption', '')}
"""


def v2t_choices(results: list[dict]) -> list[str]:
    return [v2t_choice_label(i, r) for i, r in enumerate(results, 1)]


def empty_t2v_outputs():
    import gradio as gr

    return (
        "",
        [],
        None,
        None,
        None,
        "Run a search to see ranked segments.",
        gr.update(choices=[], value=None),
        None,
    )


def empty_v2t_outputs():
    return (
        None,
        None,
        None,
        "Loading a random kitchen clip…",
        "",
        "",
    )
