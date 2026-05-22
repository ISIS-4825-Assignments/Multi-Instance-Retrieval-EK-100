"""
Gradio Space: EPIC-KITCHENS-100 Multi-Instance Retrieval (AVION ViT-L + SMS).
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
os.environ.setdefault("GRADIO_SERVER_PORT", "7860")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

import sys
from pathlib import Path

SPACE_DIR = Path(__file__).resolve().parent
if str(SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(SPACE_DIR))

import gradio as gr

try:
    import spaces
except ImportError:
    spaces = None


def _space_has_zero_gpu() -> bool:
    hw = (os.environ.get("SPACES_HARDWARE") or os.environ.get("SPACE_HARDWARE") or "").lower()
    if "zero" in hw or "a10g" in hw:
        return True
    return os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true", "yes")


from data_loader import load_assets, segment_row
from presets import parse_preset_choice, pick_random_v2t_clip, preset_choices
from rank import find_txt_id_for_query, rank_text_to_video, rank_video_to_text
from text_model import (
    cpu_free_text_eta_note,
    encode_text,
    free_text_device,
    hf_space_is_cpu_only,
    last_encode_seconds,
    last_load_seconds,
)
from viz import (
    empty_t2v_outputs,
    empty_v2t_outputs,
    frame_images_for_gradio,
    parse_t2v_choice,
    preview_image,
    t2v_choices,
    t2v_detail_md,
    t2v_frames_for_item,
    v2t_clip_header_md,
    v2t_predictions_html,
)


def _rank_text(query: str, *, device: str | None = None):
    query = (query or "").strip()
    if not query:
        return [], "Enter a query."

    assets = load_assets()
    txt_id = find_txt_id_for_query(assets, query)
    if txt_id is not None and txt_id in assets.txt_id_to_col:
        return rank_text_to_video(query, assets=assets)

    if assets.video_embeds is None:
        return (
            [],
            "Free-text needs `video_embeds.npy` on the demo dataset (see README). "
            "Preset captions still work instantly via the official score matrix.",
        )

    dev = device or free_text_device()
    vec = encode_text(query, device=dev)
    results, mode = rank_text_to_video(query, text_vec=vec, assets=assets)
    elapsed = last_encode_seconds()
    load_sec = last_load_seconds()
    if dev == "cpu":
        parts = ["free text (CPU encoder + precomputed video embeddings)"]
        if elapsed is not None:
            parts.append(f"{elapsed:.1f}s total")
        if load_sec is not None and load_sec > 1.0:
            parts.append(f"model load {load_sec:.1f}s")
        mode = " · ".join(parts)
    return results, mode


def _rank_text_safe(query: str, *, device: str | None = None):
    try:
        return _rank_text(query, device=device)
    except Exception as e:
        return [], f"Text encoder error: {e}"


def _rank_text_zero_gpu(query: str):
    return _rank_text(query, device="cuda")


if spaces is not None and _space_has_zero_gpu():
    _rank_text_zero_gpu = spaces.GPU(duration=300)(_rank_text_zero_gpu)
else:
    _rank_text_zero_gpu = None


def _needs_live_encoder(query: str) -> bool:
    assets = load_assets()
    if assets.video_embeds is None:
        return False
    txt_id = find_txt_id_for_query(assets, query)
    return txt_id is None or txt_id not in assets.txt_id_to_col


def _run_free_text_rank(query: str):
    if _rank_text_zero_gpu is not None and not hf_space_is_cpu_only():
        return _rank_text_zero_gpu(query)
    return _rank_text_safe(query, device="cpu")


def run_t2v_search(query: str, progress=gr.Progress(track_tqdm=False)):
    query = (query or "").strip()
    if not query:
        return empty_t2v_outputs()

    if _needs_live_encoder(query):
        progress(0, desc="Encoding query on CPU (first time may take a few minutes)…")
        if hf_space_is_cpu_only():
            progress(0.05, desc=cpu_free_text_eta_note()[:120] + "…")

    results, mode = (
        _run_free_text_rank(query)
        if _needs_live_encoder(query)
        else _rank_text_safe(query)
    )

    if not results:
        return (
            mode,
            [],
            None,
            None,
            None,
            mode,
            gr.update(choices=[], value=None),
            None,
        )

    assets = load_assets()
    item = results[0]
    f0, f1, f2 = t2v_frames_for_item(assets, item)
    choices = t2v_choices(results)
    return (
        f"**Mode:** {mode}",
        results,
        f0,
        f1,
        f2,
        t2v_detail_md(item, 1, mode),
        gr.update(choices=choices, value=choices[0] if choices else None),
        preview_image(assets, item["vis_id"]),
    )


def on_t2v_pick(choice: str | None, results: list, mode_note: str):
    if not results:
        return None, None, None, "No results.", None
    assets = load_assets()
    idx = parse_t2v_choice(choice)
    idx = max(0, min(idx, len(results) - 1))
    item = results[idx]
    f0, f1, f2 = t2v_frames_for_item(assets, item)
    mode = (mode_note or "").replace("**Mode:** ", "") or "retrieval"
    return (
        f0,
        f1,
        f2,
        t2v_detail_md(item, idx + 1, mode),
        preview_image(assets, item["vis_id"]),
    )


def search_preset(choice: str):
    _, caption = parse_preset_choice(choice)
    if not caption:
        return empty_t2v_outputs()
    return run_t2v_search(caption)


def _v2t_ground_truth(assets, vis_id: str) -> str | None:
    row = segment_row(assets, vis_id)
    if row is None:
        return None
    return str(row["narration"])


def run_v2t(vis_id: str):
    vis_id = (vis_id or "").strip()
    if not vis_id:
        return empty_v2t_outputs()

    assets = load_assets()
    if vis_id not in assets.vis_id_to_row:
        return (
            None,
            None,
            None,
            f"Clip **`{vis_id}`** is not in the test similarity matrix.",
            "<p style='color:#52525b!important;'>No ranked captions.</p>",
            "",
        )

    results = rank_video_to_text(vis_id)
    gt = _v2t_ground_truth(assets, vis_id)
    f0, f1, f2 = frame_images_for_gradio(assets, vis_id, 3)
    if not results:
        return (
            f0,
            f1,
            f2,
            v2t_clip_header_md(assets, vis_id),
            "<p style='color:#52525b!important;'>No ranked captions for this clip.</p>",
            vis_id,
        )

    return (
        f0,
        f1,
        f2,
        v2t_clip_header_md(assets, vis_id),
        v2t_predictions_html(results, ground_truth=gt),
        vis_id,
    )


def load_random_v2t(current_vis_id: str | None):
    try:
        vis_id = pick_random_v2t_clip(exclude=current_vis_id)
    except RuntimeError as err:
        return (
            None,
            None,
            None,
            f"**Video → Text unavailable:** {err}",
            "",
            "",
        )
    return run_v2t(vis_id)


_CPU_NOTE = (
    "\n\n> **CPU Space (free tier):** presets and Video→Text are instant. "
    "Custom text uses a live encoder on CPU — **~2–4 min** the first time, "
    "**~5–20 s** after that (needs `video_embeds.npy` on the demo dataset)."
    if hf_space_is_cpu_only()
    else ""
)

INTRO = f"""
**EPIC-KITCHENS-100 Multi-Instance Retrieval** — [AVION ViT-L + SMS](https://huggingface.co/jsurrea/avion-vitl-ek100-sms) (**69.68 nDCG AVG**).

**Text → Video:** describe an action → see the best matching kitchen clips.  
**Video → Text:** a random kitchen clip → ranked captions from the frames (benchmark scores).{_CPU_NOTE}
"""

_p0 = preset_choices()

with gr.Blocks(title="EK-100 MIR Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Tab("Text → Video"):
        if hf_space_is_cpu_only():
            gr.Markdown(cpu_free_text_eta_note())

        with gr.Row():
            preset = gr.Dropdown(choices=_p0, label="Preset query", value=_p0[0] if _p0 else None)
            free = gr.Textbox(
                label="Or type your own (CPU if not a preset)",
                placeholder='e.g. peel potato, open dishwasher',
                lines=1,
                scale=2,
            )
            t2v_go = gr.Button("Search", variant="primary", scale=0)

        t2v_mode = gr.Markdown()
        t2v_state = gr.State([])

        with gr.Row():
            t2v_pick = gr.Radio(label="Top segments (click to preview frames)", choices=[], value=None)
            t2v_thumb = gr.Image(label="Preview", height=200, interactive=False)

        with gr.Row():
            t2v_f0 = gr.Image(label="Start", height=180, interactive=False)
            t2v_f1 = gr.Image(label="Middle", height=180, interactive=False)
            t2v_f2 = gr.Image(label="End", height=180, interactive=False)

        t2v_detail = gr.Markdown()

        t2v_outputs = [t2v_mode, t2v_state, t2v_f0, t2v_f1, t2v_f2, t2v_detail, t2v_pick, t2v_thumb]

        t2v_go.click(run_t2v_search, inputs=free, outputs=t2v_outputs)
        free.submit(run_t2v_search, inputs=free, outputs=t2v_outputs)
        preset.change(search_preset, inputs=preset, outputs=t2v_outputs)
        t2v_pick.change(
            on_t2v_pick,
            inputs=[t2v_pick, t2v_state, t2v_mode],
            outputs=[t2v_f0, t2v_f1, t2v_f2, t2v_detail, t2v_thumb],
        )

    with gr.Tab("Video → Text"):
        gr.Markdown(
            "A **random** test clip loads each time. Ranked captions are predicted from the three frames."
        )
        v2t_random_btn = gr.Button("🎲 Another random clip", variant="primary")
        v2t_vis_state = gr.State("")

        with gr.Row():
            v2t_f0 = gr.Image(label="Start", height=160, interactive=False)
            v2t_f1 = gr.Image(label="Middle", height=160, interactive=False)
            v2t_f2 = gr.Image(label="End", height=160, interactive=False)

        v2t_clip_md = gr.Markdown()
        v2t_preds = gr.HTML()

        v2t_outputs = [v2t_f0, v2t_f1, v2t_f2, v2t_clip_md, v2t_preds, v2t_vis_state]

        v2t_random_btn.click(load_random_v2t, inputs=v2t_vis_state, outputs=v2t_outputs)

    demo.load(search_preset, inputs=preset, outputs=t2v_outputs)
    demo.load(load_random_v2t, inputs=v2t_vis_state, outputs=v2t_outputs)

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_api=False,
        _frontend=False,
    )
