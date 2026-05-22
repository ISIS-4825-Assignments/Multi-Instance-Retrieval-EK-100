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
    segment_video_for_gradio,
    t2v_choices,
    t2v_frames_for_item,
    t2v_status_html,
    v2t_choices,
    v2t_clip_meta_html,
    v2t_ground_truth_html,
)

_DEMO_CSS = (SPACE_DIR / "demo.css").read_text(encoding="utf-8")


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
            "Free-text needs video_embeds.npy on the demo dataset. Presets still work.",
        )

    dev = device or free_text_device()
    vec = encode_text(query, device=dev)
    return rank_text_to_video(query, text_vec=vec, assets=assets)


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


def _t2v_pack(query: str, results: list, mode: str, *, error: bool = False):
    import gradio as gr

    if error or not results:
        return (
            t2v_status_html(mode, [], query, error=error or bool(mode)),
            [],
            None,
            None,
            None,
            gr.update(choices=[], value=None),
            None,
            "",
        )

    assets = load_assets()
    item = results[0]
    f0, f1, f2 = t2v_frames_for_item(assets, item)
    choices = t2v_choices(results)
    return (
        t2v_status_html("", results, query),
        results,
        f0,
        f1,
        f2,
        gr.update(choices=choices, value=choices[0] if choices else None),
        segment_video_for_gradio(assets, item["vis_id"]),
        mode,
    )


def run_t2v_search(query: str, progress=gr.Progress(track_tqdm=False)):
    query = (query or "").strip()
    if not query:
        return empty_t2v_outputs()

    if _needs_live_encoder(query):
        progress(0, desc="Encoding query (~90s first time, ~5s after)…")

    results, mode = (
        _run_free_text_rank(query)
        if _needs_live_encoder(query)
        else _rank_text_safe(query)
    )

    is_err = bool(mode) and not results and (
        "error" in mode.lower() or "unavailable" in mode.lower() or "needs" in mode.lower()
    )
    return _t2v_pack(query, results, mode, error=is_err)


def on_t2v_pick(choice: str | None, results: list):
    if not results:
        return None, None, None, None
    assets = load_assets()
    idx = parse_t2v_choice(choice)
    idx = max(0, min(idx, len(results) - 1))
    item = results[idx]
    f0, f1, f2 = t2v_frames_for_item(assets, item)
    return (
        f0,
        f1,
        f2,
        segment_video_for_gradio(assets, item["vis_id"]),
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


def _v2t_pack(vis_id: str, results: list, *, error_msg: str | None = None):
    import gradio as gr

    assets = load_assets()
    if error_msg:
        return (
            None,
            None,
            None,
            f'<div class="ek-status-err">{error_msg}</div>',
            "",
            None,
            gr.update(choices=[], value=None),
            [],
            vis_id,
        )

    gt = _v2t_ground_truth(assets, vis_id)
    f0, f1, f2 = frame_images_for_gradio(assets, vis_id, 3)
    video = segment_video_for_gradio(assets, vis_id)
    choices = v2t_choices(results, gt)
    return (
        f0,
        f1,
        f2,
        v2t_clip_meta_html(assets, vis_id),
        v2t_ground_truth_html(gt),
        video,
        gr.update(choices=choices, value=choices[0] if choices else None),
        results,
        vis_id,
    )


def run_v2t(vis_id: str):
    vis_id = (vis_id or "").strip()
    if not vis_id:
        return empty_v2t_outputs()

    assets = load_assets()
    if vis_id not in assets.vis_id_to_row:
        return _v2t_pack(
            vis_id,
            [],
            error_msg=f"Clip {vis_id} is not in the test set.",
        )

    results = rank_video_to_text(vis_id)
    if not results:
        return _v2t_pack(vis_id, [], error_msg="No captions ranked for this clip.")
    return _v2t_pack(vis_id, results)


def load_random_v2t(current_vis_id: str | None):
    try:
        vis_id = pick_random_v2t_clip(exclude=current_vis_id)
    except RuntimeError as err:
        return _v2t_pack("", [], error_msg=str(err))
    return run_v2t(vis_id)


INTRO = """
<div class="ek-intro">
  <p class="ek-intro-lead">
    <a href="https://huggingface.co/jsurrea/avion-vitl-ek100-sms">AVION ViT-L + SMS</a>
    on EPIC-KITCHENS-100 (69.68 nDCG AVG).
  </p>
  <p class="ek-intro-line"><span class="ek-intro-tab">Text → Video</span> — describe an action and browse ranked clips.</p>
  <p class="ek-intro-line"><span class="ek-intro-tab">Video → Text</span> — watch a random clip and see predicted captions.</p>
</div>
"""

_p0 = preset_choices()

with gr.Blocks(title="EK-100 MIR Demo", theme=gr.themes.Soft(), css=_DEMO_CSS) as demo:
    gr.HTML(INTRO)

    with gr.Tab("Text → Video"):
        if hf_space_is_cpu_only():
            gr.HTML(cpu_free_text_eta_note())

        with gr.Row():
            preset = gr.Dropdown(
                choices=_p0,
                label="Preset query (instant)",
                value=_p0[0] if _p0 else None,
                scale=2,
            )
            free = gr.Textbox(
                label="Or type your own",
                placeholder="e.g. peel potato, open dishwasher",
                lines=1,
                scale=2,
            )
            t2v_go = gr.Button("Search", variant="primary", scale=0)

        t2v_state = gr.State([])
        t2v_mode_state = gr.State("")

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                t2v_video = gr.Video(
                    label="Segment video",
                    height=280,
                    interactive=False,
                )
                with gr.Row():
                    t2v_f0 = gr.Image(label="Start", height=150, interactive=False)
                    t2v_f1 = gr.Image(label="Middle", height=150, interactive=False)
                    t2v_f2 = gr.Image(label="End", height=150, interactive=False)
            with gr.Column(scale=4, elem_classes=["ek-bento-panel"]):
                t2v_status = gr.HTML()
                gr.HTML('<p class="ek-section-title">Top segments</p>')
                t2v_pick = gr.Radio(
                    label=None,
                    choices=[],
                    value=None,
                    show_label=False,
                    elem_classes=["ek-chip-radio"],
                )

        t2v_outputs = [
            t2v_status,
            t2v_state,
            t2v_f0,
            t2v_f1,
            t2v_f2,
            t2v_pick,
            t2v_video,
            t2v_mode_state,
        ]

        t2v_go.click(run_t2v_search, inputs=free, outputs=t2v_outputs)
        free.submit(run_t2v_search, inputs=free, outputs=t2v_outputs)
        preset.change(search_preset, inputs=preset, outputs=t2v_outputs)
        t2v_pick.change(
            on_t2v_pick,
            inputs=[t2v_pick, t2v_state],
            outputs=[t2v_f0, t2v_f1, t2v_f2, t2v_video],
        )

    with gr.Tab("Video → Text"):
        gr.HTML(
            '<p class="ek-hint">Random test clip on load. '
            "Another clip loads a new sample.</p>"
        )
        v2t_random_btn = gr.Button("Another random clip", variant="primary")
        v2t_vis_state = gr.State("")

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                v2t_video = gr.Video(
                    label="Segment video",
                    height=280,
                    interactive=False,
                )
                with gr.Row():
                    v2t_f0 = gr.Image(label="Start", height=140, interactive=False)
                    v2t_f1 = gr.Image(label="Middle", height=140, interactive=False)
                    v2t_f2 = gr.Image(label="End", height=140, interactive=False)
            with gr.Column(scale=4, elem_classes=["ek-bento-panel"]):
                v2t_meta = gr.HTML()
                v2t_gt = gr.HTML()
                gr.HTML('<p class="ek-section-title">Predicted captions</p>')
                v2t_pick = gr.Radio(
                    label=None,
                    choices=[],
                    value=None,
                    show_label=False,
                    elem_classes=["ek-chip-radio"],
                )

        v2t_results_state = gr.State([])

        v2t_outputs = [
            v2t_f0,
            v2t_f1,
            v2t_f2,
            v2t_meta,
            v2t_gt,
            v2t_video,
            v2t_pick,
            v2t_results_state,
            v2t_vis_state,
        ]

        v2v_all_outputs = t2v_outputs + v2t_outputs

        def on_startup(preset_val, v2t_vis):
            return search_preset(preset_val) + load_random_v2t(v2t_vis)

        v2t_random_btn.click(load_random_v2t, inputs=v2t_vis_state, outputs=v2t_outputs)
        demo.load(on_startup, inputs=[preset, v2t_vis_state], outputs=v2v_all_outputs)

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_api=False,
    )
