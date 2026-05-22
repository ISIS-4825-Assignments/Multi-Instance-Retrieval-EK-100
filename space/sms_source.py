"""Resolve AVION/SMS source tree from Hugging Face (cached) or SMS_LOSS_ROOT."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from config import HF_MODEL_REPO, HF_SMS_SOURCE_REPO


def _has_avion_tree(root: Path) -> bool:
    return (root / "avion" / "models" / "model_clip.py").is_file()


@lru_cache(maxsize=1)
def resolve_sms_root() -> Path:
    """
    Directory whose parent is on sys.path so `import avion` works.
    Prefer SMS_LOSS_ROOT; otherwise snapshot `avion/**` from Hugging Face.
    """
    env = os.environ.get("SMS_LOSS_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if _has_avion_tree(root):
            return root
        raise FileNotFoundError(f"SMS_LOSS_ROOT={root} does not contain avion/models/model_clip.py")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise ImportError(
            "huggingface_hub is required to download the AVION source tree. "
            "Install with: pip install huggingface_hub"
        ) from err

    repo = HF_SMS_SOURCE_REPO
    local_only = os.environ.get("EK100_SMS_LOCAL_ONLY", "").lower() in ("1", "true", "yes")
    cached = Path(
        snapshot_download(
            repo_id=repo,
            allow_patterns=["avion/**"],
            local_files_only=local_only,
        )
    )
    if not _has_avion_tree(cached):
        raise FileNotFoundError(
            f"Model repo `{repo}` has no `avion/` package on the Hub. "
            "A maintainer must upload it once, e.g. "
            "uploading `avion/` to the HF model repo (see competition README / maintainer notes)."
        )
    return cached


def ensure_sms_on_path() -> Path:
    root = resolve_sms_root()
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root
