#!/usr/bin/env python3
"""
Download EPIC-KITCHENS-100 Multi-Instance Retrieval data.

Saves everything under EK100_MIR/ (next to this script by default).
Override with: EK100_MIR_ROOT=/your/path python download_data.py

Data downloaded:
  data/features/     — TBN features (RGB + Flow + Audio, ~24 GB)
  data/annotations/  — retrieval train/test pkl (~11 MB)
  data/MI-MM_data.zip + data/MI-MM/  — S3D features, models, relevancy (~8.6 GB extracted)
  data/JPoSE_data.zip + data/JPoSE/  — JPoSE/MLP models + features (~2.2 GB extracted)
"""

import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path

ROOT = Path(os.environ.get("EK100_MIR_ROOT", Path(__file__).parent.parent / "EK100_MIR"))

DIRS = [
    ROOT / "data" / "features",
    ROOT / "data" / "annotations",
    ROOT / "data" / "MI-MM",
    ROOT / "data" / "JPoSE",
]

ANNOT_BASE = (
    "https://github.com/epic-kitchens/epic-kitchens-100-annotations"
    "/raw/master/retrieval_annotations"
)


def _downloader():
    if shutil.which("wget"):
        return "wget"
    if shutil.which("curl"):
        return "curl"
    return "python"


def _wget_cmd(url, dest, tmp):
    return ["wget", "-q", "--show-progress", "-c", "-O", str(tmp), url]


def _curl_cmd(url, dest, tmp):
    return ["curl", "-L", "-#", "-C", "-", "-o", str(tmp), url]


def download(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        gb = dest.stat().st_size / 1e9
        print(f"  SKIP  {label}  ({gb:.2f} GB already on disk)")
        return

    print(f"  DL    {label} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    tool = _downloader()
    if tool == "wget":
        cmd = _wget_cmd(url, dest, tmp)
    elif tool == "curl":
        cmd = _curl_cmd(url, dest, tmp)
    else:
        _python_download(url, tmp, label)
        tmp.rename(dest)
        print(f"  OK    {label}  ({dest.stat().st_size/1e9:.2f} GB)")
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        sys.exit(f"ERROR: download failed for {label}")

    tmp.rename(dest)
    print(f"  OK    {label}  ({dest.stat().st_size/1e9:.2f} GB)")


def _python_download(url: str, dest: Path, label: str) -> None:
    import urllib.request

    def _progress(count, block, total):
        pct = min(100, count * block * 100 // total) if total > 0 else 0
        mb = count * block / 1e6
        print(f"\r  {pct:3d}%  {mb:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _has_data(folder: Path, subfolders: list[str]) -> bool:
    return any(
        (folder / sub).is_dir()
        and any(f.is_file() for f in (folder / sub).rglob("*"))
        for sub in subfolders
    )


def extract_nested_zip(zip_path: Path, dest_dir: Path, label: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = zip_path.parent / (zip_path.stem + "_outer")
    tmp_dir.mkdir(exist_ok=True)

    print(f"  Inspecting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        entries    = zf.namelist()
        inner_zips = [e for e in entries if e.endswith(".zip")]
        if inner_zips:
            for name in entries:
                zf.extract(name, tmp_dir)
            src = tmp_dir / inner_zips[0]
            print(f"  Found inner zip: {src.name}")
        else:
            src = zip_path

    print(f"  Extracting {src.name} → {dest_dir}")
    result = subprocess.run(["unzip", "-o", "-q", str(src), "-d", str(dest_dir)])
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if result.returncode != 0:
        sys.exit(f"ERROR: extraction failed for {label}")
    print(f"  OK    {label} extracted")


def extract_zip(zip_path: Path, dest_dir: Path, label: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {zip_path.name} → {dest_dir}")
    result = subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(dest_dir)])
    if result.returncode != 0:
        sys.exit(f"ERROR: extraction failed for {label}")
    print(f"  OK    {label} extracted")


def summary() -> None:
    print(f"\n{'='*55}")
    print("  Final layout:")
    print(f"{'='*55}")
    total = 0
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.suffix in (".pkl", ".zip", ".pth", ".tar"):
            gb = path.stat().st_size / 1e9
            total += gb
            print(f"  {path.relative_to(ROOT)}  ({gb:.2f} GB)")
    print(f"\n  Total: {total:.2f} GB")


def main() -> None:
    print(f"\n{'='*55}")
    print("  EK100-MIR Data Downloader")
    print(f"  Root: {ROOT.resolve()}")
    print(f"{'='*55}\n")

    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)

    # ── TBN Features (~24 GB) ──────────────────────────────
    print("── TBN Features (RGB + Flow + Audio) ──────────────────")
    download(
        "https://www.dropbox.com/s/41pgqys8e9i2rjh/features_train.pkl?dl=1",
        ROOT / "data" / "features" / "features_train.pkl",
        "features_train.pkl  (~20.6 GB)",
    )
    download(
        "https://www.dropbox.com/s/1kereto93y09ecz/features_test.pkl?dl=1",
        ROOT / "data" / "features" / "features_test.pkl",
        "features_test.pkl   (~3.0 GB)",
    )

    # ── MI-MM Data (~0.68 GB zip, ~8.6 GB extracted) ───────
    print("\n── MI-MM Data (S3D features + models + relevancy) ─────")
    mimm_zip = ROOT / "data" / "MI-MM_data.zip"
    mimm_dir = ROOT / "data" / "MI-MM"
    download(
        "https://www.dropbox.com/sh/5gl70rk7qznw4cs/AAAjJHVQyMB3BHLbVeGOdVo2a?dl=1",
        mimm_zip,
        "MI-MM_data.zip      (~0.68 GB)",
    )
    if _has_data(mimm_dir, ["dataframes", "features", "models", "relevancy", "resources"]):
        print(f"  SKIP  MI-MM already extracted")
    else:
        extract_nested_zip(mimm_zip, mimm_dir, "MI-MM data")

    # ── JPoSE / MLP Data (~1.64 GB zip, ~2.2 GB extracted) ─
    print("\n── JPoSE / MLP Data (models + video/text features) ────")
    jpose_zip = ROOT / "data" / "JPoSE_data.zip"
    jpose_dir = ROOT / "data" / "JPoSE"
    download(
        "https://www.dropbox.com/s/bs6y50xkl1rbe20/JPoSE_data.zip?dl=1",
        jpose_zip,
        "JPoSE_data.zip      (~1.64 GB)",
    )
    if _has_data(jpose_dir / "data", ["models", "video_features", "text_features"]):
        print(f"  SKIP  JPoSE already extracted")
    else:
        extract_zip(jpose_zip, jpose_dir, "JPoSE data")

    # ── Annotations (~11 MB) ────────────────────────────────
    print("\n── Annotations ─────────────────────────────────────────")
    for fname in ["EPIC_100_retrieval_train.pkl", "EPIC_100_retrieval_test.pkl"]:
        download(
            f"{ANNOT_BASE}/{fname}",
            ROOT / "data" / "annotations" / fname,
            fname,
        )

    summary()
    print(f"\n  Done. Data is at: {ROOT.resolve()}\n")


if __name__ == "__main__":
    main()
