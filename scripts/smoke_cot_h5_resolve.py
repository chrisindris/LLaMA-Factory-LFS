#!/usr/bin/env python3
"""Smoke-test multi-root H5 image resolve for the CoT train mix.

Verifies that a single media_dir is *not* required: Scene30k, Spatial-SSRL, and
3DThinker-10k paths resolve via env roots + h5_image_store path dispatch.

Usage (inside the training container / venv with h5py + project src on PYTHONPATH):

  export SCANNET_H5_DIR=/scratch/indrisch/ScanNet_h5/scans
  export SPATIALSSRL_H5_DIR=/scratch/indrisch/Spatial-SSRL_images_h5
  export THINKER10K_H5_DIR=/scratch/indrisch/3DThinker10K_images_h5
  export PYTHONPATH=/scratch/indrisch/LLaMA-Factory/src:${PYTHONPATH:-}
  python scripts/smoke_cot_h5_resolve.py

Optional: --media-dir /scratch/indrisch/ScanNet_h5 to also probe media_dir-joined paths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image


def _sample_spatial_path() -> str:
    path = Path(
        "/scratch/indrisch/huggingface/hub/"
        "datasets--internlm--Spatial-SSRL-81k/snapshots/"
        "54b82086060a5612f95588b4979446da2282bcd9/SFT-coldstart.json"
    )
    if not path.is_file():
        return "coldstart_SFT_images/img_0.jpg"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = data[0].get("images") if isinstance(data[0], dict) else None
    if not images:
        raise RuntimeError(f"No images in first Spatial row of {path}")
    return images[0]


def _sample_thinker_path() -> str:
    path = Path(__file__).resolve().parents[1] / "data/3DThinker-10K/out/3dthinker10k_cot.jsonl"
    if not path.is_file():
        return "data/other_all_image_resize/among/ball_440/back_115.jpg"
    with path.open("r", encoding="utf-8") as f:
        row = json.loads(f.readline())
    images = row.get("images") or []
    if not images:
        raise RuntimeError(f"No images in first Thinker row of {path}")
    return images[0]


def _sample_scene30k_path() -> str:
    return "ScanNet/scans/scene0000_00/color/0.jpg"


def _check_one(label: str, path: str, media_dir: str | None) -> None:
    from llamafactory.data.data_packing.h5_image_store import (
        can_resolve_h5_image,
        resolve_h5_image,
    )

    print(f"[{label}] path={path!r}")
    if not can_resolve_h5_image(path):
        raise SystemExit(f"[{label}] can_resolve_h5_image failed for original path")
    resolved = resolve_h5_image(path)
    if isinstance(resolved, bytes):
        img = Image.open(BytesIO(resolved))
        kind = f"bytes({len(resolved)})"
    else:
        img = resolved
        kind = "PIL"
    print(f"  resolve original: {kind} size={img.size} mode={img.mode}")

    if media_dir:
        joined = os.path.join(media_dir, path)
        print(f"  joined={joined!r} isfile={os.path.isfile(joined)}")
        if can_resolve_h5_image(joined):
            r2 = resolve_h5_image(joined)
            if isinstance(r2, bytes):
                img2 = Image.open(BytesIO(r2))
            else:
                img2 = r2
            print(f"  resolve joined: size={img2.size} mode={img2.mode}")
        else:
            # Joined ScanNet-style paths may still resolve via original only; not fatal
            # if original worked and join is nonsense under media_dir.
            print("  joined not resolvable via H5 (ok if original worked)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--media-dir",
        default=os.environ.get("MEDIA_DIR", "/scratch/indrisch/ScanNet_h5"),
        help="Optional media_dir to probe join behavior (default: ScanNet_h5)",
    )
    parser.add_argument("--no-media-dir-probe", action="store_true")
    args = parser.parse_args()

    media_dir = None if args.no_media_dir_probe else args.media_dir
    print("env:")
    for k in ("SCANNET_H5_DIR", "SPATIALSSRL_H5_DIR", "THINKER10K_H5_DIR"):
        print(f"  {k}={os.environ.get(k, '(unset; using code defaults)')}")
    print(f"media_dir probe={media_dir!r}")
    print(f"sys.path[0]={sys.path[0]!r}")

    cases = [
        ("Scene30k", _sample_scene30k_path()),
        ("SpatialSSRL_coldstart", _sample_spatial_path()),
        ("3DThinker10k", _sample_thinker_path()),
    ]
    for label, path in cases:
        _check_one(label, path, media_dir)

    print("ALL GOOD: multi-root H5 resolve works without a unified media tree")


if __name__ == "__main__":
    main()
