# Copyright 2025 the LlamaFactory team.
#
# 3DThinker-10k packed H5 image store (fixed 480x640 WHC uint8 + JSON index).
# Layout: data/3DThinker-10K/3dthinker10k_h5_dataloader_spec.md

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]

DEFAULT_ROOT = "/scratch/indrisch/3DThinker10K_images_h5"
DEFAULT_H5_NAME = "3dthinker10k_images.h5"
DEFAULT_INDEX_NAME = "3dthinker10k_images_index.json"
ENV_ROOT = "THINKER10K_H5_DIR"


def default_root() -> str:
    return os.environ.get(ENV_ROOT, DEFAULT_ROOT)


def to_index_key(annotation_path: str) -> str:
    """Map annotation image path to the H5 index key."""
    p = annotation_path.strip().replace("\\", "/").lstrip("./")
    if p.startswith("data/"):
        p = p[len("data/") :]
    # media_dir may have joined an absolute prefix; keep trailing other_all_image_resize/...
    marker = "other_all_image_resize/"
    idx = p.find(marker)
    if idx > 0:
        p = p[idx:]
    return p


class Thinker10KH5ImageStore:
    """Load 3DThinker-10k images from a single packed H5 file."""

    def __init__(
        self,
        root: Optional[str | os.PathLike] = None,
        h5_name: str = DEFAULT_H5_NAME,
        index_name: str = DEFAULT_INDEX_NAME,
    ) -> None:
        if h5py is None:
            raise ModuleNotFoundError("h5py is required for 3DThinker-10k H5 image loading")

        self.root = Path(root if root is not None else default_root()).resolve()
        self.h5_path = self.root / h5_name
        index_path = self.root / index_name
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing 3DThinker-10k index: {index_path}")
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"Missing 3DThinker-10k H5: {self.h5_path}")

        with index_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        self.index: dict[str, int] = {}
        for k, v in raw.items():
            if str(k).startswith("_"):
                continue
            self.index[str(k)] = int(v)

        self._h5: Optional["h5py.File"] = None
        self._pid = os.getpid()

    def close(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None

    def _ensure_pid(self) -> None:
        pid = os.getpid()
        if pid != self._pid:
            self.close()
            self._pid = pid

    def _get_h5(self) -> "h5py.File":
        self._ensure_pid()
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def contains(self, path: str) -> bool:
        return to_index_key(path) in self.index

    def get_row(self, path: str) -> int:
        key = to_index_key(path)
        try:
            return self.index[key]
        except KeyError as e:
            raise KeyError(f"Image key not in 3DThinker-10k H5 index: {path!r} -> {key!r}") from e

    def get_pil(self, path: str) -> Image.Image:
        row = self.get_row(path)
        h5 = self._get_h5()
        image_whc = np.asarray(h5["images"][row], dtype=np.uint8)  # (480, 640, 3) WHC
        image_hwc = np.transpose(image_whc, (1, 0, 2))  # (640, 480, 3)
        return Image.fromarray(image_hwc, mode="RGB")

    def get_batch_pil(self, paths: list[str]) -> list[Image.Image]:
        return [self.get_pil(p) for p in paths]

    def validate_shapes(self) -> dict[str, Any]:
        h5 = self._get_h5()
        shape = tuple(h5["images"].shape)
        return {
            "h5_shape": shape,
            "index_size": len(self.index),
            "max_index": max(self.index.values()) if self.index else -1,
        }
