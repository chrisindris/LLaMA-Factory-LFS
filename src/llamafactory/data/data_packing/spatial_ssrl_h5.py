# Copyright 2025 the LlamaFactory team.
#
# Spatial-SSRL packed H5 image store (WHC uint8 canvases + JSON index).
# Layout: data/Spatial-SSRL/h5_dataloader_spec.md

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]

DEFAULT_ROOT = "/scratch/indrisch/Spatial-SSRL_images_h5"
DEFAULT_INDEX_NAME = "spatial_ssrl_images_index.json"
ENV_ROOT = "SPATIALSSRL_H5_DIR"

# Longest-prefix first
_PREFIX_MAP: list[tuple[str, str]] = [
    ("coldstart_SFT_images/", "coldstart_SFT/"),
    ("images/crop/", "crop/"),
    ("images/depth/", "depth/"),
    ("images/flip/", "flip/"),
    ("images/position/", "position/"),
    ("images/shuffle/", "shuffle/"),
]

_KNOWN_ROOTS = (
    "crop/",
    "depth/",
    "flip/",
    "position/",
    "shuffle/",
    "coldstart_SFT/",
)


def default_root() -> str:
    return os.environ.get(ENV_ROOT, DEFAULT_ROOT)


class SpatialSSRLH5ImageStore:
    """Load Spatial-SSRL images from packed H5 using the global JSON index."""

    def __init__(
        self,
        root: Optional[str | os.PathLike] = None,
        index_name: str = DEFAULT_INDEX_NAME,
    ) -> None:
        if h5py is None:
            raise ModuleNotFoundError("h5py is required for Spatial-SSRL H5 image loading")

        self.root = Path(root if root is not None else default_root()).resolve()
        index_path = self.root / index_name
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing Spatial-SSRL global index: {index_path}")

        with index_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Skip reserved keys starting with '_'
        self.index: dict[str, dict[str, Any]] = {
            k: v for k, v in raw.items() if not str(k).startswith("_") and isinstance(v, dict)
        }
        self._files: dict[str, "h5py.File"] = {}
        self._pid = os.getpid()

    def close(self) -> None:
        for handle in self._files.values():
            try:
                handle.close()
            except Exception:
                pass
        self._files.clear()

    def _ensure_pid(self) -> None:
        pid = os.getpid()
        if pid != self._pid:
            self.close()
            self._pid = pid

    def normalize_key(self, path: str) -> str:
        p = path.replace("\\", "/").lstrip("./")

        for src, dst in _PREFIX_MAP:
            if p.startswith(src):
                return dst + p[len(src) :]
            idx = p.find("/" + src)
            if idx >= 0:
                return dst + p[idx + 1 + len(src) :]
            # path may be media_dir-joined without leading slash on src
            idx = p.find(src)
            if idx >= 0 and (idx == 0 or p[idx - 1] == "/"):
                return dst + p[idx + len(src) :]

        for root in _KNOWN_ROOTS:
            if p.startswith(root):
                return p
            idx = p.find("/" + root)
            if idx >= 0:
                return p[idx + 1 :]

        return p

    def contains(self, path: str) -> bool:
        return self.normalize_key(path) in self.index

    def get_entry(self, path: str) -> dict[str, Any]:
        key = self.normalize_key(path)
        try:
            return self.index[key]
        except KeyError as e:
            raise KeyError(f"Image key not in Spatial-SSRL H5 index after normalize: {path!r} -> {key!r}") from e

    def _h5(self, h5file: str) -> "h5py.File":
        self._ensure_pid()
        if h5file not in self._files:
            path = self.root / h5file
            if not path.is_file():
                raise FileNotFoundError(path)
            self._files[h5file] = h5py.File(path, "r")
        return self._files[h5file]

    def get_pil(self, path: str) -> Image.Image:
        entry = self.get_entry(path)
        h5 = self._h5(str(entry["h5file"]))
        canvas = h5["images"][int(entry["h5index"])]  # (max_w, max_h, 3) WHC
        w, h = int(entry["width"]), int(entry["height"])
        whc = np.asarray(canvas[:w, :h, :], dtype=np.uint8)
        hwc = np.transpose(whc, (1, 0, 2))
        return Image.fromarray(hwc, mode="RGB")

    def get_batch_pil(self, paths: list[str]) -> list[Image.Image]:
        return [self.get_pil(p) for p in paths]
