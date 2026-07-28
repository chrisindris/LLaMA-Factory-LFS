# Copyright 2025 the LlamaFactory team.
#
# Multi-backend H5 image resolution for LLaMA-Factory multimodal training.
# Dispatches among ScanNet (JPEG-bytes H5), Spatial-SSRL, and 3DThinker-10k packs.

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Union

from PIL import Image

# Process-local stores (rebuilt after fork)
_SPATIAL_STORE = None
_SPATIAL_PID: Optional[int] = None
_THINKER_STORE = None
_THINKER_PID: Optional[int] = None

_SCANNET_RE = re.compile(r"(.*)(scene\d+_\d+).*color/(.*\.jpe?g)", re.IGNORECASE)


def _get_spatial_store():
    global _SPATIAL_STORE, _SPATIAL_PID
    pid = os.getpid()
    if _SPATIAL_STORE is None or _SPATIAL_PID != pid:
        from .spatial_ssrl_h5 import SpatialSSRLH5ImageStore

        if _SPATIAL_STORE is not None:
            try:
                _SPATIAL_STORE.close()
            except Exception:
                pass
        _SPATIAL_STORE = SpatialSSRLH5ImageStore()
        _SPATIAL_PID = pid
    return _SPATIAL_STORE


def _get_thinker_store():
    global _THINKER_STORE, _THINKER_PID
    pid = os.getpid()
    if _THINKER_STORE is None or _THINKER_PID != pid:
        from .thinker10k_h5 import Thinker10KH5ImageStore

        if _THINKER_STORE is not None:
            try:
                _THINKER_STORE.close()
            except Exception:
                pass
        _THINKER_STORE = Thinker10KH5ImageStore()
        _THINKER_PID = pid
    return _THINKER_STORE


def looks_like_scannet_path(path: str) -> bool:
    return bool(_SCANNET_RE.search(path.replace("\\", "/")))


def looks_like_spatial_ssrl_path(path: str) -> bool:
    p = path.replace("\\", "/")
    markers = (
        "coldstart_SFT_images/",
        "coldstart_SFT/",
        "images/crop/",
        "images/depth/",
        "images/flip/",
        "images/position/",
        "images/shuffle/",
    )
    if any(m in p for m in markers):
        return True
    for root in ("crop/", "depth/", "flip/", "position/", "shuffle/", "coldstart_SFT/"):
        if p.startswith(root):
            return True
    return False


def looks_like_thinker10k_path(path: str) -> bool:
    p = path.replace("\\", "/")
    return "other_all_image_resize/" in p


def can_resolve_h5_image(path: str) -> bool:
    """Cheap probe: True if path is likely resolvable via an H5 backend."""
    if not isinstance(path, str):
        return False
    if looks_like_scannet_path(path):
        # Path-derived prefix may be from another cluster; accept if scene packs
        # exist under SCANNET_H5_DIR (or the path prefix). Lazy decode still goes
        # through retrieve_image which remaps the root.
        try:
            from .h5py_data import _resolve_scannet_scene_path, DEFAULT_SCANNET_H5_DIR, ENV_SCANNET_H5_DIR
            import re as _re

            m = _re.search(r"(.*)(scene\d+_\d+).*color/(.*\.jpe?g)", path.replace("\\", "/"), _re.IGNORECASE)
            if m:
                prefix, scene = m.group(1), m.group(2)
                try:
                    _resolve_scannet_scene_path(Path(prefix), scene)
                    return True
                except FileNotFoundError:
                    # Still return True if SCANNET_H5_DIR root exists: packs may appear later
                    # and converter needs to keep the path string for lazy load.
                    root = Path(os.environ.get(ENV_SCANNET_H5_DIR, DEFAULT_SCANNET_H5_DIR))
                    return root.is_dir()
        except Exception:
            pass
        return True
    if looks_like_thinker10k_path(path):
        try:
            return _get_thinker_store().contains(path)
        except Exception:
            return False
    if looks_like_spatial_ssrl_path(path):
        try:
            return _get_spatial_store().contains(path)
        except Exception:
            return False
    # Fall back to store probes (covers media_dir-joined absolute paths)
    try:
        if _get_spatial_store().contains(path):
            return True
    except Exception:
        pass
    try:
        if _get_thinker_store().contains(path):
            return True
    except Exception:
        pass
    return False


def resolve_h5_image(path: str) -> Union[bytes, Image.Image]:
    """
    Resolve an image path via H5 backends.

    Returns:
        - bytes (JPEG) for ScanNet-style packs
        - PIL.Image.Image (RGB) for Spatial-SSRL and 3DThinker-10k packs

    Raises:
        FileNotFoundError / KeyError / ValueError if the path cannot be resolved.
    """
    if not isinstance(path, str):
        raise TypeError(f"Expected image path str, got {type(path)}")

    # Prefer explicit pattern matches to avoid opening wrong stores.
    if looks_like_scannet_path(path):
        from .h5py_data import retrieve_image

        return retrieve_image(image_path=path)

    if looks_like_thinker10k_path(path):
        return _get_thinker_store().get_pil(path)

    if looks_like_spatial_ssrl_path(path):
        return _get_spatial_store().get_pil(path)

    # media_dir-joined or ambiguous: try stores, then ScanNet
    try:
        store = _get_spatial_store()
        if store.contains(path):
            return store.get_pil(path)
    except Exception:
        pass

    try:
        store = _get_thinker_store()
        if store.contains(path):
            return store.get_pil(path)
    except Exception:
        pass

    if looks_like_scannet_path(path) or "scene" in path:
        from .h5py_data import retrieve_image

        return retrieve_image(image_path=path)

    raise FileNotFoundError(f"Could not resolve image via H5 backends: {path}")
