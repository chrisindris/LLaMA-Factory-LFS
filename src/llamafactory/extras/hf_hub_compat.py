# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shims for huggingface_hub API drift that otherwise break Transformers import.

Transformers 5.x does ``from huggingface_hub import is_offline_mode``. That helper
was added in huggingface_hub 1.2 (briefly as ``offline_mode``) and is missing from
0.x and 1.0/1.1. User-site installs can also shadow a container copy. Patch the
symbol onto the already-imported hub module before Transformers is loaded.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any


_ENV_TRUE = {"1", "ON", "YES", "TRUE"}


def _is_env_true(value: str | None) -> bool:
    return bool(value) and value.upper() in _ENV_TRUE


def _offline_from_env() -> bool:
    return _is_env_true(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"))


def _has_attr(obj: Any, name: str) -> bool:
    try:
        getattr(obj, name)
    except Exception:
        return False
    return True


def _get_constants(hub: Any) -> Any | None:
    if _has_attr(hub, "constants"):
        return hub.constants

    # Lazy huggingface_hub packages do not expose the constants submodule via getattr.
    module_name = getattr(hub, "__name__", None)
    if not module_name or getattr(hub, "__file__", None) is None:
        return None
    try:
        return importlib.import_module(f"{module_name}.constants")
    except Exception:
        return None


def _offline_checker_from_constants(constants: Any) -> Callable[[], bool] | None:
    for name in ("is_offline_mode", "offline_mode"):
        candidate = getattr(constants, name, None)
        if callable(candidate):
            return candidate
    if _has_attr(constants, "HF_HUB_OFFLINE"):
        return lambda: bool(constants.HF_HUB_OFFLINE)
    return None


def _resolve_offline_checker(hub: Any) -> Callable[[], bool]:
    for name in ("is_offline_mode", "offline_mode"):
        candidate = getattr(hub, name, None)
        if callable(candidate):
            return candidate

    constants = _get_constants(hub)
    if constants is not None:
        checker = _offline_checker_from_constants(constants)
        if checker is not None:
            return checker

    return _offline_from_env


def _set_if_missing(obj: Any, name: str, value: Any) -> None:
    if not _has_attr(obj, name):
        setattr(obj, name, value)


def apply_huggingface_hub_compat(hub: Any | None = None) -> None:
    r"""Expose ``is_offline_mode`` on huggingface_hub when the installed copy lacks it.

    Safe to call more than once. ``hub`` is the module to patch; omit it to import
    ``huggingface_hub`` (or no-op if that package is not installed).
    """
    if hub is None:
        try:
            import huggingface_hub as hub
        except ImportError:
            return

    checker = _resolve_offline_checker(hub)
    _set_if_missing(hub, "is_offline_mode", checker)
    _set_if_missing(hub, "offline_mode", getattr(hub, "is_offline_mode", checker))

    constants = _get_constants(hub)
    if constants is not None:
        _set_if_missing(constants, "is_offline_mode", getattr(hub, "is_offline_mode", checker))
        _set_if_missing(constants, "offline_mode", getattr(hub, "is_offline_mode", checker))


apply_huggingface_hub_compat()
