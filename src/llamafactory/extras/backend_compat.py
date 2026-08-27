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

"""Treat missing accelerator native libs as "backend not available".

Transformers 5.x ``is_torch_npu_available()`` does a bare ``import torch_npu``
when the Python package is installed. On a GPU/CPU node the package can be
present while ``libascend_hal.so`` is not, which raises ``ImportError`` during
``from transformers import PreTrainedModel`` (via flash-attn NPU hooks). Catch
that and the same pattern for other optional device backends.
"""

from __future__ import annotations

import os
import shutil
from ctypes.util import find_library
from functools import lru_cache, wraps
from typing import Any


_PATCH_ATTR = "__llamafactory_backend_compat__"

# Checkers that may import a native extension (torch_npu, torch_mlu, ...).
_BACKEND_CHECKERS = (
    "is_torch_hpu_available",
    "is_torch_mlu_available",
    "is_torch_musa_available",
    "is_torch_neuron_available",
    "is_torch_npu_available",
    "is_torch_xla_available",
)

_ASCEND_HAL_PATHS = (
    "/usr/local/Ascend/driver/lib64/driver/libascend_hal.so",
    "/usr/local/Ascend/driver/lib64/libascend_hal.so",
    "/usr/lib64/libascend_hal.so",
)


def _ascend_runtime_available() -> bool:
    if shutil.which("npu-smi"):
        return True
    if find_library("ascend_hal"):
        return True
    return any(os.path.isfile(path) for path in _ASCEND_HAL_PATHS)


def _wrap_backend_checker(fn: Any, name: str) -> Any:
    @wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> bool:
        if name == "is_torch_npu_available" and not _ascend_runtime_available():
            return False
        try:
            return bool(fn(*args, **kwargs))
        except Exception:
            return False

    cached = lru_cache(maxsize=None)(wrapped)
    setattr(cached, _PATCH_ATTR, True)
    return cached


def _patch_module(module: Any) -> None:
    for name in _BACKEND_CHECKERS:
        fn = getattr(module, name, None)
        if fn is None or getattr(fn, _PATCH_ATTR, False):
            continue
        setattr(module, name, _wrap_backend_checker(fn, name))


def apply_optional_backend_compat(
    import_utils_module: Any | None = None,
    utils_module: Any | None = None,
) -> None:
    r"""Wrap Transformers device-availability helpers so a missing .so is False.

    Safe to call more than once. Omit the module arguments to patch the installed
    ``transformers.utils.import_utils`` and ``transformers.utils`` re-exports.
    """
    if import_utils_module is None:
        try:
            import transformers.utils.import_utils as import_utils_module
        except Exception:
            return

    _patch_module(import_utils_module)

    if utils_module is None:
        try:
            import transformers.utils as utils_module
        except Exception:
            utils_module = None

    if utils_module is not None:
        for name in _BACKEND_CHECKERS:
            patched = getattr(import_utils_module, name, None)
            if patched is not None:
                setattr(utils_module, name, patched)


apply_optional_backend_compat()
