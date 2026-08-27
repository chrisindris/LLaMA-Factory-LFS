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

import types

from llamafactory.extras.backend_compat import apply_optional_backend_compat


def _raising_npu_available(check_device=False) -> bool:
    raise ImportError("libascend_hal.so: cannot open shared object file: No such file or directory")


def test_npu_import_error_becomes_false(monkeypatch):
    monkeypatch.setattr("llamafactory.extras.backend_compat._ascend_runtime_available", lambda: True)
    import_utils = types.SimpleNamespace(is_torch_npu_available=_raising_npu_available)
    utils = types.SimpleNamespace(is_torch_npu_available=_raising_npu_available)

    apply_optional_backend_compat(import_utils, utils)

    assert import_utils.is_torch_npu_available() is False
    assert utils.is_torch_npu_available() is False


def test_does_not_hide_true(monkeypatch):
    monkeypatch.setattr("llamafactory.extras.backend_compat._ascend_runtime_available", lambda: True)
    import_utils = types.SimpleNamespace(is_torch_npu_available=lambda check_device=False: True)

    apply_optional_backend_compat(import_utils)

    assert import_utils.is_torch_npu_available() is True


def test_apply_is_idempotent(monkeypatch):
    monkeypatch.setattr("llamafactory.extras.backend_compat._ascend_runtime_available", lambda: True)
    import_utils = types.SimpleNamespace(is_torch_npu_available=_raising_npu_available)

    apply_optional_backend_compat(import_utils)
    first = import_utils.is_torch_npu_available
    apply_optional_backend_compat(import_utils)

    assert import_utils.is_torch_npu_available is first
    assert first() is False


def test_failed_check_is_cached(monkeypatch):
    calls = {"n": 0}

    def raising(check_device=False) -> bool:
        calls["n"] += 1
        raise ImportError("libascend_hal.so")

    monkeypatch.setattr("llamafactory.extras.backend_compat._ascend_runtime_available", lambda: True)
    import_utils = types.SimpleNamespace(is_torch_npu_available=raising)
    apply_optional_backend_compat(import_utils)

    assert import_utils.is_torch_npu_available() is False
    assert import_utils.is_torch_npu_available() is False
    assert calls["n"] == 1


def test_skips_torch_npu_import_without_ascend_runtime(monkeypatch):
    calls = {"n": 0}

    def raising(check_device=False) -> bool:
        calls["n"] += 1
        raise ImportError("libascend_hal.so")

    monkeypatch.setattr("llamafactory.extras.backend_compat._ascend_runtime_available", lambda: False)
    import_utils = types.SimpleNamespace(is_torch_npu_available=raising)
    apply_optional_backend_compat(import_utils)

    assert import_utils.is_torch_npu_available() is False
    assert calls["n"] == 0
