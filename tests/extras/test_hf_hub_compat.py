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

import sys
import types

from llamafactory.extras.hf_hub_compat import apply_huggingface_hub_compat


def _hub_module(**attrs):
    module = types.ModuleType("huggingface_hub")
    for name, value in attrs.items():
        setattr(module, name, value)
    return module


def test_injects_is_offline_mode_when_missing(monkeypatch):
    hub = _hub_module()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    apply_huggingface_hub_compat(hub)

    assert callable(hub.is_offline_mode)
    assert hub.is_offline_mode() is False


def test_aliases_offline_mode_helper():
    def offline_mode() -> bool:
        return True

    hub = _hub_module(offline_mode=offline_mode)

    apply_huggingface_hub_compat(hub)

    assert hub.is_offline_mode is offline_mode
    assert hub.is_offline_mode() is True


def test_does_not_overwrite_existing_helper():
    def is_offline_mode() -> bool:
        return True

    hub = _hub_module(is_offline_mode=is_offline_mode)

    apply_huggingface_hub_compat(hub)

    assert hub.is_offline_mode is is_offline_mode


def test_uses_constants_hf_hub_offline():
    constants = types.SimpleNamespace(HF_HUB_OFFLINE=True)
    hub = _hub_module(constants=constants)

    apply_huggingface_hub_compat(hub)

    assert hub.is_offline_mode() is True
    assert constants.is_offline_mode() is True


def test_env_fallback(monkeypatch):
    hub = _hub_module()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    apply_huggingface_hub_compat(hub)

    assert hub.is_offline_mode() is True


def test_named_import_works_after_patch(monkeypatch):
    original = sys.modules.get("huggingface_hub")
    hub = _hub_module()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    try:
        apply_huggingface_hub_compat(hub)
        from huggingface_hub import is_offline_mode

        assert is_offline_mode() is False
    finally:
        if original is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = original


def test_apply_is_idempotent():
    def is_offline_mode() -> bool:
        return False

    hub = _hub_module(is_offline_mode=is_offline_mode)
    apply_huggingface_hub_compat(hub)
    apply_huggingface_hub_compat(hub)
    assert hub.is_offline_mode is is_offline_mode


def test_lazy_getattr_missing_symbol(monkeypatch):
    hub = _hub_module()

    def _getattr(name: str):
        raise AttributeError(f"No huggingface_hub attribute {name}")

    hub.__getattr__ = _getattr
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    apply_huggingface_hub_compat(hub)

    assert hub.is_offline_mode() is False
