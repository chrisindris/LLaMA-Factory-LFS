# Copyright 2025 the LlamaFactory team.
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
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import sysconfigtool  # noqa: E402


def test_expands_project_dir_token(monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", "/tmp/anywhere")
    assert sysconfigtool.expand_value("${PROJECT_DIR}/containers/x.sif") == "/tmp/anywhere/containers/x.sif"


def test_leaves_unset_token_verbatim(monkeypatch):
    monkeypatch.delenv("PORTABLE_UNSET_TOKEN", raising=False)
    assert sysconfigtool.expand_value("${PORTABLE_UNSET_TOKEN}/x") == "${PORTABLE_UNSET_TOKEN}/x"


def test_non_string_values_pass_through():
    assert sysconfigtool.expand_value(7) == 7
    assert sysconfigtool.expand_value(None) is None


def test_read_expands_portable_section(monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", "/tmp/anywhere")
    value = sysconfigtool.read("PORTABLE", "SIF_FILE")
    assert value == "/tmp/anywhere/containers/llamafactory.sif"


def test_read_all_expands_every_value(monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", "/tmp/anywhere")
    values = sysconfigtool.read_all("PORTABLE")
    assert values
    assert not any("${PROJECT_DIR}" in str(v) for v in values.values())


def test_existing_sections_unchanged(monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", "/tmp/anywhere")
    assert sysconfigtool.read("TRILLIUM", "BEST_GPU") == "h100"
    assert sysconfigtool.read("TRILLIUM", "HF_HOME") == "/scratch/indrisch/huggingface/hub"


def test_missing_system_returns_empty():
    assert sysconfigtool.read_all("NO_SUCH_CLUSTER") == {}
    assert sysconfigtool.read("NO_SUCH_CLUSTER", "HF_HOME") is None


@pytest.mark.parametrize("key", ["HF_HOME", "SIF_FILE", "VENV_LLAMAFACTORY", "MEDIA_DIR"])
def test_portable_keys_present(key, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", "/tmp/anywhere")
    assert sysconfigtool.read("PORTABLE", key) is not None
