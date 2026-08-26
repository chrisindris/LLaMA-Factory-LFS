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

import os
from pathlib import Path

import pytest

from llamafactory.data.parser import cache_relative_file_name, expand_dataset_path, get_dataset_list


DATA_DIR = str(Path(__file__).resolve().parents[2] / "data")


@pytest.fixture
def clear_hf_cache_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)


@pytest.mark.runs_on(["cpu", "mps"])
def test_relative_file_name_unchanged(clear_hf_cache_env):
    assert expand_dataset_path("alpaca_en_demo.json") == "alpaca_en_demo.json"


@pytest.mark.runs_on(["cpu", "mps"])
def test_absolute_file_name_unchanged(clear_hf_cache_env):
    assert expand_dataset_path("/abs/path/data.json") == "/abs/path/data.json"


@pytest.mark.runs_on(["cpu", "mps"])
def test_other_env_vars_left_unchanged(clear_hf_cache_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_DIR", "/media")
    assert expand_dataset_path("$MEDIA_DIR/foo.json") == "$MEDIA_DIR/foo.json"


@pytest.mark.runs_on(["cpu", "mps"])
def test_expand_hf_hub_cache(clear_hf_cache_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/hub")
    assert expand_dataset_path("${HF_HUB_CACHE}/datasets--x/file.parquet") == "/tmp/hub/datasets--x/file.parquet"
    assert expand_dataset_path("$HF_HUB_CACHE/datasets--x/file.parquet") == "/tmp/hub/datasets--x/file.parquet"


@pytest.mark.runs_on(["cpu", "mps"])
def test_expand_hf_home(clear_hf_cache_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HOME", "/tmp/home")
    assert expand_dataset_path("${HF_HOME}/datasets--x/file.parquet") == "/tmp/home/datasets--x/file.parquet"
    assert expand_dataset_path("$HF_HOME/datasets--x/file.parquet") == "/tmp/home/datasets--x/file.parquet"


@pytest.mark.runs_on(["cpu", "mps"])
def test_hf_hub_cache_falls_back_to_hf_home(clear_hf_cache_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HOME", "/tmp/home")
    assert expand_dataset_path("${HF_HUB_CACHE}/datasets--x/file.parquet") == "/tmp/home/datasets--x/file.parquet"


@pytest.mark.runs_on(["cpu", "mps"])
def test_hf_home_falls_back_to_hf_hub_cache(clear_hf_cache_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/hub")
    assert expand_dataset_path("${HF_HOME}/datasets--x/file.parquet") == "/tmp/hub/datasets--x/file.parquet"


@pytest.mark.runs_on(["cpu", "mps"])
def test_missing_hf_cache_env_raises(clear_hf_cache_env):
    with pytest.raises(ValueError, match="neither environment variable is set"):
        expand_dataset_path("${HF_HUB_CACHE}/datasets--x/file.parquet")
    with pytest.raises(ValueError, match="Source scripts/utils/env.sh"):
        expand_dataset_path("$HF_HOME/foo")


@pytest.mark.runs_on(["cpu", "mps"])
def test_join_keeps_expanded_absolute_path(clear_hf_cache_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/hub")
    expanded = expand_dataset_path("${HF_HUB_CACHE}/datasets--x/file.parquet")
    assert os.path.join("data", expanded) == "/tmp/hub/datasets--x/file.parquet"


@pytest.mark.runs_on(["cpu", "mps"])
def test_get_dataset_list_expands_file_name(clear_hf_cache_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/hub")
    info = {
        "Scene30k": {
            "file_name": "${HF_HUB_CACHE}/datasets--x/file.parquet",
            "formatting": "alpaca",
        }
    }
    attrs = get_dataset_list(["Scene30k"], info)
    assert attrs[0].dataset_name == "/tmp/hub/datasets--x/file.parquet"
    assert os.path.join("data", attrs[0].dataset_name) == "/tmp/hub/datasets--x/file.parquet"


@pytest.mark.runs_on(["cpu", "mps"])
def test_scene30k_entry_uses_hf_hub_cache(clear_hf_cache_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/fake_hub")
    attrs = get_dataset_list(["Scene30k"], DATA_DIR)
    assert attrs[0].dataset_name.startswith("/tmp/fake_hub/datasets--cvis-tmu--Scene30K/")
    assert attrs[0].dataset_name.endswith("train-00000-of-00001.with_question_id.parquet")


@pytest.mark.runs_on(["cpu", "mps"])
def test_cache_relative_file_name(clear_hf_cache_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/hub")
    under = cache_relative_file_name("/tmp/hub/datasets--x/file.parquet")
    assert under == "${HF_HUB_CACHE}/datasets--x/file.parquet"
    assert cache_relative_file_name("/other/path.json") is None
