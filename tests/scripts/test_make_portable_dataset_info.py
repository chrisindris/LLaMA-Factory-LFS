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

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import make_portable_dataset_info as mpdi  # noqa: E402


def test_absolute_file_name_becomes_dataset_scoped_relative_path():
    registry = {"Scene30k": {"file_name": "/abs/hub/snap/data/train-00000-of-00001.parquet"}}
    new_registry, links = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert new_registry["Scene30k"]["file_name"] == "Scene30k/train-00000-of-00001.parquet"
    assert links == [("Scene30k/train-00000-of-00001.parquet", "/abs/hub/snap/data/train-00000-of-00001.parquet")]


def test_relative_file_name_is_reanchored_to_new_dir():
    registry = {"3DThinker10k": {"file_name": "3DThinker-10K/out/3dthinker10k_cot.jsonl"}}
    new_registry, links = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert new_registry["3DThinker10k"]["file_name"] == "../3DThinker-10K/out/3dthinker10k_cot.jsonl"
    assert links == []


def test_hub_url_entries_are_left_alone():
    registry = {"alpaca_en_demo": {"hf_hub_url": "llamafactory/alpaca_en"}}
    new_registry, links = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert new_registry == registry
    assert links == []


def test_source_registry_is_not_mutated():
    registry = {"Scene30k": {"file_name": "/abs/x.parquet"}}
    mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert registry["Scene30k"]["file_name"] == "/abs/x.parquet"


def test_other_keys_are_preserved():
    registry = {
        "Scene30k": {
            "file_name": "/abs/x.parquet",
            "formatting": "alpaca",
            "columns": {"prompt": "q", "response": "cot"},
        }
    }
    new_registry, _ = mpdi.rewrite_registry(registry, "/repo/data", "/repo/data/annotations")

    assert new_registry["Scene30k"]["formatting"] == "alpaca"
    assert new_registry["Scene30k"]["columns"] == {"prompt": "q", "response": "cot"}


def test_main_writes_dest_registry(tmp_path):
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": "sub/x.jsonl"}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    rc = mpdi.main(["--source", str(source), "--dest", str(dest), "--no-symlinks"])

    assert rc == 0
    written = json.loads(dest.read_text(encoding="utf-8"))
    assert written["D"]["file_name"] == "../sub/x.jsonl"


def test_main_reports_missing_absolute_target(tmp_path, capsys):
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": "/definitely/missing/x.parquet"}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    rc = mpdi.main(["--source", str(source), "--dest", str(dest), "--no-symlinks"])

    assert rc == 1
    assert "missing" in capsys.readouterr().out.lower()


def test_main_creates_symlink(tmp_path):
    target = tmp_path / "real" / "x.parquet"
    target.parent.mkdir(parents=True)
    target.write_text("data", encoding="utf-8")
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": str(target)}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    rc = mpdi.main(["--source", str(source), "--dest", str(dest)])

    assert rc == 0
    assert (tmp_path / "annotations" / "D" / "x.parquet").resolve() == target.resolve()


def test_main_is_idempotent(tmp_path):
    target = tmp_path / "real" / "x.parquet"
    target.parent.mkdir(parents=True)
    target.write_text("data", encoding="utf-8")
    source = tmp_path / "dataset_info.json"
    source.write_text(json.dumps({"D": {"file_name": str(target)}}), encoding="utf-8")
    dest = tmp_path / "annotations" / "dataset_info.json"

    assert mpdi.main(["--source", str(source), "--dest", str(dest)]) == 0
    assert mpdi.main(["--source", str(source), "--dest", str(dest)]) == 0
