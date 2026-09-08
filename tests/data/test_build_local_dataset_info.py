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

import importlib.util
import json
from pathlib import Path

import pytest


def _load_builder():
    path = Path(__file__).resolve().parents[2] / "scripts" / "utils" / "build_local_dataset_info.py"
    spec = importlib.util.spec_from_file_location("build_local_dataset_info", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_check_mapped_columns_detects_missing_json_field(tmp_path: Path):
    builder = _load_builder()
    ann = tmp_path / "scene.json"
    ann.write_text(json.dumps([{"question_with_image_tags": "q", "cot": "a"}]), encoding="utf-8")
    info = {
        "Scene30k": {
            "file_name": str(ann),
            "columns": {
                "system": "formatting_instruction",
                "prompt": "question_with_image_tags",
                "response": "cot",
            },
        }
    }
    errors = builder.check_mapped_columns(info)
    assert errors
    assert "formatting_instruction" in errors[0]


def test_check_mapped_columns_accepts_complete_json(tmp_path: Path):
    builder = _load_builder()
    ann = tmp_path / "scene.json"
    ann.write_text(
        json.dumps(
            [
                {
                    "formatting_instruction": "use think tags",
                    "question_with_image_tags": "q",
                    "cot": "a",
                }
            ]
        ),
        encoding="utf-8",
    )
    info = {
        "Scene30k": {
            "file_name": str(ann),
            "columns": {
                "system": "formatting_instruction",
                "prompt": "question_with_image_tags",
                "response": "cot",
            },
        }
    }
    assert builder.check_mapped_columns(info) == []


def test_parquet_footer_detects_formatting_instruction_column():
    builder = _load_builder()
    repo = Path(__file__).resolve().parents[2]
    formatted = repo / "data" / "train-00000-of-00001.with_question_id.formatted.parquet"
    if not formatted.is_file():
        pytest.skip("formatted Scene30k parquet is not in the checkout")
    missing = builder.columns_present_in_file(formatted, ["formatting_instruction", "cot", "question_id"])
    assert missing == []
