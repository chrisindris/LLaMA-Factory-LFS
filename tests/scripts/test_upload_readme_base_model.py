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

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from upload_model_checkpoint import hub_id_from_base_model, rewrite_checkpoint_readmes  # noqa: E402


CACHE_PATH = (
    "/project/aip-wangcs/indrisch/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
HUB_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def test_hub_id_from_cache_path():
    assert hub_id_from_base_model(CACHE_PATH) == HUB_ID
    assert hub_id_from_base_model(HUB_ID) == HUB_ID
    assert hub_id_from_base_model("models--roberta-base") == "roberta-base"


def test_rewrites_every_readme_under_checkpoint(tmp_path: Path):
    top = tmp_path / "README.md"
    top.write_text(
        "\n".join(
            [
                "---",
                f"base_model: {CACHE_PATH}",
                "library_name: peft",
                "tags:",
                f"- base_model:adapter:{CACHE_PATH}",
                "---",
                f"Finetuned from {CACHE_PATH}.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested = nested_dir / "README.md"
    nested.write_text(f"base_model: '{CACHE_PATH}'\n", encoding="utf-8")
    (tmp_path / "README.md.bak").write_text(f"base_model: {CACHE_PATH}\n", encoding="utf-8")

    rewrite_checkpoint_readmes(tmp_path, fallback_base_model=CACHE_PATH)

    top_text = top.read_text(encoding="utf-8")
    assert CACHE_PATH not in top_text
    assert top_text.count(HUB_ID) == 3
    assert f"base_model: {HUB_ID}" in top_text
    assert f"- base_model:adapter:{HUB_ID}" in top_text
    assert nested.read_text(encoding="utf-8") == f"base_model: '{HUB_ID}'\n"
    assert CACHE_PATH in (tmp_path / "README.md.bak").read_text(encoding="utf-8")


def test_fills_empty_base_model_from_fallback(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text(
        f"base_model: ''\ntags:\n- base_model:adapter:{CACHE_PATH}\n",
        encoding="utf-8",
    )

    rewrite_checkpoint_readmes(tmp_path, fallback_base_model=CACHE_PATH)

    text = readme.read_text(encoding="utf-8")
    assert text.startswith(f"base_model: {HUB_ID}\n")
    assert f"- base_model:adapter:{HUB_ID}\n" in text
    assert CACHE_PATH not in text


def test_already_hub_id_is_unchanged(tmp_path: Path):
    readme = tmp_path / "README.md"
    original = f"base_model: {HUB_ID}\n- base_model:adapter:{HUB_ID}\n"
    readme.write_text(original, encoding="utf-8")

    rewrite_checkpoint_readmes(tmp_path)

    assert readme.read_text(encoding="utf-8") == original
