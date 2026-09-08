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

from pathlib import Path

import pytest

from llamafactory.extras.logging import get_logger
from llamafactory.hparams import ModelArguments


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_TOKENS_YAML = REPO_ROOT / "data" / "control_tokens.yaml"
CONTROL_TOKENS_JSON = REPO_ROOT / "data" / "control_tokens.json"


def test_control_tokens_yaml_loads_as_description_dict():
    args = ModelArguments(
        model_name_or_path="dummy",
        new_special_tokens_config=str(CONTROL_TOKENS_YAML),
        init_special_tokens="desc_init_w_noise",
    )
    assert args.add_special_tokens == ["<think>", "</think>", "<answer>", "</answer>"]
    assert args._special_token_descriptions["<think>"].startswith("Marks the start")
    assert args.init_special_tokens == "desc_init_w_noise"


def test_control_tokens_json_list_raises_value_error_not_logger_crash():
    with pytest.raises(ValueError, match="dictionary mapping tokens to descriptions"):
        ModelArguments(
            model_name_or_path="dummy",
            new_special_tokens_config=str(CONTROL_TOKENS_JSON),
        )


def test_error_rank0_exists_on_library_logger():
    logger = get_logger()
    assert callable(getattr(logger, "error_rank0", None))
