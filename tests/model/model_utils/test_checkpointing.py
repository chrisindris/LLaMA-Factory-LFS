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
from functools import partial

import pytest
import torch

from llamafactory.extras.misc import get_current_device
from llamafactory.model.model_utils.checkpointing import get_custom_gradient_checkpointing_func
from llamafactory.train.test_utils import load_train_model


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


class _TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


def _wrap_with_probe():
    called = {"checkpoint": False}

    def fake_checkpoint(func, *args, **kwargs):
        called["checkpoint"] = True
        return func(*args, **kwargs)

    return get_custom_gradient_checkpointing_func(fake_checkpoint), called


def test_custom_gc_checkpoints_frozen_module_when_input_requires_grad():
    block = _TinyBlock()
    for param in block.parameters():
        param.requires_grad_(False)

    wrapped, called = _wrap_with_probe()
    hidden_states = torch.randn(2, 4, requires_grad=True)
    output = wrapped(block.forward, hidden_states)

    assert called["checkpoint"] is True
    assert output.shape == hidden_states.shape


def test_custom_gc_skips_frozen_module_when_input_has_no_grad():
    block = _TinyBlock()
    for param in block.parameters():
        param.requires_grad_(False)

    wrapped, called = _wrap_with_probe()
    hidden_states = torch.randn(2, 4, requires_grad=False)
    output = wrapped(block.forward, hidden_states)

    assert called["checkpoint"] is False
    assert output.shape == hidden_states.shape


def test_custom_gc_checkpoints_trainable_module():
    block = _TinyBlock()
    wrapped, called = _wrap_with_probe()
    hidden_states = torch.randn(2, 4, requires_grad=False)
    output = wrapped(block.forward, hidden_states)

    assert called["checkpoint"] is True
    assert hidden_states.requires_grad is True
    assert output.shape == (2, 4)


def test_custom_gc_checkpoints_frozen_module_via_partial_call():
    block = _TinyBlock()
    for param in block.parameters():
        param.requires_grad_(False)

    wrapped, called = _wrap_with_probe()
    hidden_states = torch.randn(2, 4, requires_grad=True)
    output = wrapped(partial(block.__call__), hidden_states)

    assert called["checkpoint"] is True
    assert output.shape == hidden_states.shape


@pytest.mark.parametrize("disable_gradient_checkpointing", [False, True])
def test_vanilla_checkpointing(disable_gradient_checkpointing: bool):
    model = load_train_model(disable_gradient_checkpointing=disable_gradient_checkpointing, **TRAIN_ARGS)
    for module in filter(lambda m: hasattr(m, "gradient_checkpointing"), model.modules()):
        assert getattr(module, "gradient_checkpointing") != disable_gradient_checkpointing


def test_unsloth_gradient_checkpointing():
    model = load_train_model(use_unsloth_gc=True, **TRAIN_ARGS)
    for module in filter(lambda m: hasattr(m, "gradient_checkpointing"), model.modules()):
        assert module._gradient_checkpointing_func.__self__.__name__ == "UnslothGradientCheckpointing"


def test_upcast_layernorm():
    model = load_train_model(upcast_layernorm=True, **TRAIN_ARGS)
    for name, param in model.named_parameters():
        if param.ndim == 1 and "norm" in name:
            assert param.dtype == torch.float32


def test_upcast_lmhead_output():
    model = load_train_model(upcast_lmhead_output=True, **TRAIN_ARGS)
    inputs = torch.randn((1, 16), dtype=torch.float16, device=get_current_device())
    outputs: torch.Tensor = model.get_output_embeddings()(inputs)
    assert outputs.dtype == torch.float32
