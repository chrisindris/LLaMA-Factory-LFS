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
from types import SimpleNamespace

import llamafactory.hparams.parser as parser


def _create_adapter_checkpoint(checkpoint_dir: Path, include_trainer_state: bool = True) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "adapter_model.safetensors").write_text("stub", encoding="utf-8")
    if include_trainer_state:
        (checkpoint_dir / "trainer_state.json").write_text("{}", encoding="utf-8")


def _build_args(
    adapter_paths: list[str] | None,
    output_dir: Path,
    resume_from_checkpoint: str | None = None,
    stage: str = "sft",
    finetuning_type: str = "lora",
    do_train: bool = True,
    overwrite_output_dir: bool = False,
    create_new_adapter: bool = False,
):
    model_args = SimpleNamespace(adapter_name_or_path=adapter_paths, model_name_or_path=None)
    training_args = SimpleNamespace(
        resume_from_checkpoint=resume_from_checkpoint,
        do_train=do_train,
        output_dir=str(output_dir),
        overwrite_output_dir=overwrite_output_dir,
    )
    finetuning_args = SimpleNamespace(
        stage=stage,
        finetuning_type=finetuning_type,
        create_new_adapter=create_new_adapter,
        resume_bundle_dir=None,
        allow_warm_start_resume=True,
    )
    return model_args, training_args, finetuning_args


def test_auto_resume_from_adapter_checkpoint(tmp_path, monkeypatch):
    adapter_checkpoint = tmp_path / "checkpoint-200"
    _create_adapter_checkpoint(adapter_checkpoint)
    model_args, training_args, finetuning_args = _build_args(
        adapter_paths=[str(adapter_checkpoint)],
        output_dir=tmp_path / "output",
    )

    monkeypatch.setattr(parser, "is_deepspeed_zero3_enabled", lambda: False)

    parser._set_resume_from_checkpoint(model_args, training_args, finetuning_args, can_resume_from_checkpoint=True)

    assert training_args.resume_from_checkpoint == str(adapter_checkpoint)


def test_explicit_resume_overrides_adapter_auto_resume(tmp_path, monkeypatch):
    adapter_checkpoint = tmp_path / "checkpoint-200"
    _create_adapter_checkpoint(adapter_checkpoint)
    model_args, training_args, finetuning_args = _build_args(
        adapter_paths=[str(adapter_checkpoint)],
        output_dir=tmp_path / "output",
        resume_from_checkpoint="manual-checkpoint",
    )

    monkeypatch.setattr(parser, "is_deepspeed_zero3_enabled", lambda: False)

    parser._set_resume_from_checkpoint(model_args, training_args, finetuning_args, can_resume_from_checkpoint=True)

    assert training_args.resume_from_checkpoint == "manual-checkpoint"


def test_missing_trainer_state_skips_adapter_auto_resume(tmp_path, monkeypatch):
    adapter_checkpoint = tmp_path / "checkpoint-200"
    _create_adapter_checkpoint(adapter_checkpoint, include_trainer_state=False)
    model_args, training_args, finetuning_args = _build_args(
        adapter_paths=[str(adapter_checkpoint)],
        output_dir=tmp_path / "missing_output",
    )

    monkeypatch.setattr(parser, "is_deepspeed_zero3_enabled", lambda: False)

    parser._set_resume_from_checkpoint(model_args, training_args, finetuning_args, can_resume_from_checkpoint=True)

    assert training_args.resume_from_checkpoint is None


def test_zero3_skips_adapter_auto_resume(tmp_path, monkeypatch):
    adapter_checkpoint = tmp_path / "checkpoint-200"
    _create_adapter_checkpoint(adapter_checkpoint)
    model_args, training_args, finetuning_args = _build_args(
        adapter_paths=[str(adapter_checkpoint)],
        output_dir=tmp_path / "output",
    )

    monkeypatch.setattr(parser, "is_deepspeed_zero3_enabled", lambda: True)

    parser._set_resume_from_checkpoint(model_args, training_args, finetuning_args, can_resume_from_checkpoint=True)

    assert training_args.resume_from_checkpoint is None


def test_output_dir_fallback_still_works_after_adapter_skip(tmp_path, monkeypatch):
    adapter_checkpoint = tmp_path / "checkpoint-200"
    _create_adapter_checkpoint(adapter_checkpoint, include_trainer_state=False)

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    fallback_checkpoint = output_dir / "checkpoint-300"
    fallback_checkpoint.mkdir(parents=True, exist_ok=True)

    model_args, training_args, finetuning_args = _build_args(
        adapter_paths=[str(adapter_checkpoint)],
        output_dir=output_dir,
    )

    monkeypatch.setattr(parser, "is_deepspeed_zero3_enabled", lambda: False)
    monkeypatch.setattr(parser, "get_last_checkpoint", lambda _: str(fallback_checkpoint))

    parser._set_resume_from_checkpoint(model_args, training_args, finetuning_args, can_resume_from_checkpoint=True)

    assert training_args.resume_from_checkpoint == str(fallback_checkpoint)
