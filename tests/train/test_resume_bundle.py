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

from llamafactory.train.resume_bundle import ResumeClass, classify_resume_source, inventory_resume_dir


def _write(path: Path, text: str = "stub") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _adapter_files(directory: Path) -> None:
    _write(directory / "adapter_config.json", "{}")
    _write(directory / "adapter_model.safetensors")


def test_inventory_full_hf_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-100"
    _adapter_files(ckpt)
    _write(
        ckpt / "trainer_state.json",
        '{"global_step": 100, "epoch": 1.0, "max_steps": 500, "num_train_epochs": 5.0}',
    )
    _write(ckpt / "scheduler.pt")
    _write(ckpt / "optimizer.pt")

    inventory = inventory_resume_dir(ckpt)

    assert inventory.classification == ResumeClass.FULL
    assert inventory.resume_capable
    assert inventory.global_step == 100
    assert inventory.epoch == 1.0
    assert inventory.missing_required == []


def test_inventory_partial_missing_optim(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-100"
    _adapter_files(ckpt)
    _write(ckpt / "trainer_state.json", '{"global_step": 100}')
    _write(ckpt / "scheduler.pt")

    inventory = inventory_resume_dir(ckpt)

    assert inventory.classification == ResumeClass.PARTIAL
    assert not inventory.resume_capable
    assert "optimizer" in inventory.missing_required


def test_inventory_weights_only_adapter(tmp_path: Path):
    ckpt = tmp_path / "adapter_only"
    _adapter_files(ckpt)

    inventory = inventory_resume_dir(ckpt)

    assert inventory.classification == ResumeClass.WEIGHTS_ONLY
    assert "trainer_state" in inventory.missing_required


def test_classify_resume_source_prefers_sidecar(tmp_path: Path):
    merged = tmp_path / "merged_model"
    sidecar = merged / "resume_bundle"
    _adapter_files(sidecar)
    _write(sidecar / "trainer_state.json", '{"global_step": 620}')
    _write(sidecar / "scheduler.pt")
    _write(sidecar / "optimizer.pt")
    _write(merged / "model.safetensors")

    inventory = classify_resume_source(merged)

    assert inventory.classification == ResumeClass.FULL
    assert Path(inventory.directory) == sidecar.resolve()
