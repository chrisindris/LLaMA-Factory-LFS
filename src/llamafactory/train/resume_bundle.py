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

"""Resume-bundle inventory, classification, and packaging helpers.

A resume bundle is either a full LoRA training checkpoint directory or a
`resume_bundle/` sidecar next to a merged dense model. Classification:

- ``full``: LoRA + trainer_state + scheduler + optimizer state (DeepSpeed or HF)
- ``partial``: LoRA + trainer_state but missing optim and/or scheduler
- ``weights_only``: dense or adapter weights without train continuity
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


RESUME_MANIFEST_NAME = "resume_manifest.json"
RESUME_BUNDLE_SUBDIR = "resume_bundle"
SCHEMA_VERSION = "1.0"

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHT_CANDIDATES = ("adapter_model.safetensors", "adapter_model.bin")
TRAINER_STATE_NAME = "trainer_state.json"
SCHEDULER_NAME = "scheduler.pt"
TRAINING_ARGS_NAME = "training_args.bin"
OPTIMIZER_CANDIDATES = ("optimizer.pt", "optimizer.bin")

_GLOBAL_STEP_DIR_RE = re.compile(r"^global_step(\d+)$")
_RNG_STATE_RE = re.compile(r"^rng_state(?:_(\d+))?\.pth$")
_OPTIM_SHARD_RE = re.compile(r"zero_pp_rank_(\d+).*optim_states")


class ResumeClass(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    WEIGHTS_ONLY = "weights_only"


@dataclass
class ArtifactStatus:
    present: bool
    path: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class ResumeInventory:
    directory: str
    classification: ResumeClass
    artifacts: dict[str, ArtifactStatus] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    world_size_inferred: Optional[int] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None
    max_steps: Optional[int] = None
    num_train_epochs: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    @property
    def resume_capable(self) -> bool:
        return self.classification == ResumeClass.FULL


def _exists_file(path: Path) -> bool:
    return path.is_file()


def _find_first(directory: Path, names: tuple[str, ...]) -> Optional[Path]:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _find_global_step_dir(directory: Path) -> Optional[Path]:
    matches: list[tuple[int, Path]] = []
    if not directory.is_dir():
        return None
    for child in directory.iterdir():
        if not child.is_dir():
            continue
        m = _GLOBAL_STEP_DIR_RE.match(child.name)
        if m:
            matches.append((int(m.group(1)), child))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def _count_optim_shards(global_step_dir: Path) -> tuple[int, list[str]]:
    ranks: set[int] = set()
    paths: list[str] = []
    for child in global_step_dir.iterdir():
        if not child.is_file():
            continue
        m = _OPTIM_SHARD_RE.search(child.name)
        if m:
            ranks.add(int(m.group(1)))
            paths.append(str(child))
    return len(ranks), sorted(paths)


def _count_rng_states(directory: Path) -> tuple[int, list[str]]:
    ranks: set[int] = set()
    paths: list[str] = []
    for child in directory.iterdir():
        if not child.is_file():
            continue
        m = _RNG_STATE_RE.match(child.name)
        if not m:
            continue
        paths.append(str(child))
        if m.group(1) is None:
            ranks.add(0)
        else:
            ranks.add(int(m.group(1)))
    return len(ranks), sorted(paths)


def _load_trainer_state_meta(trainer_state_path: Path) -> dict[str, Any]:
    try:
        with trainer_state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "max_steps": state.get("max_steps"),
        "num_train_epochs": state.get("num_train_epochs"),
    }


def inventory_resume_dir(directory: str | Path, expected_world_size: Optional[int] = None) -> ResumeInventory:
    """Inventory a checkpoint or resume_bundle directory."""
    directory = Path(directory).expanduser().resolve()
    artifacts: dict[str, ArtifactStatus] = {}
    notes: list[str] = []

    if not directory.is_dir():
        return ResumeInventory(
            directory=str(directory),
            classification=ResumeClass.WEIGHTS_ONLY,
            artifacts={},
            missing_required=["directory"],
            notes=[f"Not a directory: {directory}"],
        )

    adapter_config = directory / ADAPTER_CONFIG_NAME
    artifacts["adapter_config"] = ArtifactStatus(adapter_config.is_file(), str(adapter_config) if adapter_config.is_file() else None)

    adapter_weights = _find_first(directory, ADAPTER_WEIGHT_CANDIDATES)
    artifacts["adapter_weights"] = ArtifactStatus(
        adapter_weights is not None,
        str(adapter_weights) if adapter_weights else None,
    )

    trainer_state = directory / TRAINER_STATE_NAME
    artifacts["trainer_state"] = ArtifactStatus(trainer_state.is_file(), str(trainer_state) if trainer_state.is_file() else None)

    scheduler = directory / SCHEDULER_NAME
    artifacts["scheduler"] = ArtifactStatus(scheduler.is_file(), str(scheduler) if scheduler.is_file() else None)

    training_args = directory / TRAINING_ARGS_NAME
    artifacts["training_args"] = ArtifactStatus(training_args.is_file(), str(training_args) if training_args.is_file() else None)

    # Optimizer: HF single-file or DeepSpeed ZeRO shards
    optimizer_file = _find_first(directory, OPTIMIZER_CANDIDATES)
    global_step_dir = _find_global_step_dir(directory)
    optim_shard_count = 0
    optim_paths: list[str] = []
    if global_step_dir is not None:
        optim_shard_count, optim_paths = _count_optim_shards(global_step_dir)
        artifacts["deepspeed_global_step_dir"] = ArtifactStatus(True, str(global_step_dir))
        artifacts["deepspeed_optim_shards"] = ArtifactStatus(
            optim_shard_count > 0,
            global_step_dir.name if optim_shard_count > 0 else None,
            detail=f"{optim_shard_count} rank shard(s)",
        )
    else:
        artifacts["deepspeed_global_step_dir"] = ArtifactStatus(False)
        artifacts["deepspeed_optim_shards"] = ArtifactStatus(False)

    artifacts["optimizer_file"] = ArtifactStatus(
        optimizer_file is not None,
        str(optimizer_file) if optimizer_file else None,
    )
    has_optimizer = (optimizer_file is not None) or (optim_shard_count > 0)

    rng_count, rng_paths = _count_rng_states(directory)
    artifacts["rng_states"] = ArtifactStatus(
        rng_count > 0,
        rng_paths[0] if rng_paths else None,
        detail=f"{rng_count} file(s)",
    )

    world_size_inferred: Optional[int] = None
    if optim_shard_count > 0:
        world_size_inferred = optim_shard_count
    elif rng_count > 0:
        world_size_inferred = rng_count

    if expected_world_size is not None and optim_shard_count > 0 and optim_shard_count != expected_world_size:
        notes.append(
            f"DeepSpeed optim shard count ({optim_shard_count}) != expected world_size ({expected_world_size})."
        )

    meta: dict[str, Any] = {}
    if trainer_state.is_file():
        meta = _load_trainer_state_meta(trainer_state)

    # Dense weights (merged model) — informative only
    has_dense = any(
        (directory / name).exists()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )
    artifacts["dense_model_weights"] = ArtifactStatus(has_dense)

    has_adapter = artifacts["adapter_config"].present and artifacts["adapter_weights"].present
    has_trainer_state = artifacts["trainer_state"].present
    has_scheduler = artifacts["scheduler"].present

    missing_required: list[str] = []
    missing_optional: list[str] = []

    if not has_adapter:
        if not artifacts["adapter_config"].present:
            missing_required.append("adapter_config")
        if not artifacts["adapter_weights"].present:
            missing_required.append("adapter_weights")
    if not has_trainer_state:
        missing_required.append("trainer_state")
    if not has_scheduler:
        missing_required.append("scheduler")
    if not has_optimizer:
        missing_required.append("optimizer")

    if not artifacts["rng_states"].present:
        missing_optional.append("rng_states")
    if not artifacts["training_args"].present:
        missing_optional.append("training_args")

    if has_adapter and has_trainer_state and has_scheduler and has_optimizer:
        classification = ResumeClass.FULL
    elif has_adapter and has_trainer_state:
        classification = ResumeClass.PARTIAL
        notes.append("Partial resume: adapter+trainer_state present but optim and/or scheduler missing.")
    else:
        classification = ResumeClass.WEIGHTS_ONLY
        if has_dense and not has_adapter:
            notes.append("Dense weights only (e.g. merged model) — warm-start only.")
        elif has_adapter and not has_trainer_state:
            notes.append("Adapter weights without trainer_state — warm-start / weight-only continue.")

    return ResumeInventory(
        directory=str(directory),
        classification=classification,
        artifacts=artifacts,
        missing_required=missing_required if classification != ResumeClass.FULL else [],
        missing_optional=missing_optional,
        world_size_inferred=world_size_inferred,
        global_step=meta.get("global_step"),
        epoch=meta.get("epoch"),
        max_steps=meta.get("max_steps"),
        num_train_epochs=meta.get("num_train_epochs"),
        notes=notes,
    )


def build_manifest(
    inventory: ResumeInventory,
    *,
    base_model_name_or_path: Optional[str] = None,
    deepspeed_stage: Optional[int] = None,
    finetuning_type: Optional[str] = None,
    lora_rank: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lr_scheduler_type: Optional[str] = None,
    warmup_ratio: Optional[float] = None,
    merged_weights: bool = False,
    resume_mode_if_incomplete: str = "warm_start",
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    artifacts_out = {
        key: {
            "present": status.present,
            "path": status.path,
            "detail": status.detail,
        }
        for key, status in inventory.artifacts.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "directory": inventory.directory,
        "classification": inventory.classification.value,
        "resume_capable": inventory.resume_capable,
        "global_step": inventory.global_step,
        "epoch": inventory.epoch,
        "max_steps": inventory.max_steps,
        "num_train_epochs": inventory.num_train_epochs,
        "world_size": inventory.world_size_inferred,
        "deepspeed_stage": deepspeed_stage,
        "finetuning_type": finetuning_type,
        "lora_rank": lora_rank,
        "base_model_name_or_path": base_model_name_or_path,
        "base_model_for_lora_resume": base_model_name_or_path,
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_ratio": warmup_ratio,
        "merged_weights": merged_weights,
        "artifacts": artifacts_out,
        "missing_required": inventory.missing_required,
        "missing_optional": inventory.missing_optional,
        "resume_mode_if_incomplete": resume_mode_if_incomplete,
        "notes": list(inventory.notes),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_resume_manifest(
    directory: str | Path,
    *,
    expected_world_size: Optional[int] = None,
    base_model_name_or_path: Optional[str] = None,
    deepspeed_stage: Optional[int] = None,
    finetuning_type: Optional[str] = None,
    lora_rank: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lr_scheduler_type: Optional[str] = None,
    warmup_ratio: Optional[float] = None,
    merged_weights: bool = False,
    resume_mode_if_incomplete: str = "warm_start",
    extra: Optional[dict[str, Any]] = None,
) -> tuple[ResumeInventory, Path]:
    directory = Path(directory).expanduser().resolve()
    inventory = inventory_resume_dir(directory, expected_world_size=expected_world_size)
    manifest = build_manifest(
        inventory,
        base_model_name_or_path=base_model_name_or_path,
        deepspeed_stage=deepspeed_stage,
        finetuning_type=finetuning_type,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        merged_weights=merged_weights,
        resume_mode_if_incomplete=resume_mode_if_incomplete,
        extra=extra,
    )
    manifest_path = directory / RESUME_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return inventory, manifest_path


def classify_resume_source(path: str | Path, expected_world_size: Optional[int] = None) -> ResumeInventory:
    path = Path(path).expanduser().resolve()
    if (path / RESUME_BUNDLE_SUBDIR).is_dir():
        return inventory_resume_dir(path / RESUME_BUNDLE_SUBDIR, expected_world_size=expected_world_size)
    return inventory_resume_dir(path, expected_world_size=expected_world_size)


def find_resume_bundle_dir(
    *,
    resume_from_checkpoint: Optional[str] = None,
    adapter_name_or_path: Optional[list[str]] = None,
    model_name_or_path: Optional[str] = None,
    resume_bundle_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """Search order for a resume source directory (not yet classified)."""
    candidates: list[Path] = []

    if resume_bundle_dir:
        candidates.append(Path(resume_bundle_dir))

    if resume_from_checkpoint:
        candidates.append(Path(resume_from_checkpoint))

    if adapter_name_or_path:
        candidates.append(Path(adapter_name_or_path[-1]))

    if model_name_or_path:
        model_path = Path(model_name_or_path)
        candidates.append(model_path / RESUME_BUNDLE_SUBDIR)
        candidates.append(model_path)

    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved.is_dir():
            return str(resolved)
    return None


# Files/dirs to copy when packaging a resume_bundle sidecar from a LoRA checkpoint
_BUNDLE_COPY_FILES = (
    ADAPTER_CONFIG_NAME,
    *ADAPTER_WEIGHT_CANDIDATES,
    TRAINER_STATE_NAME,
    SCHEDULER_NAME,
    TRAINING_ARGS_NAME,
    "latest",
    "zero_to_fp32.py",
    "README.md",
)
_BUNDLE_COPY_GLOBS = (
    "rng_state*.pth",
    "adapter_model.*",
)


def package_resume_bundle(
    source_checkpoint: str | Path,
    dest_dir: str | Path,
    *,
    base_model_name_or_path: Optional[str] = None,
    deepspeed_stage: Optional[int] = None,
    finetuning_type: Optional[str] = "lora",
    lora_rank: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lr_scheduler_type: Optional[str] = None,
    warmup_ratio: Optional[float] = None,
    merged_weights_parent: bool = True,
    use_symlinks: bool = False,
) -> tuple[ResumeInventory, Path]:
    """Copy resume artifacts from a LoRA checkpoint into dest_dir and write manifest."""
    source = Path(source_checkpoint).expanduser().resolve()
    dest = Path(dest_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source}")

    dest.mkdir(parents=True, exist_ok=True)

    def _copy(src: Path, dst: Path) -> None:
        if use_symlinks:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
        else:
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, symlinks=True)
            else:
                shutil.copy2(src, dst)

    for name in _BUNDLE_COPY_FILES:
        src = source / name
        if src.is_file():
            _copy(src, dest / name)

    for pattern in _BUNDLE_COPY_GLOBS:
        for src in source.glob(pattern):
            if src.is_file():
                _copy(src, dest / src.name)

    gs_dir = _find_global_step_dir(source)
    if gs_dir is not None:
        _copy(gs_dir, dest / gs_dir.name)

    inventory, manifest_path = write_resume_manifest(
        dest,
        base_model_name_or_path=base_model_name_or_path,
        deepspeed_stage=deepspeed_stage,
        finetuning_type=finetuning_type,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        merged_weights=merged_weights_parent,
        extra={
            "source_checkpoint": str(source),
            "packaged_as": RESUME_BUNDLE_SUBDIR,
        },
    )
    return inventory, manifest_path


def inventory_to_public_dict(inventory: ResumeInventory) -> dict[str, Any]:
    return {
        "directory": inventory.directory,
        "classification": inventory.classification.value,
        "resume_capable": inventory.resume_capable,
        "missing_required": inventory.missing_required,
        "missing_optional": inventory.missing_optional,
        "world_size_inferred": inventory.world_size_inferred,
        "global_step": inventory.global_step,
        "epoch": inventory.epoch,
        "max_steps": inventory.max_steps,
        "notes": inventory.notes,
        "artifacts": {k: asdict(v) for k, v in inventory.artifacts.items()},
    }
