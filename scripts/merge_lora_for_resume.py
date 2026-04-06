#!/usr/bin/env python3
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

"""Merge a base model snapshot and a LoRA checkpoint into a full model.

This script is designed for warm-start continuation workflows where you want to
switch from LoRA checkpoint loading (`adapter_name_or_path`) to full-model
loading (`model_name_or_path` and optionally `resume_from_checkpoint`).

Important:
- The merged output preserves model weights but does not preserve optimizer/
  scheduler/momentum continuity from the LoRA checkpoint.
- For the first run after merging, start without `resume_from_checkpoint`.
- `resume_from_checkpoint` should be used only for checkpoints created by that
  new run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)


def load_export_model():
    try:
        from llamafactory.train.tuner import export_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import llamafactory export backend. "
            "Activate the LLaMA-Factory environment and retry. "
            f"Missing module: {exc.name}"
        ) from exc

    return export_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a base model snapshot and one LoRA checkpoint into a full model "
            "using LLaMA-Factory export logic."
        )
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Path to base model snapshot (e.g. Hugging Face snapshot directory).",
    )
    parser.add_argument(
        "--adapter-checkpoint",
        required=True,
        help="Path to LoRA adapter checkpoint directory (contains adapter_config.json).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for merged full model.",
    )
    parser.add_argument(
        "--template",
        default="qwen2_vl",
        help="LLaMA-Factory template name to use during export.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--export-device",
        choices=("cpu", "auto"),
        default="cpu",
        help="Device setting for export_model.",
    )
    parser.add_argument(
        "--export-size",
        type=int,
        default=5,
        help="Shard size in GB for exported model files.",
    )
    parser.add_argument(
        "--export-legacy-format",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export .bin files instead of safetensors.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward trust_remote_code to model loading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory first if it exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate inputs and print planned export args.",
    )
    return parser.parse_args()


def _check_required_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def validate_inputs(base_model: Path, adapter_checkpoint: Path, output_dir: Path, overwrite: bool) -> None:
    if not base_model.exists() or not base_model.is_dir():
        raise ValueError(f"Base model directory does not exist: {base_model}")

    if not adapter_checkpoint.exists() or not adapter_checkpoint.is_dir():
        raise ValueError(f"Adapter checkpoint directory does not exist: {adapter_checkpoint}")

    _check_required_file(adapter_checkpoint / "adapter_config.json", "adapter config")

    if not any((adapter_checkpoint / weight_name).exists() for weight_name in ADAPTER_WEIGHT_FILES):
        raise ValueError(
            "Adapter checkpoint is missing adapter weights. "
            f"Expected one of: {', '.join(ADAPTER_WEIGHT_FILES)}"
        )

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite to replace it."
        )


def build_export_args(args: argparse.Namespace, base_model: Path, adapter_checkpoint: Path, output_dir: Path) -> dict:
    export_args = {
        "model_name_or_path": str(base_model),
        "adapter_name_or_path": str(adapter_checkpoint),
        "template": args.template,
        "trust_remote_code": args.trust_remote_code,
        "export_dir": str(output_dir),
        "export_size": args.export_size,
        "export_device": args.export_device,
        "export_legacy_format": args.export_legacy_format,
    }

    if args.cache_dir:
        export_args["cache_dir"] = args.cache_dir

    return export_args


def verify_output(output_dir: Path) -> list[str]:
    warnings: list[str] = []

    _check_required_file(output_dir / "config.json", "model config")

    if not any((output_dir / file_name).exists() for file_name in MODEL_WEIGHT_FILES):
        raise RuntimeError(
            "Merged output is missing model weights. "
            f"Expected one of: {', '.join(MODEL_WEIGHT_FILES)}"
        )

    if not any((output_dir / file_name).exists() for file_name in TOKENIZER_FILES):
        warnings.append(
            "Tokenizer files look incomplete. "
            f"Expected at least one of: {', '.join(TOKENIZER_FILES)}"
        )

    if not any((output_dir / file_name).exists() for file_name in PROCESSOR_FILES):
        warnings.append(
            "Processor files were not found. This is normal for text-only models, "
            "but may indicate an issue for multimodal models."
        )

    return warnings


def write_manifest(
    output_dir: Path,
    base_model: Path,
    adapter_checkpoint: Path,
    export_args: dict,
) -> Path:
    manifest_path = output_dir / "merge_manifest.json"
    manifest = {
        "script": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": str(base_model),
        "adapter_checkpoint": str(adapter_checkpoint),
        "output_dir": str(output_dir),
        "export_args": export_args,
        "mode": "merged_warm_start",
        "notes": [
            "Model weights are merged and preserved.",
            "Optimizer/scheduler continuity from LoRA checkpoint is not preserved.",
            "Start the first new training run without resume_from_checkpoint.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def print_next_steps(output_dir: Path) -> None:
    print("\nMerged model is ready.")
    print("\nUse this in your next training YAML:")
    print("---")
    print("### model")
    print(f"model_name_or_path: {output_dir}")
    print("# adapter_name_or_path: null")
    print("\n### train")
    print("# First run after merge:")
    print("resume_from_checkpoint: null")
    print("---")
    print("\nAfter this new run creates checkpoints, you can use resume_from_checkpoint")
    print("with those new checkpoints only.")


def main() -> int:
    args = parse_args()
    base_model = Path(args.base_model).expanduser().resolve()
    adapter_checkpoint = Path(args.adapter_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    validate_inputs(base_model, adapter_checkpoint, output_dir, args.overwrite)
    export_args = build_export_args(args, base_model, adapter_checkpoint, output_dir)

    print("Planned export arguments:")
    print(json.dumps(export_args, indent=2))

    if args.dry_run:
        print("\nDry run complete. No files were written.")
        return 0

    export_model = load_export_model()
    export_model(export_args)
    warnings = verify_output(output_dir)
    manifest_path = write_manifest(output_dir, base_model, adapter_checkpoint, export_args)

    print(f"\nWrote merge manifest: {manifest_path}")
    for warning in warnings:
        print(f"WARNING: {warning}")

    print(
        "\nWARNING: This merged output is a warm-start artifact. "
        "It does not preserve exact optimizer/scheduler state continuity from the LoRA checkpoint."
    )
    print_next_steps(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
