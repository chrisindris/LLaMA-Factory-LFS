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

"""Helpers for dumping model text predictions keyed by QUESTION_ID."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from typing import TYPE_CHECKING, Any, Optional, Union


if TYPE_CHECKING:
    import torch

try:
    from ..extras.constants import IGNORE_INDEX
except Exception:  # pragma: no cover - allow lightweight unit import
    IGNORE_INDEX = -100


logger = logging.getLogger(__name__)


def format_epoch_name(epoch: Any, is_training: bool = False) -> str:
    r"""Format epoch as integer string, dropping trailing decimals (e.g. 1.0 -> '1')."""
    if epoch is None:
        return "1" if is_training else "0"
    try:
        val = float(epoch)
        if is_training:
            return str(max(1, int(math.ceil(val)))) if val > 0 else "1"
        if val.is_integer():
            return str(int(val))
        return str(int(round(val)))
    except (ValueError, TypeError):
        return str(epoch)


def resolve_prediction_dump_path(
    template_or_path: Optional[str],
    epoch: str,
    default_prefix: str,
    output_dir: str = ".",
) -> Optional[str]:
    r"""Resolve path for prediction dumps for a specific epoch."""
    if not template_or_path:
        return os.path.join(output_dir, f"{default_prefix}_ep{epoch}.json")
    if "{epoch}" in template_or_path:
        return template_or_path.format(epoch=epoch)
    if template_or_path.endswith(".json"):
        if f"_ep{epoch}.json" in template_or_path:
            return template_or_path
        return template_or_path[:-5] + f"_ep{epoch}.json"
    return f"{template_or_path}_ep{epoch}.json"


def should_record_train_prediction(
    *,
    dump_full: bool,
    global_step: int,
    interval: int,
    last_dumped_step: int,
) -> bool:
    r"""Whether this microbatch should run a train prediction dump.

    Cap must be a **synced** flag (broadcast after gather), not a rank-0-only
    ``store.train_full()``. Using a local cap to skip gather deadlocks NCCL.
    ``last_dumped_step`` limits dumps to once per optimizer step.
    """
    if dump_full:
        return False
    interval = max(int(interval), 1)
    step = int(global_step)
    if step <= 0 or step % interval != 0:
        return False
    if step == int(last_dumped_step):
        return False
    return True


def flatten_gathered_pairs(gathered: Any) -> list[tuple[str, str]]:
    r"""Normalize gather_object / all_gather_object results to a flat pair list."""
    if gathered is None:
        return []
    merged: list[tuple[str, str]] = []
    if isinstance(gathered, list) and gathered and isinstance(gathered[0], list):
        for chunk in gathered:
            if isinstance(chunk, list):
                merged.extend(chunk)
    elif isinstance(gathered, list) and gathered and isinstance(gathered[0], tuple):
        merged = list(gathered)
    else:
        for chunk in gathered or []:
            if isinstance(chunk, list):
                merged.extend(chunk)
            elif isinstance(chunk, tuple) and len(chunk) == 2:
                merged.append(chunk)
    return merged


def atomic_write_json(path: str, data: dict[str, Any]) -> None:
    r"""Write JSON atomically (temp file + replace)."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="pred_dump_", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class PredictionDumpStore:
    r"""In-memory store for train/eval prediction JSON dumps partitioned by epoch."""

    def __init__(
        self,
        train_path: Optional[str] = None,
        eval_path: Optional[str] = None,
        train_path_template: Optional[str] = None,
        eval_path_template: Optional[str] = None,
        max_train_samples: int = 0,
        output_dir: str = ".",
    ) -> None:
        self.train_path_template = train_path_template or train_path
        self.eval_path_template = eval_path_template or eval_path
        self.max_train_samples = max(int(max_train_samples), 0)
        self.output_dir = output_dir
        self.train_data_by_epoch: dict[str, dict[str, dict[str, str]]] = {}
        self.train_record_counts: dict[str, int] = {}
        self.eval_data_by_epoch: dict[str, dict[str, str]] = {}
        self.last_train_epoch: str = "1"
        self.last_eval_epoch: str = "0"

    def get_train_record_count(self, epoch: Optional[Union[str, int, float]] = None) -> int:
        epoch_key = format_epoch_name(epoch, is_training=True) if epoch is not None else self.last_train_epoch
        return self.train_record_counts.get(epoch_key, 0)

    @property
    def train_record_count(self) -> int:
        return self.get_train_record_count(self.last_train_epoch)

    def train_capacity_remaining(self, epoch: Optional[Union[str, int, float]] = None) -> Optional[int]:
        if self.max_train_samples <= 0:
            return None
        count = self.get_train_record_count(epoch)
        return max(self.max_train_samples - count, 0)

    def train_full(self, epoch: Optional[Union[str, int, float]] = None) -> bool:
        if self.max_train_samples <= 0:
            return False
        return self.get_train_record_count(epoch) >= self.max_train_samples

    def add_train_records(
        self,
        step: int,
        pairs: list[tuple[str, str]],
        epoch: Optional[Union[str, int, float]] = None,
    ) -> int:
        r"""Add train records as D[QUESTION_ID][STEP] = text for given epoch. Returns number added."""
        if not pairs:
            return 0

        epoch_key = format_epoch_name(epoch, is_training=True) if epoch is not None else self.last_train_epoch
        self.last_train_epoch = epoch_key

        step_key = str(int(step))
        added = 0
        epoch_data = self.train_data_by_epoch.setdefault(epoch_key, {})
        for question_id, text in pairs:
            if not question_id:
                continue
            if self.train_full(epoch_key):
                break
            bucket = epoch_data.setdefault(question_id, {})
            # count only first write for this (qid, step); overwrites do not inflate the cap
            is_new = step_key not in bucket
            bucket[step_key] = text
            if is_new:
                self.train_record_counts[epoch_key] = self.train_record_counts.get(epoch_key, 0) + 1
                added += 1
        return added

    def add_eval_records(
        self,
        pairs: list[tuple[str, str]],
        epoch: Optional[Union[str, int, float]] = None,
    ) -> int:
        r"""Add eval records as D[QUESTION_ID] = text (overwrite on repeat) for given epoch."""
        epoch_key = format_epoch_name(epoch, is_training=False) if epoch is not None else self.last_eval_epoch
        self.last_eval_epoch = epoch_key
        epoch_data = self.eval_data_by_epoch.setdefault(epoch_key, {})
        added = 0
        for question_id, text in pairs:
            if not question_id:
                continue
            epoch_data[question_id] = text
            added += 1
        return added

    def get_train_path(self, epoch: Optional[Union[str, int, float]] = None) -> Optional[str]:
        if not self.train_path_template:
            return None
        epoch_key = format_epoch_name(epoch, is_training=True) if epoch is not None else self.last_train_epoch
        return resolve_prediction_dump_path(self.train_path_template, epoch_key, "train_predictions", self.output_dir)

    def get_eval_path(self, epoch: Optional[Union[str, int, float]] = None) -> Optional[str]:
        if not self.eval_path_template:
            return None
        epoch_key = format_epoch_name(epoch, is_training=False) if epoch is not None else self.last_eval_epoch
        return resolve_prediction_dump_path(self.eval_path_template, epoch_key, "eval_predictions", self.output_dir)

    @property
    def train_path(self) -> Optional[str]:
        return self.get_train_path(self.last_train_epoch)

    @property
    def eval_path(self) -> Optional[str]:
        return self.get_eval_path(self.last_eval_epoch)

    @property
    def train_data(self) -> dict[str, dict[str, str]]:
        return self.train_data_by_epoch.get(self.last_train_epoch, {})

    @property
    def eval_data(self) -> dict[str, str]:
        return self.eval_data_by_epoch.get(self.last_eval_epoch, {})

    def flush_train(self, epoch: Optional[Union[str, int, float]] = None) -> Optional[str]:
        path = self.get_train_path(epoch)
        if not path:
            return None
        epoch_key = format_epoch_name(epoch, is_training=True) if epoch is not None else self.last_train_epoch
        data = self.train_data_by_epoch.get(epoch_key, {})
        atomic_write_json(path, data)
        count = self.get_train_record_count(epoch_key)
        logger.info("Wrote train predictions (%s records) to %s", count, path)
        return path

    def flush_eval(self, epoch: Optional[Union[str, int, float]] = None) -> Optional[str]:
        path = self.get_eval_path(epoch)
        if not path:
            return None
        epoch_key = format_epoch_name(epoch, is_training=False) if epoch is not None else self.last_eval_epoch
        data = self.eval_data_by_epoch.get(epoch_key, {})
        atomic_write_json(path, data)
        logger.info("Wrote eval predictions (%s records) to %s", len(data), path)
        return path


def decode_teacher_forced_batch(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    tokenizer,
    skip_special_tokens: bool = True,
) -> list[str]:
    r"""Greedy decode response tokens from teacher-forced logits.

    For each batch row, takes argmax logits at positions where labels != IGNORE_INDEX
    (shifted by one for causal LM: logits[:, t] predicts labels[:, t+1]... standard
    HF causal models use logits[..., :-1] vs labels[..., 1:]).
    """
    if logits is None or labels is None:
        return []

    # Align next-token prediction: pred at position t predicts token t+1.
    # Argmax in-place on a view; do not keep a shifted logits copy (VL vocab is huge).
    pred_ids = logits[:, :-1, :].argmax(dim=-1)
    labels_cpu = labels[:, 1:].detach().cpu()
    pred_ids_cpu = pred_ids.cpu()
    del pred_ids

    batch_texts: list[str] = []
    for i in range(pred_ids_cpu.size(0)):
        mask = labels_cpu[i] != IGNORE_INDEX
        token_ids = pred_ids_cpu[i][mask].tolist()
        if not token_ids:
            batch_texts.append("")
            continue
        text = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        batch_texts.append(text)
    return batch_texts


def prompt_lengths_from_labels(labels: "torch.Tensor") -> list[int]:
    r"""Return prompt length per row: index of first non-IGNORE label, or full length."""
    lengths: list[int] = []
    labels_cpu = labels.detach().cpu()
    for i in range(labels_cpu.size(0)):
        row = labels_cpu[i]
        non_ignore = (row != IGNORE_INDEX).nonzero(as_tuple=False)
        if non_ignore.numel() == 0:
            lengths.append(int(row.numel()))
        else:
            lengths.append(int(non_ignore[0].item()))
    return lengths


def normalize_question_ids(question_ids: Any, batch_size: int) -> list[str]:
    r"""Normalize collator side-channel to a flat list of string IDs (length batch_size)."""
    if question_ids is None:
        return [""] * batch_size

    if isinstance(question_ids, (list, tuple)):
        result: list[str] = []
        for item in question_ids:
            if isinstance(item, (list, tuple)):
                # packed sample: keep first id only for dump simplicity
                result.append(str(item[0]) if item else "")
            else:
                result.append("" if item is None else str(item))
        if len(result) < batch_size:
            result.extend([""] * (batch_size - len(result)))
        return result[:batch_size]

    return [str(question_ids)] * batch_size
