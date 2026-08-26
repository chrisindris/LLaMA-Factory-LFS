# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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
import os
import time
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_current_memory
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..prediction_dump import (
    PredictionDumpStore,
    decode_teacher_forced_batch,
    flatten_gathered_pairs,
    normalize_question_ids,
    prompt_lengths_from_labels,
    should_record_train_prediction,
)
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        self._debug_mm_steps = max(int(getattr(finetuning_args, "debug_mm_steps", 0)), 0)
        self._debug_mm_seen = 0
        self._debug_mm_pre_step_seen = 0
        self._debug_mm_started_at = time.time()
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        # Prediction JSON dumps (QUESTION_ID keyed); optional debug feature.
        self._pred_dump_warned_missing_qid = False
        self._eval_pred_buffer: list[tuple[str, str]] = []
        # Synced across ranks after gather; never use rank-0-only store.train_full() to skip.
        self._train_dump_full = False
        self._last_dumped_train_step = -1
        train_path = None
        eval_path = None
        if finetuning_args.save_train_predictions:
            train_path = finetuning_args.train_predictions_file or os.path.join(
                self.args.output_dir, "train_predictions.json"
            )
        if finetuning_args.save_eval_predictions:
            eval_path = finetuning_args.eval_predictions_file or os.path.join(
                self.args.output_dir, "eval_predictions.json"
            )
        self.prediction_dump: Optional[PredictionDumpStore] = None
        if train_path or eval_path:
            self.prediction_dump = PredictionDumpStore(
                train_path=train_path,
                eval_path=eval_path,
                max_train_samples=finetuning_args.train_prediction_max_samples,
            )

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self, *args, **kwargs) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer(*args, **kwargs)

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    def _get_rank(self) -> int:
        return int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))

    def _bytes_to_gb(self, value: int) -> float:
        return round(value / (1024**3), 3)

    def _get_gpu_memory_snapshot(self) -> list[dict[str, float]]:
        if not torch.cuda.is_available():
            return []

        snapshot = []
        for device_id in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
            snapshot.append(
                {
                    "device": int(device_id),
                    "free_gb": self._bytes_to_gb(free_bytes),
                    "total_gb": self._bytes_to_gb(total_bytes),
                }
            )
        return snapshot

    def _get_model_cuda_bytes(self) -> tuple[int, int]:
        param_bytes = 0
        buffer_bytes = 0
        for param in self.model.parameters():
            if param.is_cuda:
                param_bytes += param.numel() * param.element_size()

        for buffer in self.model.buffers():
            if buffer.is_cuda:
                buffer_bytes += buffer.numel() * buffer.element_size()

        return param_bytes, buffer_bytes

    def _get_debug_frame_total(self, debug_samples: Optional[list[dict[str, Any]]]) -> Optional[int]:
        if not debug_samples:
            return None

        total_frames = 0
        found_frames = False
        for sample in debug_samples:
            media = sample.get("media") if isinstance(sample, dict) else None
            if not isinstance(media, dict):
                continue

            images = media.get("images")
            if isinstance(images, dict):
                image_count = images.get("count")
                if isinstance(image_count, (int, float)):
                    total_frames += int(image_count)
                    found_frames = True

            videos = media.get("videos")
            if isinstance(videos, dict):
                frame_total = videos.get("frame_total")
                if isinstance(frame_total, (int, float)):
                    total_frames += int(frame_total)
                    found_frames = True
                    continue

                frame_counts = videos.get("frame_counts")
                if isinstance(frame_counts, list) and frame_counts:
                    total_frames += sum(int(count) for count in frame_counts)
                    found_frames = True

        return total_frames if found_frames else None

    def _summarize_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        tensor_shapes: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        for key, value in inputs.items():
            if torch.is_tensor(value):
                tensor_shapes[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                total_bytes += value.numel() * value.element_size()

        summary: dict[str, Any] = {
            "tensor_shapes": tensor_shapes,
            "batch_bytes": total_bytes,
            "batch_gb": self._bytes_to_gb(total_bytes),
        }

        image_feature_keys = (
            "pixel_values",
            "pixel_values_videos",
            "image_features",
            "image_embeds",
            "vision_x",
        )
        image_features: dict[str, dict[str, Any]] = {}
        for key in image_feature_keys:
            value = inputs.get(key)
            if torch.is_tensor(value):
                feature_bytes = value.numel() * value.element_size()
                image_features[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "bytes": feature_bytes,
                    "gb": self._bytes_to_gb(feature_bytes),
                }

        if image_features:
            summary["image_features"] = image_features

        image_tokens: dict[str, Any] = {}
        image_grid = inputs.get("image_grid_thw")
        if torch.is_tensor(image_grid) and image_grid.numel() > 0:
            grid_cpu = image_grid.detach().to("cpu")
            tokens_per_image = torch.prod(grid_cpu, dim=-1)
            image_tokens.update(
                {
                    "image_grid_thw_shape": list(image_grid.shape),
                    "image_tokens_total": int(tokens_per_image.sum().item()),
                    "image_tokens_min": int(tokens_per_image.min().item()),
                    "image_tokens_max": int(tokens_per_image.max().item()),
                    "image_token_count": int(tokens_per_image.numel()),
                }
            )

        image_num_patches = inputs.get("image_num_patches")
        if image_num_patches is None:
            image_num_patches = inputs.get("num_patches")
        if torch.is_tensor(image_num_patches) and image_num_patches.numel() > 0:
            patches_cpu = image_num_patches.detach().to("cpu")
            image_tokens.update(
                {
                    "image_num_patches_shape": list(image_num_patches.shape),
                    "image_num_patches_total": int(patches_cpu.sum().item()),
                }
            )

        video_grid = inputs.get("video_grid_thw")
        if torch.is_tensor(video_grid) and video_grid.numel() > 0:
            grid_cpu = video_grid.detach().to("cpu")
            tokens_per_video = torch.prod(grid_cpu, dim=-1)
            image_tokens.update(
                {
                    "video_grid_thw_shape": list(video_grid.shape),
                    "video_tokens_total": int(tokens_per_video.sum().item()),
                }
            )

        if image_tokens:
            summary["image_tokens"] = image_tokens

        return summary

    def _build_mm_debug_payload(
        self, inputs: dict[str, Any], debug_samples: Optional[list[dict[str, Any]]] = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rank": self._get_rank(),
            "step": int(getattr(self.state, "global_step", -1)),
            "gpu_memory": self._get_gpu_memory_snapshot(),
        }

        if torch.cuda.is_available():
            payload["gpu_allocated_gb"] = self._bytes_to_gb(torch.cuda.memory_allocated())
            payload["gpu_reserved_gb"] = self._bytes_to_gb(torch.cuda.memory_reserved())
            payload["gpu_device"] = int(torch.cuda.current_device())

        param_bytes, buffer_bytes = self._get_model_cuda_bytes()
        payload["model_cuda_param_gb"] = self._bytes_to_gb(param_bytes)
        payload["model_cuda_buffer_gb"] = self._bytes_to_gb(buffer_bytes)
        payload["model_cuda_total_gb"] = self._bytes_to_gb(param_bytes + buffer_bytes)
        if debug_samples is not None:
            payload["samples"] = debug_samples
            video_frame_totals: list[int] = []
            video_frame_counts: list[list[int]] = []
            for sample in debug_samples:
                if not isinstance(sample, dict):
                    continue

                media = sample.get("media")
                if not isinstance(media, dict):
                    continue

                videos = media.get("videos")
                if not isinstance(videos, dict):
                    continue

                frame_counts = videos.get("frame_counts")
                if isinstance(frame_counts, list) and len(frame_counts) != 0:
                    sample_frame_counts = [int(count) for count in frame_counts]
                    video_frame_counts.append(sample_frame_counts)
                    video_frame_totals.append(sum(sample_frame_counts))
            if video_frame_counts:
                payload["video_frame_counts"] = video_frame_counts
                payload["video_frame_total"] = int(sum(video_frame_totals))
                payload["video_frame_total_min"] = int(min(video_frame_totals))
                payload["video_frame_total_max"] = int(max(video_frame_totals))
        payload.update(self._summarize_inputs(inputs))
        return payload

    def _should_log_mm_debug(self) -> bool:
        return self.finetuning_args.debug_mm_training and self._debug_mm_seen < self._debug_mm_steps

    def _should_log_mm_pre_step(self) -> bool:
        return self.finetuning_args.debug_mm_training

    def _log_mm_pre_step(self, inputs: dict[str, Any]) -> None:
        debug_samples = inputs.get("debug_samples") if hasattr(inputs, "get") else None
        payload = self._build_mm_debug_payload(inputs, debug_samples=debug_samples)
        frame_total = self._get_debug_frame_total(debug_samples if isinstance(debug_samples, list) else None)
        free_bytes, total_bytes = get_current_memory()
        current_steps = int(getattr(self.state, "global_step", 0))
        total_steps = int(getattr(self.state, "max_steps", 0))
        elapsed_seconds = max(time.time() - self._debug_mm_started_at, 0.0)
        seconds_per_it = elapsed_seconds / current_steps if current_steps > 0 else 0.0
        remaining_seconds = max((total_steps - current_steps) * seconds_per_it, 0.0) if total_steps > 0 else 0.0

        latest_log = {}
        for entry in reversed(getattr(self.state, "log_history", [])):
            if isinstance(entry, dict) and any(key in entry for key in ("loss", "grad_norm", "learning_rate", "epoch")):
                latest_log = entry
                break

        message = (
            f"[rank{payload['rank']}] mm_debug pre_step step={self._debug_mm_pre_step_seen} "
            f"optimizer_step={payload['step']} frames={frame_total if frame_total is not None else 'unknown'} "
            f"batch_gb={payload['batch_gb']:.3f} free_gb={free_bytes / (1024**3):.3f} "
            f"total_gb={total_bytes / (1024**3):.3f}"
        )
        if torch.cuda.is_available():
            message += (
                f" gpu_allocated_gb={payload.get('gpu_allocated_gb', 0.0):.3f}"
                f" gpu_reserved_gb={payload.get('gpu_reserved_gb', 0.0):.3f}"
                f" model_cuda_total_gb={payload['model_cuda_total_gb']:.3f}"
            )

        logger.info_rank0(message)

        if latest_log:
            log_snapshot = {k: latest_log.get(k) for k in ("loss", "grad_norm", "learning_rate", "epoch") if latest_log.get(k) is not None}
            if log_snapshot:
                progress_line = (
                    f"[rank{payload['rank']}] mm_debug pre_step progress="
                    f"{current_steps}/{total_steps if total_steps > 0 else '?'} "
                    f"elapsed={elapsed_seconds:.0f}s remaining={remaining_seconds:.0f}s "
                    f"s/it={seconds_per_it:.2f} "
                    + json.dumps(log_snapshot, default=str)
                )
                logger.info_rank0(progress_line)

        self._debug_mm_pre_step_seen += 1

    def _log_mm_debug(
        self,
        inputs: dict[str, Any],
        when: str,
        error: Optional[BaseException] = None,
        debug_samples: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        try:
            payload = self._build_mm_debug_payload(inputs, debug_samples=debug_samples)
            payload["event"] = when
            if error is not None:
                payload["error"] = {"type": type(error).__name__, "message": str(error)}

            message = f"[rank{payload['rank']}] mm_debug {when}: {json.dumps(payload, default=str)}"
            if error is None:
                logger.info(message)
            else:
                logger.error(message)
        except Exception as log_error:  # avoid masking the original error
            logger.warning(f"[rank{self._get_rank()}] mm_debug logging failed: {log_error}")

    @override
    def training_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        num_items_in_batch: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        if self._should_log_mm_pre_step() and hasattr(inputs, "get"):
            try:
                self._log_mm_pre_step(inputs)
            except Exception as log_error:  # avoid masking the original training error
                logger.warning(f"[rank{self._get_rank()}] mm_debug pre_step logging failed: {log_error}")

        return super().training_step(model, inputs, num_items_in_batch)

    def _get_tokenizer(self):
        return getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)

    def _should_record_train_prediction_now(self) -> bool:
        if not self.finetuning_args.save_train_predictions or self.prediction_dump is None:
            return False
        return should_record_train_prediction(
            dump_full=self._train_dump_full,
            global_step=int(getattr(self.state, "global_step", 0)),
            interval=int(self.finetuning_args.train_prediction_interval),
            last_dumped_step=self._last_dumped_train_step,
        )

    def _warn_missing_question_ids_once(self) -> None:
        if not self._pred_dump_warned_missing_qid:
            logger.warning_rank0(
                "save_*_predictions is enabled but batch has no question_ids. "
                "IDs are normally auto-assigned at convert time as {dataset}_{row_index}; "
                "if you still see this, ensure overwrite_cache=true so convert re-runs, "
                "or stamp annotations via scripts/assign_question_ids.py. "
                "Gathering empty pairs so ranks stay aligned."
            )
            self._pred_dump_warned_missing_qid = True

    @staticmethod
    def _flatten_gathered_pairs(gathered: Any) -> list[tuple[str, str]]:
        r"""Normalize gather_object / all_gather_object results to a flat pair list."""
        return flatten_gathered_pairs(gathered)

    def _distributed_world_size(self) -> int:
        world_size = 1
        if hasattr(self, "accelerator") and self.accelerator is not None:
            world_size = int(getattr(self.accelerator, "num_processes", 1) or 1)
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                world_size = max(world_size, int(dist.get_world_size()))
        except Exception:
            pass
        return world_size

    def _gather_prediction_pairs(self, local_pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        r"""Gather (question_id, text) pairs from all ranks.

        Note: ``Accelerator.gather_object`` is **not** available on accelerate 1.x
        (e.g. 1.11.0). Use ``accelerate.utils.gather_object`` or
        ``torch.distributed.all_gather_object`` instead. No package upgrade required.
        """
        # Single-process fast path (also covers smoke 1-GPU torchrun).
        world_size = self._distributed_world_size()
        if world_size <= 1:
            return list(local_pairs)

        # 1) accelerate.utils.gather_object (correct API for accelerate>=0.20)
        try:
            from accelerate.utils import gather_object as accel_gather_object

            gathered = accel_gather_object(local_pairs)
            merged = self._flatten_gathered_pairs(gathered)
            if merged or not local_pairs:
                return merged
        except Exception as err_accel:
            logger.warning_rank0(f"accelerate.utils.gather_object failed ({err_accel}); trying torch.distributed")

        # 2) torch.distributed.all_gather_object
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                obj_list: list[Any] = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(obj_list, local_pairs)
                return self._flatten_gathered_pairs(obj_list)
        except Exception as err_dist:
            logger.warning_rank0(
                f"torch.distributed.all_gather_object failed ({err_dist}); "
                "using local-rank pairs only (incomplete under multi-GPU)"
            )

        # 3) Last resort: local only (incomplete multi-GPU dump)
        return list(local_pairs)

    def _sync_train_dump_full(self) -> None:
        r"""Broadcast rank-0 store.train_full() so every rank skips (or dumps) together.

        If the collective fails, leave ``_train_dump_full`` unchanged (False) so all
        ranks keep dumping and gathering — extra work, but not a deadlock.
        """
        full_int = 0
        if self.prediction_dump is not None and self.is_world_process_zero():
            full_int = int(self.prediction_dump.train_full())

        world_size = self._distributed_world_size()
        if world_size <= 1:
            self._train_dump_full = bool(full_int)
            return

        try:
            import torch.distributed as dist

            if not (dist.is_available() and dist.is_initialized()):
                self._train_dump_full = bool(full_int)
                return

            device = torch.device("cpu")
            if dist.get_backend() == "nccl" and torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            flag = torch.tensor([full_int], device=device, dtype=torch.int32)
            dist.broadcast(flag, src=0)
            self._train_dump_full = bool(int(flag.item()))
        except Exception as err:
            logger.warning_rank0(f"train dump full-flag broadcast failed ({err}); keeping dump enabled")

    def _record_train_pairs(self, pairs: list[tuple[str, str]]) -> None:
        r"""Gather local pairs from every rank, then rank-0 writes.

        Always gather, including when ``pairs`` is empty. Skipping the collective
        on some ranks (empty texts or a local train_full() cap) deadlocks NCCL.
        """
        if self.prediction_dump is None:
            return
        step = int(getattr(self.state, "global_step", 0))
        all_pairs = self._gather_prediction_pairs(list(pairs or []))
        if self.is_world_process_zero():
            added = self.prediction_dump.add_train_records(step, all_pairs)
            if added or all_pairs:
                self.prediction_dump.flush_train()
            logger.info_rank0(
                f"train prediction dump: step={step} local={len(pairs)} "
                f"gathered={len(all_pairs)} added={added} total={self.prediction_dump.train_record_count}"
            )
        self._last_dumped_train_step = step
        self._sync_train_dump_full()

    def _record_eval_pairs(self, pairs: list[tuple[str, str]]) -> None:
        if not pairs:
            return
        self._eval_pred_buffer.extend(pairs)

    def _flush_eval_predictions(self) -> None:
        if self.prediction_dump is None or not self.finetuning_args.save_eval_predictions:
            self._eval_pred_buffer = []
            return
        local_n = len(self._eval_pred_buffer)
        all_pairs = self._gather_prediction_pairs(self._eval_pred_buffer)
        self._eval_pred_buffer = []
        if self.is_world_process_zero():
            added = self.prediction_dump.add_eval_records(all_pairs)
            self.prediction_dump.flush_eval()
            logger.info_rank0(
                f"eval prediction dump: local_buffer={local_n} gathered={len(all_pairs)} "
                f"added={added} total={len(self.prediction_dump.eval_data)}"
            )

    def _forward_logits_for_dump(self, model: "torch.nn.Module", inputs: dict[str, Any]) -> Optional["torch.Tensor"]:
        r"""Run a no-grad forward that materializes logits for teacher-forced dumps.

        Liger fused CE sets ``skip_logits=True`` whenever ``model.training and labels
        is not None``, so the training forward often returns ``logits=None``. Dropping
        labels for this diagnostic pass forces logits to be returned without affecting
        the training loss path.
        """
        model_inputs = {
            k: v for k, v in inputs.items() if k not in ("labels", "question_ids", "debug_samples", "_indices")
        }
        was_training = model.training
        try:
            # Keep train/eval mode as-is for correct dropout/BN, but no_grad for dump.
            with torch.no_grad():
                outputs = model(**model_inputs)
            logits = getattr(outputs, "logits", None)
            if logits is None and isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                logits = outputs[0] if torch.is_tensor(outputs[0]) else None
            del outputs
            return logits
        except Exception as err:
            logger.warning_rank0(f"prediction dump logits forward failed: {err}")
            return None
        finally:
            if was_training and not model.training:
                model.train()

    def _release_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _texts_from_teacher_forced(
        self,
        logits: Optional["torch.Tensor"],
        labels: "torch.Tensor",
        model: Optional["torch.nn.Module"] = None,
        inputs: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return []
        owned_logits = False
        try:
            if logits is None and model is not None and inputs is not None:
                logits = self._forward_logits_for_dump(model, inputs)
                owned_logits = logits is not None
            if logits is None:
                logger.warning_rank0(
                    "teacher_forced prediction dump got no logits "
                    "(Liger may skip them when labels are present; dump forward also failed)."
                )
                return []
            return decode_teacher_forced_batch(logits, labels, tokenizer, skip_special_tokens=True)
        finally:
            if owned_logits:
                del logits
                self._release_cuda_cache()

    def _texts_from_generate(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        labels: Optional["torch.Tensor"],
    ) -> list[str]:
        r"""Free-form generation on prompt-only slices. Expensive; for debug dumps only."""
        tokenizer = self._get_tokenizer()
        if tokenizer is None or labels is None or "input_ids" not in inputs:
            return []

        gen_kwargs = dict(getattr(self, "_gen_kwargs", {}) or {})
        # Ensure generation does not require labels
        prompt_lens = prompt_lengths_from_labels(labels)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        batch_size = input_ids.size(0)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # Build left-aligned prompt batch (pad to max prompt length in batch)
        max_prompt = max(prompt_lens) if prompt_lens else 0
        if max_prompt <= 0:
            return [""] * batch_size

        prompt_ids = input_ids.new_full((batch_size, max_prompt), pad_id)
        prompt_mask = input_ids.new_zeros((batch_size, max_prompt))
        for i, plen in enumerate(prompt_lens):
            plen = min(int(plen), int(input_ids.size(1)))
            if plen <= 0:
                continue
            prompt_ids[i, :plen] = input_ids[i, :plen]
            if attention_mask is not None:
                prompt_mask[i, :plen] = attention_mask[i, :plen]
            else:
                prompt_mask[i, :plen] = 1

        gen_inputs: dict[str, Any] = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
        }
        # Pass through multimodal tensors when present (full batch; models usually index by token layout)
        for key, value in inputs.items():
            if key in ("input_ids", "attention_mask", "labels", "question_ids", "debug_samples"):
                continue
            if torch.is_tensor(value) or value is not None:
                gen_inputs[key] = value

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                generated = model.generate(**gen_inputs, **gen_kwargs)
        finally:
            if was_training:
                model.train()

        texts: list[str] = []
        for i in range(batch_size):
            plen = min(int(prompt_lens[i]), int(generated.size(1)))
            new_tokens = generated[i, plen:]
            texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return texts

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        debug_samples = inputs.pop("debug_samples", None)
        question_ids_raw = inputs.pop("question_ids", None)

        if (os.getenv("CLUSTER") == "KILLARNEY" and os.getenv("RUNNING_MODE") == "VENV") or os.getenv(
            "RUNNING_MODE"
        ) == "SMOKE":
            # HACK: to avoid "liger_fused_linear_cross_entropy() got an unexpected keyword argument '_indices'"
            # Defense in depth: never forward dataset index bookkeeping into the model
            # (Liger fused CE rejects unexpected kwargs like _indices).
            inputs.pop("_indices", None)
        if model.training and self._should_log_mm_debug():
            self._log_mm_debug(inputs, when="pre_forward", debug_samples=debug_samples)
            self._debug_mm_seen += 1

        need_train_dump = model.training and self._should_record_train_prediction_now()
        train_mode = self.finetuning_args.train_prediction_mode
        labels_for_dump = inputs.get("labels")

        try:
            result = super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        except Exception as error:
            if self.finetuning_args.debug_mm_training:
                message = str(error)
                if "CUDA out of memory" in message or "Image features and image tokens do not match" in message:
                    self._log_mm_debug(inputs, when="exception", error=error, debug_samples=debug_samples)
            raise

        if need_train_dump:
            batch_size = int(inputs["input_ids"].size(0)) if "input_ids" in inputs else 0
            qids = normalize_question_ids(question_ids_raw, batch_size)
            pairs: list[tuple[str, str]] = []
            if not any(qids):
                self._warn_missing_question_ids_once()
            else:
                texts: list[str] = []
                if train_mode == "teacher_forced":
                    # Do NOT rely on the loss forward: Liger fused CE sets skip_logits=True
                    # whenever training and labels are present, so outputs.logits is None.
                    if labels_for_dump is not None:
                        texts = self._texts_from_teacher_forced(None, labels_for_dump, model=model, inputs=inputs)
                elif train_mode == "generate":
                    try:
                        texts = self._texts_from_generate(model, inputs, labels_for_dump)
                    except Exception as gen_err:
                        logger.warning_rank0(f"train generate prediction dump failed: {gen_err}")
                        texts = []
                if texts:
                    pairs = [(qid, text) for qid, text in zip(qids, texts) if qid]
                else:
                    logger.warning_rank0(
                        f"train prediction dump produced no texts "
                        f"(mode={train_mode}, qids={len([q for q in qids if q])}, batch={batch_size})"
                    )
            # All ranks must gather, even with empty local pairs.
            self._record_train_pairs(pairs)

        return result

    # def compute_loss(self, model, inputs, *args, **kwargs):
    #     if self.finetuning_args.use_asft_loss:
    #         with torch.no_grad():
    #             ref_outputs = self.ref_model(
    #                 input_ids=inputs["input_ids"],
    #                 attention_mask=inputs.get("attention_mask", None),
    #             )
    #             ref_logits = ref_outputs.logits
    #         outputs = model(**inputs)
    #         return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
    #     else:
    #         return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        question_ids_raw = inputs.pop("question_ids", None)
        inputs.pop("debug_samples", None)

        labels_for_dump = inputs.get("labels")
        dump_eval = self.finetuning_args.save_eval_predictions and self.prediction_dump is not None

        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        # When dumping eval predictions without stock predict_with_generate, still compute loss.
        # Stay loss-only for HuggingFace: returning [B, S, vocab] logits makes evaluation_loop
        # concat them on GPU (Qwen2.5-VL vocab is 152064 → tens of GiB per long CoT row).
        if dump_eval and not self.args.predict_with_generate:
            loss, _, label_ids = super().prediction_step(
                model,
                inputs,
                prediction_loss_only=True,
                ignore_keys=ignore_keys,
                **gen_kwargs,
            )
            batch_size = int(inputs["input_ids"].size(0)) if "input_ids" in inputs else 0
            qids = normalize_question_ids(question_ids_raw, batch_size)
            if not any(qids):
                self._warn_missing_question_ids_once()
            else:
                texts: list[str] = []
                if self.finetuning_args.eval_prediction_mode == "teacher_forced":
                    try:
                        if labels_for_dump is not None:
                            texts = self._texts_from_teacher_forced(None, labels_for_dump, model=model, inputs=inputs)
                    except Exception as err:
                        logger.warning_rank0(f"eval teacher_forced dump failed: {err}")
                else:  # generate
                    try:
                        # restore labels for prompt length if popped
                        if labels_for_dump is None:
                            labels_for_dump = labels
                        texts = self._texts_from_generate(model, inputs, labels_for_dump)
                    except Exception as err:
                        logger.warning_rank0(f"eval generate dump failed: {err}")
                if texts:
                    self._record_eval_pairs([(qid, text) for qid, text in zip(qids, texts) if qid])
                else:
                    logger.warning_rank0(
                        f"eval prediction dump produced no texts "
                        f"(mode={self.finetuning_args.eval_prediction_mode}, "
                        f"qids={len([q for q in qids if q])}, batch={batch_size})"
                    )
            return loss, None, label_ids if label_ids is not None else labels

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()
            if dump_eval:
                tokenizer = self._get_tokenizer()
                batch_size = int(generated_tokens.size(0))
                qids = normalize_question_ids(question_ids_raw, batch_size)
                if tokenizer is not None and any(qids):
                    texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    self._record_eval_pairs([(qid, text) for qid, text in zip(qids, texts) if qid])

        return loss, generated_tokens, labels

    @override
    def evaluate(self, *args, **kwargs):
        self._eval_pred_buffer = []
        metrics = super().evaluate(*args, **kwargs)
        if self.finetuning_args.save_eval_predictions:
            self._flush_eval_predictions()
        return metrics

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
