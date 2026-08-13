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
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

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

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

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

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        debug_samples = inputs.pop("debug_samples", None)
        if os.getenv("CLUSTER") == "KILLARNEY" and os.getenv("RUNNING_MODE") == "VENV":
            # HACK: to avoid "liger_fused_linear_cross_entropy() got an unexpected keyword argument '_indices'"
            # Defense in depth: never forward dataset index bookkeeping into the model
            # (Liger fused CE rejects unexpected kwargs like _indices).
            inputs.pop("_indices", None)
        if model.training and self._should_log_mm_debug():
            self._log_mm_debug(inputs, when="pre_forward", debug_samples=debug_samples)
            self._debug_mm_seen += 1

        try:
            return super().compute_loss(model, inputs, *args, **kwargs)
        except Exception as error:
            if self.finetuning_args.debug_mm_training:
                message = str(error)
                if "CUDA out of memory" in message or "Image features and image tokens do not match" in message:
                    self._log_mm_debug(inputs, when="exception", error=error, debug_samples=debug_samples)
            raise

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
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

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

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
