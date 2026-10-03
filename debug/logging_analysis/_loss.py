"""Optional checkpoint probe loss: reference-target teacher-forced CE.

This is **not** the trainer's original batch loss and cannot be recovered from
prediction-dump strings. It evaluates recovered annotation targets at a
checkpoint using this tree's SFT preprocessing and label masking.

Call the metric ``checkpoint probe loss`` / ``reference-target loss``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from _logparse import WarningRecord


logger = logging.getLogger(__name__)

SKIP_REASONS = (
    "missing_image",
    "missing_video",
    "unresolvable_path",
    "processor_error",
    "annotation_missing",
    "checkpoint_missing",
    "OOM",
    "other",
)

EXPECTED_API = (
    "llamafactory.hparams.get_infer_args",
    "llamafactory.model.load_model / load_tokenizer",
    "llamafactory.data.get_template_and_fix_tokenizer",
    "llamafactory.data.converter.get_dataset_converter",
    "llamafactory.data.processor.supervised.SupervisedDatasetProcessor",
    "llamafactory.data.collator.SFTDataCollatorWith4DAttentionMask",
    "llamafactory.extras.constants.IGNORE_INDEX",
)


def discover_checkpoints(root: str | Path) -> dict[int, Path]:
    root_path = Path(root)
    found: dict[int, Path] = {}
    if not root_path.is_dir():
        return found
    for child in root_path.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        suffix = child.name.split("checkpoint-", 1)[-1]
        try:
            step = int(suffix)
        except ValueError:
            continue
        found[step] = child
    return found


def map_steps_to_checkpoints(
    steps: list[int],
    checkpoints: dict[int, Path],
    *,
    nearest: bool,
    warnings: list[WarningRecord],
) -> dict[int, tuple[Path | None, str]]:
    """Return step -> (path or None, policy label)."""
    mapped: dict[int, tuple[Path | None, str]] = {}
    available = sorted(checkpoints)
    for step in steps:
        if step in checkpoints:
            mapped[step] = (checkpoints[step], "exact")
            continue
        if not nearest or not available:
            mapped[step] = (None, "missing")
            warnings.append(
                WarningRecord(
                    code="checkpoint_missing",
                    message=f"No checkpoint-{step} under the checkpoint root.",
                    step=step,
                )
            )
            continue
        nearest_step = min(available, key=lambda value: (abs(value - step), value))
        mapped[step] = (checkpoints[nearest_step], f"nearest:{nearest_step}")
        warnings.append(
            WarningRecord(
                code="checkpoint_nearest",
                message=f"Using checkpoint-{nearest_step} for logged step {step} (--nearest-checkpoint).",
                step=step,
            )
        )
    return mapped


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a mapping")
    return data


def _infer_kwargs_from_train_yaml(train_cfg: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    keys = [
        "model_name_or_path",
        "template",
        "finetuning_type",
        "trust_remote_code",
        "image_max_pixels",
        "video_max_pixels",
        "cache_dir",
        "cutoff_len",
        "media_dir",
        "image_sample_count",
        "flash_attn",
    ]
    kwargs: dict[str, Any] = {key: train_cfg[key] for key in keys if key in train_cfg}
    kwargs.setdefault("template", "qwen2_vl")
    kwargs.setdefault("finetuning_type", "lora")
    kwargs.setdefault("trust_remote_code", True)
    kwargs["adapter_name_or_path"] = str(checkpoint)
    return kwargs


def select_probe_ids(
    frame: pd.DataFrame,
    *,
    max_per_dataset: int,
    seed: int,
    persist_path: Path | None,
) -> dict[str, list[str]]:
    if persist_path and persist_path.exists():
        saved = json.loads(persist_path.read_text(encoding="utf-8"))
        logger.info("Reusing probe question IDs from %s", persist_path)
        return {str(k): list(v) for k, v in saved.items()}

    selected: dict[str, list[str]] = {}
    for dataset, sub in frame.groupby("dataset"):
        ids = sorted(set(sub["question_id"].astype(str)))
        if max_per_dataset > 0 and len(ids) > max_per_dataset:
            ser = pd.Series(ids)
            ids = sorted(ser.sample(n=max_per_dataset, random_state=seed).tolist())
        selected[str(dataset)] = ids
    if persist_path:
        persist_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    return selected


def _skip_row(question_id: str, dataset: str, step: int, reason: str, detail: str = "") -> dict[str, Any]:
    return {
        "question_id": question_id,
        "dataset": dataset,
        "step": step,
        "probe_loss": pd.NA,
        "probe_token_nll": pd.NA,
        "target_token_count": pd.NA,
        "skip_reason": reason,
        "skip_detail": detail,
        "checkpoint_policy": "",
    }


def compute_probe_losses(
    *,
    frame: pd.DataFrame,
    annotation_indices: dict[str, Any],
    dataset_info: dict[str, Any],
    checkpoint_root: Path,
    training_config: Path | None,
    model_name_or_path: str | None,
    llamafactory_root: Path | None,
    loss_batch_size: int,
    max_samples_per_dataset: int,
    seed: int,
    nearest_checkpoint: bool,
    output_dir: Path,
    warnings: list[WarningRecord],
) -> pd.DataFrame:
    """Evaluate reference-target CE at checkpoints. Isolated from log-only analysis."""
    if llamafactory_root is not None:
        import sys

        src = str(Path(llamafactory_root) / "src")
        if src not in sys.path:
            sys.path.insert(0, src)

    try:
        import torch

        from llamafactory.data import get_template_and_fix_tokenizer
        from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
        from llamafactory.data.converter import get_dataset_converter
        from llamafactory.data.parser import DatasetAttr
        from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
        from llamafactory.extras.constants import IGNORE_INDEX
        from llamafactory.hparams import get_infer_args
        from llamafactory.model import load_model, load_tokenizer
    except Exception as exc:
        message = (
            "Checkpoint probe loss could not import LLaMA-Factory internals. "
            f"{type(exc).__name__}: {exc}. Expected APIs: " + ", ".join(EXPECTED_API)
        )
        logger.error(message)
        warnings.append(WarningRecord(code="loss_import_failed", message=message))
        raise RuntimeError(message) from exc

    train_cfg = _load_yaml(training_config) if training_config else {}
    checkpoints = discover_checkpoints(checkpoint_root)
    steps = sorted({int(step) for step in frame["step"].dropna().tolist()})
    mapped = map_steps_to_checkpoints(steps, checkpoints, nearest=nearest_checkpoint, warnings=warnings)

    probe_ids = select_probe_ids(
        frame,
        max_per_dataset=max_samples_per_dataset,
        seed=seed,
        persist_path=output_dir / "probe_question_ids.json",
    )
    wanted: dict[str, set[str]] = {dataset: set(ids) for dataset, ids in probe_ids.items()}

    results: list[dict[str, Any]] = []

    for step, (ckpt_path, policy) in mapped.items():
        if ckpt_path is None:
            for dataset, ids in wanted.items():
                for qid in ids:
                    results.append(_skip_row(qid, dataset, step, "checkpoint_missing", policy))
            continue

        infer_kwargs = _infer_kwargs_from_train_yaml(train_cfg, ckpt_path)
        if model_name_or_path:
            infer_kwargs["model_name_or_path"] = model_name_or_path
        if "model_name_or_path" not in infer_kwargs:
            warnings.append(
                WarningRecord(
                    code="loss_missing_model",
                    message="Need --model-name-or-path or --training-config with model_name_or_path.",
                    step=step,
                )
            )
            for dataset, ids in wanted.items():
                for qid in ids:
                    results.append(_skip_row(qid, dataset, step, "other", "missing model_name_or_path"))
            continue

        try:
            model_args, data_args, finetuning_args, _gen = get_infer_args(infer_kwargs)
            tokenizer_module = load_tokenizer(model_args)
            tokenizer = tokenizer_module["tokenizer"]
            template = get_template_and_fix_tokenizer(tokenizer, data_args)
            model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
            model.eval()
            collator = SFTDataCollatorWith4DAttentionMask(
                template=template,
                model=model,
                label_pad_token_id=IGNORE_INDEX,
                pad_to_multiple_of=None,
                **tokenizer_module,
            )
            processor = SupervisedDatasetProcessor(
                template=template,
                tokenizer=tokenizer,
                processor=tokenizer_module.get("processor"),
                data_args=data_args,
            )
        except Exception as exc:
            logger.exception("Failed to load checkpoint %s", ckpt_path)
            warnings.append(
                WarningRecord(code="checkpoint_load_failed", message=str(exc), step=step, extra=str(ckpt_path))
            )
            for dataset, ids in wanted.items():
                for qid in ids:
                    results.append(_skip_row(qid, dataset, step, "other", str(exc)))
            continue

        try:
            for dataset, ids in wanted.items():
                spec = dataset_info.get(dataset) if isinstance(dataset_info.get(dataset), dict) else {}
                index = annotation_indices.get(dataset)
                formatting = spec.get("formatting", "alpaca") if spec else "alpaca"
                dataset_attr = DatasetAttr(load_from="file", dataset_name=dataset)
                if spec:
                    dataset_attr.join(spec)
                if not hasattr(dataset_attr, "question_id"):
                    dataset_attr.question_id = None
                converter = get_dataset_converter(formatting, dataset_attr, data_args)

                encoded_batch: list[dict[str, Any]] = []
                batch_meta: list[tuple[str, str]] = []
                for qid in sorted(ids):
                    record = None
                    if index is not None:
                        from _annotations import lookup_annotation

                        q_index = None
                        if "_" in qid:
                            try:
                                q_index = int(qid.rsplit("_", 1)[-1])
                            except ValueError:
                                q_index = None
                        record = lookup_annotation(index, qid, q_index)
                    if record is None:
                        results.append(_skip_row(qid, dataset, step, "annotation_missing"))
                        continue
                    try:
                        aligned = converter(record)
                    except ValueError as exc:
                        reason = "unresolvable_path"
                        msg = str(exc).lower()
                        if "image" in msg or "media" in msg:
                            reason = "missing_image"
                        results.append(_skip_row(qid, dataset, step, reason, str(exc)))
                        continue
                    except Exception as exc:
                        results.append(_skip_row(qid, dataset, step, "other", str(exc)))
                        continue
                    try:
                        encoded = processor.preprocess_dataset(
                            {
                                "_prompt": [aligned["_prompt"]],
                                "_response": [aligned["_response"]],
                                "_system": [aligned.get("_system") or ""],
                                "_tools": [aligned.get("_tools") or ""],
                                "_images": [aligned.get("_images")],
                                "_videos": [aligned.get("_videos")],
                                "_audios": [aligned.get("_audios")],
                                "_question_id": [qid],
                            }
                        )
                    except Exception as exc:
                        msg = str(exc).lower()
                        reason = "processor_error"
                        if "image" in msg:
                            reason = "missing_image"
                        elif "video" in msg:
                            reason = "missing_video"
                        results.append(_skip_row(qid, dataset, step, reason, str(exc)))
                        continue
                    if not encoded.get("input_ids"):
                        results.append(_skip_row(qid, dataset, step, "processor_error", "empty encode"))
                        continue
                    feature = {key: encoded[key][0] for key in encoded}
                    encoded_batch.append(feature)
                    batch_meta.append((qid, dataset))
                    if len(encoded_batch) >= max(loss_batch_size, 1):
                        results.extend(
                            _forward_probe_batch(
                                model, collator, encoded_batch, batch_meta, step, policy, IGNORE_INDEX, torch
                            )
                        )
                        encoded_batch, batch_meta = [], []
                if encoded_batch:
                    results.extend(
                        _forward_probe_batch(
                            model, collator, encoded_batch, batch_meta, step, policy, IGNORE_INDEX, torch
                        )
                    )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pd.DataFrame(results)


def _forward_probe_batch(
    model: Any,
    collator: Any,
    features: list[dict[str, Any]],
    meta: list[tuple[str, str]],
    step: int,
    policy: str,
    ignore_index: int,
    torch: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        batch = collator(features)
        batch = {
            key: value.cuda() if torch.is_tensor(value) and torch.cuda.is_available() else value
            for key, value in batch.items()
            if key != "question_ids"
        }
        labels = batch.get("labels")
        with torch.inference_mode():
            outputs = model(
                **{
                    key: value
                    for key, value in batch.items()
                    if key in ("input_ids", "attention_mask", "labels") or torch.is_tensor(value)
                }
            )
            logits = outputs.logits
            # Shift for causal LM.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            vocab = shift_logits.size(-1)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
            token_loss = loss_fct(shift_logits.view(-1, vocab), shift_labels.view(-1)).view_as(shift_labels)
            mask = shift_labels != ignore_index
            token_counts = mask.sum(dim=1)
            summed = (token_loss * mask).sum(dim=1)
        for i, (qid, dataset) in enumerate(meta):
            n_tok = int(token_counts[i].item())
            if n_tok <= 0:
                rows.append(_skip_row(qid, dataset, step, "other", "no target tokens"))
                continue
            mean_nll = float((summed[i] / n_tok).item())
            rows.append(
                {
                    "question_id": qid,
                    "dataset": dataset,
                    "step": step,
                    "probe_loss": mean_nll,
                    "probe_token_nll": mean_nll,
                    "target_token_count": n_tok,
                    "skip_reason": "",
                    "skip_detail": "",
                    "checkpoint_policy": policy,
                }
            )
    except torch.cuda.OutOfMemoryError as exc:
        logger.warning("OOM during probe loss at step %s: %s", step, exc)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for qid, dataset in meta:
            rows.append(_skip_row(qid, dataset, step, "OOM", str(exc)))
    except Exception as exc:
        logger.warning("Probe forward failed at step %s: %s", step, exc)
        for qid, dataset in meta:
            rows.append(_skip_row(qid, dataset, step, "other", str(exc)))
    return rows


def summarize_probe_loss(loss_frame: pd.DataFrame) -> pd.DataFrame:
    if loss_frame.empty:
        return loss_frame
    valid = loss_frame[loss_frame["probe_loss"].notna()].copy()
    rows = []
    if valid.empty:
        skip = loss_frame.groupby(["step", "dataset"])["skip_reason"].value_counts().rename("n").reset_index()
        return skip
    for (step, dataset), sub in valid.groupby(["step", "dataset"]):
        n = len(sub)
        token_counts = pd.to_numeric(sub["target_token_count"], errors="coerce").fillna(0)
        nll = pd.to_numeric(sub["probe_loss"], errors="coerce")
        token_weighted = (nll * token_counts).sum() / token_counts.sum() if token_counts.sum() else float("nan")
        rows.append(
            {
                "step": step,
                "dataset": dataset,
                "n_examples": n,
                "mean_loss": float(nll.mean()),
                "median_loss": float(nll.median()),
                "std_loss": float(nll.std(ddof=1)) if n > 1 else 0.0,
                "token_weighted_loss": float(token_weighted),
                "target_token_count_sum": int(token_counts.sum()),
            }
        )
    return pd.DataFrame(rows)
