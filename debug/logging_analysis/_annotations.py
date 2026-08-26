"""Load dataset_info.json, annotation files, and extract ground-truth text."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from _logparse import WarningRecord, make_question_id


logger = logging.getLogger(__name__)

BOXED_RE = re.compile(r"\\+boxed\{([^{}]*)\}")
RESPONSE_FALLBACK_FIELDS = (
    "output",
    "cot",
    "response",
    "answer",
    "answers",
    "best_trace_formatted",
    "text_output",
)
PROMPT_FALLBACK_FIELDS = (
    "question_with_image_tags",
    "instruction",
    "prompt",
    "question",
    "input",
    "text_input",
)


@dataclass
class AnnotationIndex:
    dataset: str
    path: Path | None
    by_question_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_index: dict[int, dict[str, Any]] = field(default_factory=dict)
    n_records: int = 0
    response_field: str | None = None
    prompt_field: str | None = None
    load_error: str | None = None


def load_dataset_info(path: str | Path) -> dict[str, Any]:
    info_path = Path(path)
    with info_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{info_path}: expected a JSON object, got {type(data)}")
    return data


def expand_dataset_path(path: str, hf_hub_cache: str | None = None, dataset_dir: str | Path | None = None) -> str:
    """Expand ``${HF_HUB_CACHE}`` / ``$HF_HOME`` and join relative names to dataset_dir."""
    if not path:
        return path
    expanded = os.path.expanduser(path)
    cache_root = hf_hub_cache or os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    hf_hub = os.environ.get("HF_HUB_CACHE") or hf_hub_cache or cache_root
    hf_home = os.environ.get("HF_HOME") or hf_hub_cache or cache_root

    def _repl(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name == "HF_HUB_CACHE":
            if not hf_hub:
                raise ValueError(
                    "file_name uses $HF_HUB_CACHE but neither --hf-hub-cache nor HF_HUB_CACHE/HF_HOME is set."
                )
            return hf_hub
        if not hf_home:
            raise ValueError("file_name uses $HF_HOME but neither --hf-hub-cache nor HF_HOME/HF_HUB_CACHE is set.")
        return hf_home

    pattern = re.compile(r"\$\{(HF_HUB_CACHE|HF_HOME)\}|\$(HF_HUB_CACHE|HF_HOME)(?![A-Za-z0-9_])")
    if pattern.search(expanded):
        # Local expander matches llamafactory.data.parser.expand_dataset_path
        # without importing the training stack (version checks, torch, etc.).
        expanded = pattern.sub(_repl, expanded)

    if not os.path.isabs(expanded) and dataset_dir is not None:
        candidate = Path(dataset_dir) / expanded
        if candidate.exists() or not Path(expanded).exists():
            expanded = str(candidate)
    return expanded


def mapped_column(spec: dict[str, Any] | None, column: str) -> str | None:
    if not spec:
        return None
    columns = spec.get("columns") if isinstance(spec, dict) else None
    if not isinstance(columns, dict):
        return None
    value = columns.get(column)
    return str(value) if value else None


def resolve_annotation_path(
    dataset: str,
    dataset_info: dict[str, Any],
    *,
    hf_hub_cache: str | None = None,
    dataset_dir: str | Path | None = None,
    annotation_overrides: dict[str, str] | None = None,
) -> Path | None:
    if annotation_overrides and dataset in annotation_overrides:
        return Path(expand_dataset_path(annotation_overrides[dataset], hf_hub_cache, dataset_dir))
    spec = dataset_info.get(dataset)
    if not isinstance(spec, dict) or not spec.get("file_name"):
        return None
    return Path(expand_dataset_path(str(spec["file_name"]), hf_hub_cache, dataset_dir))


def _record_from_mapping_item(key: Any, value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        record = dict(value)
        record.setdefault("question_id", str(key))
        return record
    if isinstance(value, str):
        return {"question_id": str(key), "output": value}
    return None


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        records = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise TypeError(f"{path}: JSON list item {i} is {type(item)}, expected dict")
            records.append(item)
        return records
    if isinstance(data, dict):
        list_keys = [
            key
            for key, value in data.items()
            if isinstance(value, list) and value and all(isinstance(x, dict) for x in value)
        ]
        dict_of_records = all(isinstance(v, dict | str) for v in data.values()) and data
        if len(list_keys) == 1 and not (dict_of_records and list_keys[0] in {"question_id"}):
            # Common wrapper: {"data": [ {...}, ... ]}
            return list(data[list_keys[0]])
        if len(list_keys) > 1:
            raise ValueError(
                f"{path}: ambiguous JSON object with multiple list-of-record fields {list_keys}. "
                "Pass a file that is a list, JSONL, parquet, or a dict of id -> record."
            )
        records = []
        for key, value in data.items():
            record = _record_from_mapping_item(key, value)
            if record is None:
                raise TypeError(f"{path}: unsupported JSON object value type {type(value)} at key {key!r}")
            records.append(record)
        return records
    raise TypeError(f"{path}: unsupported JSON structure {type(data)}")


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{line_no}: expected JSON object, got {type(obj)}")
            records.append(obj)
    return records


def _load_parquet_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to read parquet annotation files") from exc
    frame = pd.read_parquet(path)
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        cleaned = {}
        for key, value in row.items():
            if hasattr(value, "tolist") and not isinstance(value, bytes | str):
                try:
                    value = value.tolist()
                except Exception:
                    pass
            cleaned[key] = value
        records.append(cleaned)
    return records


def load_annotation_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl_records(path)
    if suffix == ".parquet":
        return _load_parquet_records(path)
    if suffix == ".json":
        return _load_json_records(path)
    # Sniff.
    with path.open("rb") as handle:
        head = handle.read(2048).lstrip()
    if head.startswith(b"["):
        return _load_json_records(path)
    if head.startswith(b"{"):
        return _load_json_records(path)
    return _load_jsonl_records(path)


def index_annotations(
    records: list[dict[str, Any]],
    dataset: str,
    path: Path | None = None,
    response_field: str | None = None,
    prompt_field: str | None = None,
) -> AnnotationIndex:
    index = AnnotationIndex(
        dataset=dataset,
        path=path,
        n_records=len(records),
        response_field=response_field,
        prompt_field=prompt_field,
    )
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        index.by_index[i] = record
        qid = record.get("question_id")
        if qid is None:
            qid = record.get("QUESTION_ID")
        if qid is not None and str(qid).strip() != "":
            index.by_question_id[str(qid)] = record
        constructed = make_question_id(dataset, i)
        index.by_question_id.setdefault(constructed, record)
    return index


def lookup_annotation(
    index: AnnotationIndex | None, question_id: str, question_index: int | None
) -> dict[str, Any] | None:
    if index is None:
        return None
    if question_id in index.by_question_id:
        return index.by_question_id[question_id]
    if question_index is not None and question_index in index.by_index:
        return index.by_index[question_index]
    return None


def _string_field(record: dict[str, Any], name: str | None) -> str | None:
    if not name or name not in record:
        return None
    value = record[name]
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return None  # multi-reference handled separately
    return None


def extract_boxed_answers(text: str | None) -> list[str]:
    if not text:
        return []
    return [match.strip() for match in BOXED_RE.findall(text) if match.strip()]


def _assistant_from_messages(record: dict[str, Any]) -> str | None:
    for key in ("conversations", "messages"):
        messages = record.get(key)
        if not isinstance(messages, list) or not messages:
            continue
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("from") or "").lower()
            if role in {"assistant", "gpt", "model"}:
                content = message.get("content") or message.get("value")
                if isinstance(content, str):
                    return content
        last = messages[-1]
        if isinstance(last, dict):
            content = last.get("content") or last.get("value")
            if isinstance(content, str):
                return content
    return None


def extract_ground_truth_text(
    record: dict[str, Any] | None,
    *,
    preferred_field: str | None = None,
    warnings: list[WarningRecord] | None = None,
    log_key: str | None = None,
) -> tuple[str | None, str | None, list[str]]:
    """Return (text, field_used, extra_reference_strings).

    Does not silently pick among several distinct plausible fields.
    """
    if record is None:
        return None, None, []

    extra_refs: list[str] = []
    if preferred_field:
        if preferred_field in record:
            value = record[preferred_field]
            if isinstance(value, str):
                return value, preferred_field, extra_refs
            if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
                return value[0], preferred_field, list(value)
        if warnings is not None:
            warnings.append(
                WarningRecord(
                    code="gt_field_missing",
                    message=f"Preferred ground-truth field {preferred_field!r} missing or not text.",
                    log_key=log_key,
                )
            )

    present: list[tuple[str, str]] = []
    for name in RESPONSE_FALLBACK_FIELDS:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            present.append((name, value))
        elif isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            present.append((name, value[0]))
            extra_refs.extend(value)

    unique_texts = {text for _, text in present}
    if len(present) == 1:
        return present[0][1], present[0][0], extra_refs
    if len(present) > 1 and len(unique_texts) == 1:
        return present[0][1], present[0][0], extra_refs
    if len(present) > 1:
        if warnings is not None:
            warnings.append(
                WarningRecord(
                    code="ambiguous_gt_field",
                    message="Multiple distinct ground-truth fields present; pass --gt-field. Found: "
                    + ",".join(name for name, _ in present),
                    log_key=log_key,
                    extra=",".join(name for name, _ in present),
                )
            )
        return None, None, extra_refs

    messages_text = _assistant_from_messages(record)
    if messages_text:
        return messages_text, "messages", extra_refs
    return None, None, extra_refs


def extract_prompt_text(record: dict[str, Any] | None, preferred_field: str | None = None) -> str | None:
    if record is None:
        return None
    if preferred_field:
        text = _string_field(record, preferred_field)
        if text is not None:
            return text
    for name in PROMPT_FALLBACK_FIELDS:
        text = _string_field(record, name)
        if text is not None:
            return text
    return None


def load_annotation_indices(
    datasets: Iterable[str],
    dataset_info: dict[str, Any],
    *,
    hf_hub_cache: str | None = None,
    dataset_dir: str | Path | None = None,
    annotation_overrides: dict[str, str] | None = None,
    gt_field_overrides: dict[str, str] | None = None,
    warnings: list[WarningRecord] | None = None,
) -> dict[str, AnnotationIndex]:
    indices: dict[str, AnnotationIndex] = {}
    for dataset in sorted(set(datasets)):
        if dataset == "UNKNOWN":
            continue
        spec = dataset_info.get(dataset) if isinstance(dataset_info.get(dataset), dict) else {}
        response_field = (gt_field_overrides or {}).get(dataset) or mapped_column(spec, "response")
        prompt_field = mapped_column(spec, "prompt")
        try:
            path = resolve_annotation_path(
                dataset,
                dataset_info,
                hf_hub_cache=hf_hub_cache,
                dataset_dir=dataset_dir,
                annotation_overrides=annotation_overrides,
            )
        except ValueError as exc:
            logger.warning("Could not resolve annotation path for %s: %s", dataset, exc)
            if warnings is not None:
                warnings.append(WarningRecord(code="annotation_path_unresolved", message=str(exc), extra=dataset))
            indices[dataset] = AnnotationIndex(
                dataset=dataset, path=None, load_error=str(exc), response_field=response_field
            )
            continue
        if path is None:
            message = f"No file_name in dataset_info for {dataset}"
            logger.warning(message)
            if warnings is not None:
                warnings.append(WarningRecord(code="annotation_path_missing", message=message, extra=dataset))
            indices[dataset] = AnnotationIndex(
                dataset=dataset, path=None, load_error=message, response_field=response_field
            )
            continue
        if not path.exists():
            message = f"Annotation file does not exist: {path}"
            logger.warning(message)
            if warnings is not None:
                warnings.append(WarningRecord(code="annotation_file_missing", message=message, extra=dataset))
            indices[dataset] = AnnotationIndex(
                dataset=dataset, path=path, load_error=message, response_field=response_field
            )
            continue
        try:
            records = load_annotation_records(path)
        except Exception as exc:
            message = f"Failed to load {path}: {exc}"
            logger.warning(message)
            if warnings is not None:
                warnings.append(WarningRecord(code="annotation_load_failed", message=message, extra=dataset))
            indices[dataset] = AnnotationIndex(
                dataset=dataset, path=path, load_error=message, response_field=response_field
            )
            continue
        logger.info("Loaded %s annotation records for %s from %s", len(records), dataset, path)
        indices[dataset] = index_annotations(
            records,
            dataset,
            path=path,
            response_field=response_field,
            prompt_field=prompt_field,
        )
    return indices


def parse_kv_overrides(values: list[str] | None) -> dict[str, str]:
    """Parse CLI items of the form NAME=VALUE."""
    out: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Expected NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"Empty name in override {item!r}")
        out[name] = value
    return out
