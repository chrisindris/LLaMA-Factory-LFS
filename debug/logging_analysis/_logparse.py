"""Parse LLaMA-Factory prediction dumps and recover dataset / question IDs.

Train dumps are ``D[QUESTION_ID][global_step] = text``. In this fork the
QUESTION_ID is often a staged annotation path plus a trailing file-order
index, for example::

    /tmp/cot_stage/annotations/SpatialSSRL_coldstart.json_1661

because ``dataset_info.json`` does not map ``columns.question_id`` and
``align_dataset`` falls back to ``{file_path}_{row_index}``.

Eval dumps are ``D[QUESTION_ID] = text`` (no inner step dict).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

QUESTION_INDEX_RE = re.compile(r"_(\d+)$")
EPOCH_IN_NAME_RE = re.compile(r"_ep(\d+)\.json$", re.IGNORECASE)

# Canonical names match data/dataset_info.json keys.
BUILTIN_ALIASES: dict[str, list[str]] = {
    "Scene30k": ["Scene30k"],
    "3DThinker10k": ["3DThinker10k", "3dthinker10k", "3DThinker", "3dthinker10k_cot"],
    "SpatialSSRL_coldstart": ["SpatialSSRL_coldstart", "SpatialSSRL"],
}

UNKNOWN_DATASET = "UNKNOWN"


@dataclass
class WarningRecord:
    code: str
    message: str
    log_key: str | None = None
    step: int | None = None
    extra: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "log_key": self.log_key,
            "step": self.step,
            "extra": self.extra,
        }


@dataclass
class ParsedObservation:
    log_key: str
    step: int | None
    prediction: Any
    source_log: str
    source_kind: str
    epoch: str | None
    dataset: str
    question_index: int | None
    question_id: str
    matched_alias: str | None = None
    ambiguous_datasets: list[str] = field(default_factory=list)
    parse_notes: str | None = None


def epoch_from_log_path(path: str | Path) -> str | None:
    match = EPOCH_IN_NAME_RE.search(Path(path).name)
    return match.group(1) if match else None


def infer_source_kind(path: str | Path, payload: dict[str, Any]) -> str:
    name = Path(path).name.lower()
    if "eval" in name:
        return "eval"
    if "train" in name:
        return "train"
    if not payload:
        return "train"
    first_val = next(iter(payload.values()))
    if isinstance(first_val, dict):
        return "train"
    return "eval"


def load_prediction_log(path: str | Path) -> dict[str, Any]:
    log_path = Path(path)
    with log_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{log_path}: expected a JSON object mapping QUESTION_ID -> predictions, got {type(data)}")
    return data


def _iter_step_predictions(inner: Any) -> Iterator[tuple[int | None, Any, str | None]]:
    """Yield (step, prediction, note) from a dump value."""
    if isinstance(inner, dict):
        if not inner:
            yield None, None, "empty_inner_dict"
            return
        for step_key, prediction in inner.items():
            note = None
            try:
                step = int(step_key)
            except (TypeError, ValueError):
                note = f"non_integer_step:{step_key!r}"
                step = None
            yield step, prediction, note
        return
    # Eval-style: a bare string (or other scalar).
    yield None, inner, None


def file_name_aliases(file_name: str | None) -> list[str]:
    if not file_name:
        return []
    base = Path(str(file_name)).name
    aliases = [base]
    stem = base
    for suffix in (".jsonl", ".json", ".parquet", ".csv", ".arrow"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            aliases.append(stem)
            break
    aliases.append(stem.replace(".with_question_id", ""))
    aliases.append(base.replace(".with_question_id", ""))
    cleaned: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        alias = alias.strip()
        if not alias or alias in seen:
            continue
        seen.add(alias)
        cleaned.append(alias)
    return cleaned


def build_matchers(
    dataset_info: dict[str, Any] | None = None,
    extra_aliases: dict[str, list[str]] | None = None,
) -> list[tuple[str, str]]:
    """Return (alias, canonical) pairs sorted by alias length descending."""
    grouped: dict[str, list[str]] = {}
    for canonical, aliases in BUILTIN_ALIASES.items():
        grouped.setdefault(canonical, [])
        grouped[canonical].extend(aliases)
    if dataset_info:
        for name, spec in dataset_info.items():
            grouped.setdefault(name, [])
            grouped[name].append(name)
            if isinstance(spec, dict):
                grouped[name].extend(file_name_aliases(spec.get("file_name")))
    if extra_aliases:
        for canonical, aliases in extra_aliases.items():
            grouped.setdefault(canonical, [])
            grouped[canonical].extend(aliases)

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for canonical, aliases in grouped.items():
        for alias in aliases:
            alias_key = str(alias).strip()
            if not alias_key:
                continue
            item = (alias_key.lower(), canonical)
            if item in seen:
                continue
            seen.add(item)
            pairs.append((alias_key, canonical))
    pairs.sort(key=lambda item: (-len(item[0]), item[1].lower()))
    return pairs


def identify_dataset(log_key: str, matchers: list[tuple[str, str]]) -> tuple[str, str | None, list[str]]:
    """Return (canonical, matched_alias, other_canonical_hits)."""
    key_lower = log_key.lower()
    hits: list[tuple[str, str]] = []
    seen_canonical: set[str] = set()
    for alias, canonical in matchers:
        if alias.lower() in key_lower and canonical not in seen_canonical:
            hits.append((alias, canonical))
            seen_canonical.add(canonical)
    if not hits:
        return UNKNOWN_DATASET, None, []
    winner_alias, winner = hits[0]
    others = [canonical for _, canonical in hits[1:]]
    return winner, winner_alias, others


def parse_question_index(log_key: str) -> int | None:
    match = QUESTION_INDEX_RE.search(log_key)
    if not match:
        return None
    return int(match.group(1))


def make_question_id(dataset: str, question_index: int | None) -> str:
    if question_index is None:
        return dataset
    return f"{dataset}_{question_index}"


def flatten_prediction_log(
    data: dict[str, Any],
    source_log: str | Path,
    matchers: list[tuple[str, str]],
    warnings: list[WarningRecord],
    source_kind: str | None = None,
    epoch: str | None = None,
) -> list[ParsedObservation]:
    source_log_str = str(source_log)
    kind = source_kind or infer_source_kind(source_log_str, data)
    epoch_val = epoch if epoch is not None else epoch_from_log_path(source_log_str)
    rows: list[ParsedObservation] = []

    for log_key, inner in data.items():
        key = str(log_key)
        dataset, matched_alias, ambiguous = identify_dataset(key, matchers)
        if dataset == UNKNOWN_DATASET:
            warnings.append(
                WarningRecord(
                    code="unknown_dataset",
                    message="Could not determine dataset from log key; labeled UNKNOWN.",
                    log_key=key,
                )
            )
            logger.warning("Unknown dataset for log key: %s", key)
        elif ambiguous:
            warnings.append(
                WarningRecord(
                    code="ambiguous_dataset",
                    message=f"Multiple datasets matched {key!r}; using {dataset}. Also matched: {ambiguous}",
                    log_key=key,
                    extra=",".join(ambiguous),
                )
            )
            logger.warning("Ambiguous dataset for %s: chose %s over %s", key, dataset, ambiguous)

        question_index = parse_question_index(key)
        if question_index is None:
            warnings.append(
                WarningRecord(
                    code="malformed_outer_key",
                    message="Log key does not end with _<integer> question index.",
                    log_key=key,
                )
            )
        question_id = make_question_id(dataset, question_index)

        n_inner = 0
        for step, prediction, note in _iter_step_predictions(inner):
            n_inner += 1
            if note == "empty_inner_dict":
                warnings.append(
                    WarningRecord(
                        code="malformed_inner_dict",
                        message="Inner prediction dict is empty.",
                        log_key=key,
                    )
                )
            if note and note.startswith("non_integer_step"):
                warnings.append(
                    WarningRecord(
                        code="non_integer_step",
                        message=f"Could not convert step key to int: {note}",
                        log_key=key,
                    )
                )
            if not isinstance(prediction, str) and prediction is not None:
                warnings.append(
                    WarningRecord(
                        code="non_string_prediction",
                        message=f"Prediction is {type(prediction).__name__}, not str.",
                        log_key=key,
                        step=step,
                    )
                )
            if kind == "eval" and step is None:
                # Use epoch as a synthetic step so aggregations still have an axis.
                try:
                    step = int(epoch_val) if epoch_val is not None else 0
                except ValueError:
                    step = 0
            rows.append(
                ParsedObservation(
                    log_key=key,
                    step=step,
                    prediction=prediction
                    if isinstance(prediction, str)
                    else ("" if prediction is None else str(prediction)),
                    source_log=source_log_str,
                    source_kind=kind,
                    epoch=epoch_val,
                    dataset=dataset,
                    question_index=question_index,
                    question_id=question_id,
                    matched_alias=matched_alias,
                    ambiguous_datasets=ambiguous,
                    parse_notes=note,
                )
            )
        if isinstance(inner, dict) and n_inner > 1:
            warnings.append(
                WarningRecord(
                    code="multiple_steps",
                    message=f"Log entry contains {n_inner} step/prediction pairs.",
                    log_key=key,
                )
            )
    return rows


def load_and_flatten_logs(
    log_paths: Iterable[str | Path],
    matchers: list[tuple[str, str]],
    warnings: list[WarningRecord],
) -> list[ParsedObservation]:
    observations: list[ParsedObservation] = []
    for path in log_paths:
        data = load_prediction_log(path)
        observations.extend(flatten_prediction_log(data, path, matchers, warnings))
    return observations
