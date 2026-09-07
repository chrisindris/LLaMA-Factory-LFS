"""Discover eval save folders and load trainer_log / eval_results.

Standalone LLaMA-Factory eval dumps typically contain::

    eval_predictions.json
    eval_results.json
    all_results.json
    trainer_log.jsonl  # TRAINER_LOG; some callers say trainer_log.json
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from _logparse import EPOCH_IN_NAME_RE, WarningRecord, infer_run_name_from_path


logger = logging.getLogger(__name__)

EPOCHS_IN_NAME_RE = re.compile(r"(?:^|[^0-9])(\d+)epochs?(?:$|[^a-z0-9])", re.IGNORECASE)
SKIP_RUN_NAME_PARTS = frozenset({"eval", "lora", "sft", "output", "outputs", "saves"})
TRAINER_LOG_NAMES = ("trainer_log.jsonl", "trainer_log.json")


@dataclass
class EvalDirFiles:
    path: Path
    predictions: list[Path] = field(default_factory=list)
    trainer_log: Path | None = None
    eval_results: Path | None = None
    all_results: Path | None = None


@dataclass
class AnalysisRun:
    run_name: str
    step: int | None = None
    eval_dir: Path | None = None
    prediction_path: Path | None = None
    trainer_log_path: Path | None = None
    eval_results_path: Path | None = None
    explicit_name: bool = False


def parse_named_path(item: str | Path) -> tuple[str | None, Path]:
    """Parse ``NAME=PATH`` or a bare path.

    Unlike annotation overrides, a bare path is allowed. A left-hand side that
    looks like a path (contains ``/`` or starts with ``.``) is not treated as a name.
    """
    raw = str(item)
    if "=" in raw:
        name, value = raw.split("=", 1)
        name = name.strip()
        value = value.strip()
        if name and "/" not in name and "\\" not in name and not name.startswith("."):
            return name, Path(value)
    return None, Path(raw)


def parse_eval_steps(values: list[str] | None) -> dict[str, int]:
    """Parse repeatable ``--eval-step NAME=INT`` items."""
    out: dict[str, int] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Expected NAME=INT for --eval-step, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"Empty name in --eval-step {item!r}")
        try:
            out[name] = int(value)
        except ValueError as exc:
            raise ValueError(f"Expected integer step in --eval-step {item!r}") from exc
    return out


def infer_run_name(path: str | Path) -> str:
    """Pick a short label from an eval folder or prediction file path."""
    return infer_run_name_from_path(path, skip_parts=SKIP_RUN_NAME_PARTS)


def find_trainer_log(directory: Path | None) -> Path | None:
    if directory is None or not directory.is_dir():
        return None
    for name in TRAINER_LOG_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def find_eval_results(directory: Path | None) -> Path | None:
    if directory is None or not directory.is_dir():
        return None
    for name in ("eval_results.json", "all_results.json"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def discover_eval_dir(path: str | Path) -> EvalDirFiles:
    folder = Path(path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Eval dir is not a directory: {folder}")
    exact = folder / "eval_predictions.json"
    if exact.is_file():
        predictions = [exact]
    else:
        predictions = sorted(p for p in folder.glob("eval_predictions*.json") if p.is_file())
    eval_results = folder / "eval_results.json"
    all_results = folder / "all_results.json"
    return EvalDirFiles(
        path=folder,
        predictions=predictions,
        trainer_log=find_trainer_log(folder),
        eval_results=eval_results if eval_results.is_file() else None,
        all_results=all_results if all_results.is_file() else None,
    )


def epoch_from_run_label(*parts: str | Path | None) -> int | None:
    """Return N from ``_epN.json`` or ``Nepochs`` in any path/name fragment."""
    for part in parts:
        if part is None:
            continue
        text = str(part)
        match = EPOCH_IN_NAME_RE.search(Path(text).name)
        if match:
            return int(match.group(1))
        match = EPOCHS_IN_NAME_RE.search(text)
        if match:
            return int(match.group(1))
    return None


def needs_synthetic_step(run: AnalysisRun) -> bool:
    """Train dumps keep inner global_step keys; eval dumps need an assigned step."""
    if run.prediction_path is None:
        return True
    name = run.prediction_path.name.lower()
    if "train" in name and "eval" not in name:
        return False
    if "eval" in name:
        return True
    return run.eval_dir is not None


def _unique_run_names(runs: list[AnalysisRun]) -> None:
    seen: dict[str, int] = {}
    for run in runs:
        base = run.run_name
        count = seen.get(base, 0) + 1
        seen[base] = count
        if count == 1:
            continue
        run.run_name = f"{base}_{count}"
        logger.warning("Duplicate run_name %r; renamed later run to %s", base, run.run_name)


def assign_run_steps(
    runs: list[AnalysisRun],
    eval_step_overrides: dict[str, int] | None,
    warnings: list[WarningRecord],
) -> None:
    """Give each eval run a distinct numeric step so aggregations do not pool them."""
    overrides = eval_step_overrides or {}
    used: dict[int, str] = {}
    cli_index = 0
    for run in runs:
        if not needs_synthetic_step(run):
            continue
        step: int | None = None
        if run.run_name in overrides:
            step = overrides[run.run_name]
        if step is None:
            step = epoch_from_run_label(run.prediction_path, run.run_name, run.eval_dir)
        if step is None:
            step = cli_index
        original = step
        while step in used:
            warnings.append(
                WarningRecord(
                    code="duplicate_eval_step",
                    message=(
                        f"Eval run {run.run_name!r} would share step {step} with {used[step]!r}; using {step + 1}."
                    ),
                    step=step,
                    extra=run.run_name,
                )
            )
            step += 1
        if original != step:
            logger.warning("Bumped step for run %s from %s to %s", run.run_name, original, step)
        used[step] = run.run_name
        run.step = step
        cli_index += 1


def _runs_from_eval_dir(
    folder: Path,
    *,
    name: str | None,
    explicit: bool,
    warnings: list[WarningRecord],
) -> list[AnalysisRun]:
    files = discover_eval_dir(folder)
    eval_results = files.eval_results or files.all_results
    run_name = name or infer_run_name(folder)
    if not files.predictions:
        warnings.append(
            WarningRecord(
                code="eval_predictions_missing",
                message=f"No eval_predictions.json in {folder}",
                extra=str(folder),
            )
        )
        if files.trainer_log is None and eval_results is None:
            warnings.append(
                WarningRecord(
                    code="empty_eval_dir",
                    message=f"Eval dir has neither predictions nor trainer_log: {folder}",
                    extra=str(folder),
                )
            )
            return []
        return [
            AnalysisRun(
                run_name=run_name,
                eval_dir=folder,
                prediction_path=None,
                trainer_log_path=files.trainer_log,
                eval_results_path=eval_results,
                explicit_name=explicit,
            )
        ]
    return [
        AnalysisRun(
            run_name=run_name,
            eval_dir=folder,
            prediction_path=pred,
            trainer_log_path=files.trainer_log,
            eval_results_path=eval_results,
            explicit_name=explicit,
        )
        for pred in files.predictions
    ]


def resolve_analysis_runs(
    logs: list[str] | None,
    eval_dirs: list[str] | None,
    eval_steps: list[str] | None,
    warnings: list[WarningRecord],
) -> list[AnalysisRun]:
    """Turn ``--log`` / ``--eval-dir`` / ``--eval-step`` into labeled runs."""
    runs: list[AnalysisRun] = []
    seen_predictions: set[str] = set()

    def _remember(run: AnalysisRun) -> None:
        if run.prediction_path is not None:
            key = str(run.prediction_path.resolve()) if run.prediction_path.exists() else str(run.prediction_path)
            if key in seen_predictions:
                logger.info("Skipping duplicate prediction path %s", run.prediction_path)
                return
            seen_predictions.add(key)
        runs.append(run)

    for item in logs or []:
        path = Path(item)
        if path.is_dir():
            for run in _runs_from_eval_dir(path, name=None, explicit=False, warnings=warnings):
                _remember(run)
            continue
        parent = path.parent if path.parent.as_posix() not in ("", ".") else path.parent
        _remember(
            AnalysisRun(
                run_name=infer_run_name(path),
                eval_dir=parent if parent.is_dir() else None,
                prediction_path=path,
                trainer_log_path=find_trainer_log(parent) if parent.is_dir() else None,
                eval_results_path=find_eval_results(parent) if parent.is_dir() else None,
            )
        )

    for item in eval_dirs or []:
        name, folder = parse_named_path(item)
        if not folder.exists():
            warnings.append(
                WarningRecord(
                    code="eval_dir_missing",
                    message=f"Eval dir does not exist: {folder}",
                    extra=str(folder),
                )
            )
            continue
        for run in _runs_from_eval_dir(folder, name=name, explicit=bool(name), warnings=warnings):
            _remember(run)

    _unique_run_names(runs)
    assign_run_steps(runs, parse_eval_steps(eval_steps), warnings)
    return runs


def load_trainer_log(path: str | Path) -> list[dict[str, Any]]:
    """Load ``trainer_log.jsonl``, a JSON array, or a single JSON object."""
    log_path = Path(path)
    text = log_path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if log_path.suffix.lower() != ".jsonl":
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            history = data.get("log_history")
            if isinstance(history, list):
                return [row for row in history if isinstance(row, dict)]
            return [data]
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(stripped.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("%s line %s: invalid JSON (%s)", log_path, line_no, exc)
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_eval_results(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a JSON object of eval metrics, got {type(data)}")
    return data


def summarize_trainer_log(rows: list[dict[str, Any]]) -> dict[str, Any]:
    last_eval: dict[str, Any] | None = None
    last_row: dict[str, Any] = rows[-1] if rows else {}
    for row in rows:
        if row.get("eval_loss") is not None:
            last_eval = row
    src = last_eval or last_row
    return {
        "eval_loss": src.get("eval_loss"),
        "current_steps": src.get("current_steps", src.get("step")),
        "total_steps": src.get("total_steps"),
        "elapsed_time": src.get("elapsed_time"),
        "remaining_time": src.get("remaining_time"),
        "percentage": src.get("percentage"),
        "n_log_lines": len(rows),
    }


def collect_trainer_tables(
    runs: list[AnalysisRun],
    warnings: list[WarningRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (summary rows, flattened history rows) for all runs."""
    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    seen_summary: set[tuple[str, str | None]] = set()
    for run in runs:
        history: list[dict[str, Any]] = []
        if run.trainer_log_path is not None and run.trainer_log_path.is_file():
            history = load_trainer_log(run.trainer_log_path)
        else:
            warnings.append(
                WarningRecord(
                    code="trainer_log_missing",
                    message=f"No trainer_log.jsonl / trainer_log.json for run {run.run_name}",
                    extra=str(run.eval_dir or run.prediction_path),
                )
            )
        tsum = summarize_trainer_log(history)
        results: dict[str, Any] = {}
        if run.eval_results_path is not None and run.eval_results_path.is_file():
            try:
                results = load_eval_results(run.eval_results_path)
            except (OSError, TypeError, json.JSONDecodeError) as exc:
                warnings.append(
                    WarningRecord(
                        code="eval_results_unreadable",
                        message=f"Could not read {run.eval_results_path}: {exc}",
                        extra=str(run.eval_results_path),
                    )
                )
        eval_loss = results.get("eval_loss")
        if eval_loss is None:
            eval_loss = tsum.get("eval_loss")
        key = (run.run_name, str(run.trainer_log_path) if run.trainer_log_path else None)
        if key not in seen_summary:
            seen_summary.add(key)
            summary_rows.append(
                {
                    "run_name": run.run_name,
                    "step": run.step,
                    "eval_dir": str(run.eval_dir) if run.eval_dir else None,
                    "eval_predictions": str(run.prediction_path) if run.prediction_path else None,
                    "trainer_log": str(run.trainer_log_path) if run.trainer_log_path else None,
                    "eval_results": str(run.eval_results_path) if run.eval_results_path else None,
                    "eval_loss": eval_loss,
                    "eval_runtime": results.get("eval_runtime"),
                    "eval_samples_per_second": results.get("eval_samples_per_second"),
                    "eval_steps_per_second": results.get("eval_steps_per_second"),
                    "eval_model_preparation_time": results.get("eval_model_preparation_time"),
                    "elapsed_time": tsum.get("elapsed_time"),
                    "n_trainer_log_lines": tsum.get("n_log_lines"),
                    "trainer_current_steps": tsum.get("current_steps"),
                    "trainer_total_steps": tsum.get("total_steps"),
                }
            )
        for row in history:
            history_rows.append({"run_name": run.run_name, "run_step": run.step, **row})
    return summary_rows, history_rows
