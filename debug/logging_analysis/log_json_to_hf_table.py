"""Log LLaMA-Factory prediction JSON dumps to an existing W&B run.

Dumps are ``D[QUESTION_ID][step] = text``. ``eval_predictions*.json`` and
``train_predictions*.json`` share that shape. One directory is grouped into
one table per kind, with a numeric ``step`` column.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd


COLUMNS = ["ID", "dataset", "qid", "prediction", "step"]

# QUESTION_IDs are often hub paths + trailing row index, or short ids such as
# Scene30k_5 / SpatialSSRL_coldstart_18 / 3DThinker10k_5696.
_DATASET_PATTERNS = (
    (re.compile(r"scene30k", re.I), "Scene30k"),
    (re.compile(r"spatial[-_]?ssrl|sft-coldstart", re.I), "SpatialSSRL_coldstart"),
    (re.compile(r"3dthinker10k_cot|3dthinker", re.I), "3DThinker10k"),
)
_QID_RE = re.compile(r"_(\d+)$")
_SAMPLE_IDS = ("Scene30k_5", "SpatialSSRL_coldstart_18", "3DThinker10k_5696")


class PredictionDumpError(ValueError):
    """A prediction JSON file is not ``D[QUESTION_ID][step] = text``."""


def extract_dataset_and_qid(question: str) -> tuple[str, str | None]:
    """Return ``(dataset name, trailing question index)`` for a log key."""
    qid_match = _QID_RE.search(question)
    qid = qid_match.group(1) if qid_match else None
    for pattern, name in _DATASET_PATTERNS:
        if pattern.search(question):
            return name, qid
    return "UNKNOWN", qid


def table_key_for(path: Path) -> str:
    """W&B key for one prediction file. Epoch files of a kind share a key."""
    name = path.name
    if name.startswith("eval_predictions"):
        return "eval_predictions"
    if name.startswith("train_predictions"):
        return "train_predictions"
    return path.stem


def rows_from_prediction_file(path: Path) -> list[dict[str, object]]:
    """Flatten one dump into table rows. ``step`` is an ``int``."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise PredictionDumpError(f"{path}: expected a JSON object, got {type(data).__name__}")

    rows: list[dict[str, object]] = []
    for question_id, value in data.items():
        question = str(question_id)
        dataset, qid = extract_dataset_and_qid(question)
        if isinstance(value, str):
            raise PredictionDumpError(f"{path}: {question!r} is a bare string. Expected D[QUESTION_ID][step] = text.")
        if not isinstance(value, dict):
            raise PredictionDumpError(
                f"{path}: {question!r} has unsupported value type {type(value).__name__}. "
                "Expected D[QUESTION_ID][step] = text."
            )
        for step_key, text in value.items():
            step_text = step_key.strip() if isinstance(step_key, str) else None
            # JSON object keys are strings. Reject bools (int subclass) and anything that is not a base-10 integer.
            if isinstance(step_key, bool) or step_text is None or re.fullmatch(r"-?\d+", step_text) is None:
                raise PredictionDumpError(f"{path}: step key {step_key!r} for {question!r} is not an integer")
            step = int(step_text)
            if not isinstance(text, str):
                raise PredictionDumpError(
                    f"{path}: prediction for {question!r} at step {step} is {type(text).__name__}, not str"
                )
            rows.append(
                {
                    "ID": question,
                    "dataset": dataset,
                    "qid": qid,
                    "prediction": text,
                    "step": step,
                }
            )
    return rows


def frames_from_dir(directory: Path) -> dict[str, pd.DataFrame]:
    """Build one frame per table key from ``*predictions*.json`` in ``directory`` only."""
    directory = Path(directory)
    if not directory.is_dir():
        raise PredictionDumpError(f"Not a directory: {directory}")

    grouped: dict[str, list[dict[str, object]]] = {}
    for path in sorted(item for item in directory.glob("*predictions*.json") if item.is_file()):
        key = table_key_for(path)
        grouped.setdefault(key, []).extend(rows_from_prediction_file(path))

    frames: dict[str, pd.DataFrame] = {}
    for key, rows in grouped.items():
        frame = pd.DataFrame(rows, columns=COLUMNS)
        frame["step"] = frame["step"].astype("int64")
        frames[key] = frame
    return frames


def format_dry_run(frames: dict[str, pd.DataFrame]) -> str:
    """Summarize frames without contacting W&B."""
    if not frames:
        return "No *predictions*.json files found."
    lines: list[str] = []
    for key, frame in frames.items():
        steps = sorted(int(step) for step in frame["step"].unique()) if len(frame) else []
        step_text = "{" + ", ".join(str(step) for step in steps) + "}"
        lines.append(
            f"{key}: rows={len(frame)} step_dtype={frame['step'].dtype} "
            f"steps={step_text} columns={list(frame.columns)}"
        )
        for question_id in _SAMPLE_IDS:
            match = frame.loc[frame["ID"] == question_id]
            if match.empty:
                continue
            row = match.iloc[0]
            lines.append(f"  {question_id} -> dataset={row['dataset']} qid={row['qid']}")
    return "\n".join(lines)


def log_tables(frames: dict[str, pd.DataFrame], *, entity: str, project: str, run_id: str) -> None:
    """Append tables to the already-synced run. Do not pass ``step``; history already has those steps."""
    import wandb

    payload = {}
    for key, frame in frames.items():
        if frame.empty:
            print(f"Skipping empty table {key}")
            continue
        payload[key] = wandb.Table(dataframe=frame)
    if not payload:
        print("No prediction rows to log.")
        return

    # A leftover WANDB_MODE=offline would write a new local run instead of attaching to the synced one.
    os.environ["WANDB_MODE"] = "online"
    run = wandb.init(entity=entity, project=project, id=run_id, resume="must", mode="online")
    run.log(payload)
    run.finish()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log prediction JSON dumps to an existing W&B run.")
    parser.add_argument("--predictions-dir", required=True, type=Path, help="Directory to scan (not recursive).")
    parser.add_argument("--id", required=True, help="W&B run id, the same value passed to wandb sync --id.")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "cvis_tmu"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "llamafactory"))
    parser.add_argument("--dry-run", action="store_true", help="Print table summaries and do not call wandb.init.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        frames = frames_from_dir(args.predictions_dir)
    except PredictionDumpError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not frames:
        print(f"No *predictions*.json files in {args.predictions_dir}")
        return 0
    if args.dry_run:
        print(format_dry_run(frames))
        return 0
    try:
        log_tables(frames, entity=args.entity, project=args.project, run_id=args.id)
    except PredictionDumpError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
