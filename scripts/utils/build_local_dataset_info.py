#!/usr/bin/env python3
"""Build a node-local dataset_info.json with rewritten file_name paths.

Used by stage_node_local_datasets.sh so multi-node jobs can point LLaMA-Factory
at annotations staged under $SLURM_TMPDIR without mutating the shared
data/dataset_info.json. Offline only (stdlib).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = (
    "Scene30k",
    "SpatialSSRL_coldstart",
    "3DThinker10k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy selected entries from a dataset_info.json and rewrite file_name "
            "to local absolute paths."
        )
    )
    parser.add_argument(
        "--source-dataset-info",
        required=True,
        type=Path,
        help="Path to the shared data/dataset_info.json",
    )
    parser.add_argument(
        "--output-dataset-info",
        required=True,
        type=Path,
        help="Path to write the local dataset_info.json",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names to include (default: CoT mix).",
    )
    parser.add_argument(
        "--file-name",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override file_name for a dataset (repeatable). Example: "
        "Scene30k=/tmp/cot_stage/annotations/Scene30k.parquet",
    )
    parser.add_argument(
        "--full-copy",
        action="store_true",
        help="Write a full copy of dataset_info with overrides applied "
        "(default: only the requested datasets).",
    )
    return parser.parse_args()


def mapped_column_names(entry: dict[str, Any]) -> list[str]:
    columns = entry.get("columns") or {}
    return [str(v) for v in columns.values() if v]


def parquet_footer_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) < 8 or data[-4:] != b"PAR1":
        raise ValueError(f"not a parquet file: {path}")
    footer_len = int.from_bytes(data[-8:-4], "little")
    start = len(data) - 8 - footer_len
    if start < 0:
        raise ValueError(f"invalid parquet footer in {path}")
    return data[start : len(data) - 8]


def columns_present_in_file(path: Path, required: list[str]) -> list[str]:
    """Return required column names that are missing from *path* (stdlib only)."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        footer = parquet_footer_bytes(path)
        return [name for name in required if name.encode("utf-8") not in footer]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample = payload[0] if isinstance(payload, list) else payload
        keys = set(sample)
        return [name for name in required if name not in keys]
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
        if not first.strip():
            return list(required)
        keys = set(json.loads(first))
        return [name for name in required if name not in keys]
    raise ValueError(f"unsupported annotation suffix for column check: {path}")


def check_mapped_columns(dataset_info: dict[str, Any]) -> list[str]:
    """Return human-readable errors if mapped columns are missing from files."""
    errors: list[str] = []
    for name, entry in dataset_info.items():
        required = mapped_column_names(entry)
        if not required:
            continue
        file_name = entry.get("file_name")
        if not file_name:
            errors.append(f"{name}: missing file_name")
            continue
        path = Path(file_name)
        if not path.is_file():
            errors.append(f"{name}: file not found: {path}")
            continue
        try:
            missing = columns_present_in_file(path, required)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{name}: failed to inspect {path}: {exc}")
            continue
        if missing:
            errors.append(
                f"{name}: {path} is missing columns {missing} required by "
                f"dataset_info.json (mapped {entry.get('columns')}). "
                "Stage the formatted annotation (*.formatted.parquet/json/jsonl), "
                "not the raw internlm/Scene30K snapshot."
            )
    return errors


def parse_file_name_overrides(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--file-name must be NAME=PATH, got: {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise SystemExit(f"--file-name must be NAME=PATH, got: {item!r}")
        out[name] = path
    return out


def main() -> int:
    args = parse_args()
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    if not names:
        print("ERROR: no datasets requested", file=sys.stderr)
        return 1

    overrides = parse_file_name_overrides(args.file_name)
    missing_overrides = [n for n in names if n not in overrides]
    if missing_overrides:
        print(
            "ERROR: missing --file-name for: " + ", ".join(missing_overrides),
            file=sys.stderr,
        )
        return 1

    src_path: Path = args.source_dataset_info
    if not src_path.is_file():
        print(f"ERROR: source dataset_info not found: {src_path}", file=sys.stderr)
        return 1

    with src_path.open("r", encoding="utf-8") as f:
        source: dict[str, Any] = json.load(f)

    for name in names:
        if name not in source:
            print(f"ERROR: dataset {name!r} not in {src_path}", file=sys.stderr)
            return 1

    if args.full_copy:
        result: dict[str, Any] = json.loads(json.dumps(source))  # deep-ish copy
    else:
        result = {}

    for name in names:
        entry = json.loads(json.dumps(source[name]))
        entry["file_name"] = overrides[name]
        result[name] = entry

    out_path: Path = args.output_dataset_info
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote local dataset_info ({len(result)} entries) -> {out_path}")
    for name in names:
        print(f"  {name}: {overrides[name]}")

    column_errors = check_mapped_columns(result)
    if column_errors:
        for err in column_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
