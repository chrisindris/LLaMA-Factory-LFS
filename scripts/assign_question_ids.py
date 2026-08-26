#!/usr/bin/env python3
"""Assign stable QUESTION_IDs to dataset annotation files.

IDs follow ``{DATASET_NAME}_{NUMBER}`` where NUMBER is the 0-based index in
file order (JSON list / JSONL / Parquet row order). Existing fields such as
``idx`` are ignored so IDs stay reproducible from appearance order.

Examples
--------
Single file (writes a sibling ``*.with_question_id.*`` by default)::

    python scripts/assign_question_ids.py \\
      --dataset-name Scene30k \\
      --input /path/to/train.parquet

In-place::

    python scripts/assign_question_ids.py \\
      --dataset-name SpatialSSRL_coldstart \\
      --input /path/to/SFT-coldstart.json \\
      --in-place

Batch from dataset_info.json::

    python scripts/assign_question_ids.py \\
      --from-dataset-info data/dataset_info.json \\
      --datasets Scene30k,SpatialSSRL_coldstart,3DThinker10k
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional


def default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    # handle multi-suffix names like train-00000-of-00001.parquet
    return input_path.with_name(f"{stem}.with_question_id{input_path.suffix}")


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".parquet":
        return "parquet"
    # sniff JSON vs JSONL for odd names
    with open(path, "rb") as f:
        head = f.read(2048).lstrip()
    if head.startswith(b"["):
        return "json"
    return "jsonl"


def make_id(dataset_name: str, index: int) -> str:
    return f"{dataset_name}_{index}"


def assign_json_list(rows: list[dict[str, Any]], dataset_name: str, column: str, start: int) -> int:
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"Expected dict rows in JSON list; got {type(row)} at index {i}")
        row[column] = make_id(dataset_name, start + i)
    return start + len(rows)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"Expected JSON object on line {line_no} of {path}")
            yield obj


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def process_json(input_path: Path, output_path: Path, dataset_name: str, column: str, start: int) -> int:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{input_path}: expected a JSON list of objects, got {type(data)}")
    end = assign_json_list(data, dataset_name, column, start)
    _atomic_write_text(output_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return end - start


def process_jsonl(input_path: Path, output_path: Path, dataset_name: str, column: str, start: int) -> int:
    tmp_fd, tmp_name = tempfile.mkstemp(prefix="qid_", suffix=".jsonl", dir=str(output_path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        count = 0
        with open(tmp_path, "w", encoding="utf-8") as out:
            for i, row in enumerate(iter_jsonl(input_path)):
                row[column] = make_id(dataset_name, start + i)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        os.replace(tmp_path, output_path)
        return count
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def process_parquet(input_path: Path, output_path: Path, dataset_name: str, column: str, start: int) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as err:
        raise SystemExit(
            "Reading/writing parquet requires pyarrow. "
            "Activate a venv with pyarrow (e.g. project llamafactory venv) and retry."
        ) from err

    table = pq.read_table(input_path)
    n = table.num_rows
    ids = [make_id(dataset_name, start + i) for i in range(n)]
    if column in table.column_names:
        # replace existing column
        col_index = table.column_names.index(column)
        table = table.remove_column(col_index)
    table = table.append_column(column, pa.array(ids, type=pa.string()))
    tmp_fd, tmp_name = tempfile.mkstemp(prefix="qid_", suffix=".parquet", dir=str(output_path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return n


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix="qid_", suffix=path.suffix, dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def process_file(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    column: str,
    start: int = 0,
) -> int:
    fmt = detect_format(input_path)
    if fmt == "json":
        return process_json(input_path, output_path, dataset_name, column, start)
    if fmt == "jsonl":
        return process_jsonl(input_path, output_path, dataset_name, column, start)
    if fmt == "parquet":
        return process_parquet(input_path, output_path, dataset_name, column, start)
    raise ValueError(f"Unsupported format for {input_path}")


def resolve_dataset_info_path(file_name: str, dataset_dir: Path) -> Path:
    from llamafactory.data.parser import expand_dataset_path

    p = Path(expand_dataset_path(file_name))
    if p.is_absolute():
        return p
    return dataset_dir / p


def process_from_dataset_info(
    info_path: Path,
    dataset_names: list[str],
    column: str,
    in_place: bool,
    update_dataset_info: bool,
) -> None:
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    dataset_dir = info_path.parent
    results = []
    for name in dataset_names:
        if name not in info:
            raise KeyError(f'Dataset {name!r} not found in {info_path}')
        entry = info[name]
        if "file_name" not in entry:
            raise KeyError(f'Dataset {name!r} has no file_name; only file-backed datasets are supported')
        input_path = resolve_dataset_info_path(entry["file_name"], dataset_dir)
        if not input_path.exists():
            raise FileNotFoundError(f'{name}: file not found: {input_path}')

        if in_place:
            output_path = input_path
        else:
            output_path = default_output_path(input_path)

        n = process_file(input_path, output_path, name, column, start=0)
        results.append((name, input_path, output_path, n))

        columns = entry.setdefault("columns", {})
        columns[column] = column
        if update_dataset_info and not in_place:
            # store path relative to dataset_dir when possible; otherwise
            # keep a portable ${HF_HUB_CACHE}/... prefix if under the cache.
            try:
                entry["file_name"] = str(output_path.relative_to(dataset_dir))
            except ValueError:
                from llamafactory.data.parser import cache_relative_file_name

                entry["file_name"] = cache_relative_file_name(str(output_path)) or str(output_path)

    if update_dataset_info:
        _atomic_write_text(info_path, json.dumps(info, ensure_ascii=False, indent=2) + "\n")
        print(f"Updated {info_path}")

    print("\nDone. Results:")
    for name, inp, out, n in results:
        print(f"  {name}: {n} rows")
        print(f"    input:  {inp}")
        print(f"    output: {out}")
    print("\nSuggested columns entry for each dataset:")
    print(f'  "question_id": "{column}"')
    if not update_dataset_info and not in_place:
        print("\nPoint dataset_info.json file_name at the new paths, or re-run with --update-dataset-info.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name used in QUESTION_ID prefix")
    parser.add_argument("--input", type=str, default=None, help="Input annotation file (json/jsonl/parquet)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: *.with_question_id.*)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input file")
    parser.add_argument("--column", type=str, default="question_id", help="Column name to write (default: question_id)")
    parser.add_argument("--start", type=int, default=0, help="Starting index for NUMBER (default: 0)")
    parser.add_argument(
        "--from-dataset-info",
        type=str,
        default=None,
        help="Path to dataset_info.json for batch mode",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names for batch mode",
    )
    parser.add_argument(
        "--update-dataset-info",
        action="store_true",
        help="In batch mode, set columns.question_id and (if not --in-place) file_name to the new path",
    )
    args = parser.parse_args(argv)

    if args.from_dataset_info:
        if not args.datasets:
            parser.error("--datasets is required with --from-dataset-info")
        names = [n.strip() for n in args.datasets.split(",") if n.strip()]
        process_from_dataset_info(
            Path(args.from_dataset_info),
            names,
            column=args.column,
            in_place=args.in_place,
            update_dataset_info=args.update_dataset_info,
        )
        return 0

    if not args.dataset_name or not args.input:
        parser.error("Provide --dataset-name and --input, or use --from-dataset-info")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    if args.in_place:
        output_path = input_path
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = default_output_path(input_path)

    if output_path != input_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    n = process_file(input_path, output_path, args.dataset_name, args.column, start=args.start)
    print(f"Wrote {n} rows with column {args.column!r} to {output_path}")
    print("Suggested dataset_info.json columns entry:")
    print(f'  "question_id": "{args.column}"')
    print(f"Example ID: {make_id(args.dataset_name, args.start)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
