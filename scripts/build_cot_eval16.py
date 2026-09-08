#!/usr/bin/env python3
# Copyright 2026 the LlamaFactory team.
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

"""Slice a frozen 16-row CoT eval set (8 Scene30k + 4 SpatialSSRL + 4 3DThinker).

IDs are the lowest ``question_index`` values that appeared in the previous
``val_size=0.1`` holdout (see debug/logging_analysis/out/eval_compare).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_HF_CACHE_VAR_RE = re.compile(r"\$\{(HF_HUB_CACHE|HF_HOME)\}|\$(HF_HUB_CACHE|HF_HOME)(?![A-Za-z0-9_])")


COUNTS = {
    "Scene30k": 8,
    "SpatialSSRL_coldstart": 4,
    "3DThinker10k": 4,
}

DEFAULT_CSV = Path("debug/logging_analysis/out/eval_compare/detailed_predictions.csv")
DEFAULT_DATASET_INFO = Path("data/dataset_info.json")
DEFAULT_OUT_DIR = Path("data/cot_eval16")

LOCAL_FALLBACKS = {
    "Scene30k": Path("data/train-00000-of-00001.with_question_id.formatted.parquet"),
    "SpatialSSRL_coldstart": Path("data/SFT-coldstart.with_question_id.formatted.json"),
    "3DThinker10k": Path("data/3dthinker10k_cot.with_question_id.formatted.jsonl"),
}

OUTPUT_NAMES = {
    "Scene30k": "Scene30k_eval16.json",
    "SpatialSSRL_coldstart": "SpatialSSRL_eval16.json",
    "3DThinker10k": "3DThinker10k_eval16.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def expand_file_name(path: str) -> str:
    cache_root = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if not path or _HF_CACHE_VAR_RE.search(path) is None:
        return path
    if not cache_root:
        return path

    def _repl(match: re.Match[str]) -> str:
        return cache_root

    return _HF_CACHE_VAR_RE.sub(_repl, path)


def select_ids(csv_path: Path) -> dict[str, list[dict[str, Any]]]:
    import pandas as pd

    frame = pd.read_csv(csv_path, usecols=["dataset", "question_id", "question_index"])
    frame = frame.drop_duplicates(subset=["question_id"])
    selected: dict[str, list[dict[str, Any]]] = {}
    for dataset, n in COUNTS.items():
        sub = frame[frame["dataset"] == dataset].copy()
        if sub.empty:
            raise SystemExit(f"No holdout rows for dataset {dataset!r} in {csv_path}")
        sub["question_index"] = sub["question_index"].astype(int)
        sub = sub.sort_values("question_index", kind="mergesort")
        picked = sub.head(n)
        if len(picked) < n:
            raise SystemExit(f"Need {n} {dataset} holdout IDs, found {len(picked)}")
        selected[dataset] = [
            {
                "dataset": dataset,
                "question_id": str(row.question_id),
                "question_index": int(row.question_index),
            }
            for row in picked.itertuples(index=False)
        ]
    return selected


def resolve_source(dataset: str, dataset_info: dict[str, Any]) -> Path:
    entry = dataset_info.get(dataset) or {}
    file_name = entry.get("file_name")
    candidates: list[Path] = []
    if file_name:
        try:
            expanded = expand_file_name(str(file_name))
        except Exception:
            expanded = str(file_name)
        candidates.append(Path(expanded))
        if not Path(expanded).is_absolute():
            candidates.append(Path("data") / expanded)
    fallback = LOCAL_FALLBACKS.get(dataset)
    if fallback is not None:
        candidates.append(fallback)
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit(
        f"Could not find source annotations for {dataset}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        return frame.to_dict(orient="records")
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise TypeError(f"{path}: expected JSON objects")
                rows.append(obj)
        return rows
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise TypeError(f"{path}: expected a JSON list")
        return data
    raise SystemExit(f"Unsupported annotation format: {path}")


def index_by_question_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        qid = record.get("question_id")
        if qid is None:
            continue
        out[str(qid)] = record
    if not out:
        raise SystemExit("Source annotations have no question_id column")
    return out


def jsonable(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict, list)):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def write_json_list(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(jsonable(records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    if not args.holdout_csv.is_file():
        print(f"ERROR: holdout CSV not found: {args.holdout_csv}", file=sys.stderr)
        return 1
    if not args.dataset_info.is_file():
        print(f"ERROR: dataset_info not found: {args.dataset_info}", file=sys.stderr)
        return 1

    with args.dataset_info.open(encoding="utf-8") as handle:
        dataset_info = json.load(handle)

    selected = select_ids(args.holdout_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "counts": COUNTS,
        "holdout_csv": str(args.holdout_csv),
        "datasets": {},
    }
    total = 0
    for dataset, specs in selected.items():
        wanted = [item["question_id"] for item in specs]
        source = resolve_source(dataset, dataset_info)
        by_id = index_by_question_id(load_records(source))
        missing = [qid for qid in wanted if qid not in by_id]
        if missing:
            raise SystemExit(f"{dataset}: missing question_id(s) in {source}: {missing}")
        sliced = [by_id[qid] for qid in wanted]
        out_name = OUTPUT_NAMES[dataset]
        out_path = args.output_dir / out_name
        if out_path.suffix == ".jsonl":
            write_jsonl(out_path, sliced)
        else:
            write_json_list(out_path, sliced)
        manifest["datasets"][dataset] = {
            "n": len(sliced),
            "source": str(source),
            "output": str(out_path),
            "question_ids": wanted,
            "question_indexes": [item["question_index"] for item in specs],
        }
        total += len(sliced)
        print(f"{dataset}: {len(sliced)} rows from {source} -> {out_path}")
        for qid in wanted:
            print(f"  {qid}")

    if total != 16:
        raise SystemExit(f"Expected 16 eval rows, wrote {total}")

    manifest_path = args.output_dir / "question_ids.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
