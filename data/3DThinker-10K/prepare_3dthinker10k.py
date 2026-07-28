#!/usr/bin/env python3
"""Prepare 3DThinker-10k CoT annotations for LLaMA-Factory (alpaca + H5 image paths).

Source annotations are multi-line pretty-printed JSON objects (not strict JSONL).
Field mapping (CoT SFT):
  system      <- mindcube_input  (contains <image> placeholders)
  instruction <- text_input
  output      <- text_output
  images      <- image_input
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_ANNOTATIONS = (
    "/scratch/indrisch/huggingface/hub/"
    "datasets--jankin123--3DThinker-10K/"
    "snapshots/2b16e1e73cf985e5d46b84cc90c13956bc7205f2/"
    "data_output3d_begin_10k_resized.jsonl"
)
DEFAULT_H5_DIR = os.environ.get("THINKER10K_H5_DIR", "/scratch/indrisch/3DThinker10K_images_h5")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "out" / "3dthinker10k_cot.jsonl"


def load_annotations(path: str | Path) -> list[dict]:
    """Parse multi-line concatenated JSON objects via JSONDecoder.raw_decode."""
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    examples: list[dict] = []
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        examples.append(obj)
        idx = end
    return examples


def to_index_key(annotation_path: str) -> str:
    p = annotation_path.strip().replace("\\", "/").lstrip("./")
    if p.startswith("data/"):
        p = p[len("data/") :]
    marker = "other_all_image_resize/"
    pos = p.find(marker)
    if pos > 0:
        p = p[pos:]
    return p


def load_index(h5_dir: Path) -> dict[str, int]:
    index_path = h5_dir / "3dthinker10k_images_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing H5 index: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items() if not str(k).startswith("_")}


def convert_example(ex: dict) -> dict:
    return {
        "system": ex["mindcube_input"],
        "instruction": ex["text_input"],
        "output": ex["text_output"],
        "images": list(ex["image_input"]),
        "idx": ex.get("idx"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=str, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--h5-dir", type=str, default=DEFAULT_H5_DIR)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-index-check", action="store_true")
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    if not ann_path.is_file():
        print(f"ERROR: annotations not found: {ann_path}", file=sys.stderr)
        return 1

    print(f"Loading annotations from {ann_path}")
    examples = load_annotations(ann_path)
    print(f"Parsed {len(examples)} examples")
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
        print(f"Truncated to {len(examples)} examples")

    index: dict[str, int] | None = None
    if not args.skip_index_check:
        index = load_index(Path(args.h5_dir))
        print(f"Loaded H5 index with {len(index)} keys from {args.h5_dir}")

    out_rows: list[dict] = []
    n_images = 0
    missing_keys: list[str] = []
    bad_placeholder: list[int] = []

    for i, ex in enumerate(examples):
        n_ph = ex["mindcube_input"].count("<image>")
        imgs = ex["image_input"]
        if n_ph != len(imgs):
            bad_placeholder.append(i)
        if index is not None:
            for p in imgs:
                key = to_index_key(p)
                if key not in index:
                    missing_keys.append(f"{i}:{p}->{key}")
        row = convert_example(ex)
        out_rows.append(row)
        n_images += len(imgs)

    if bad_placeholder:
        print(
            f"ERROR: {len(bad_placeholder)} examples have <image> count != len(image_input); "
            f"first ids: {bad_placeholder[:5]}",
            file=sys.stderr,
        )
        return 1
    if missing_keys:
        print(
            f"ERROR: {len(missing_keys)} image paths missing from H5 index; "
            f"examples: {missing_keys[:5]}",
            file=sys.stderr,
        )
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} rows ({n_images} image refs) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
