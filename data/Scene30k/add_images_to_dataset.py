#!/usr/bin/env python3
"""Add images_every_24 column to Scene30k by looking up scene_id in SQA3Devery24."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset

SQAD_DIR = Path(
    "/scratch/indrisch/huggingface/hub/"
    "datasets--cvis-tmu--llamafactory-sqa3d-traces-multiimage-vqa/"
    "snapshots/ce5c54adc1608d1726730c9ff334e65b6dd70e46/data"
)
SCENE30K_JSONL = Path(
    "/scratch/indrisch/huggingface/hub/"
    "datasets--AIGeeksGroup--Scene-30K/"
    "snapshots/4dec3fdb6ff83357fbf6d93ec6fe0392df6a7152/Scene-30K.jsonl"
)
SCANNET_SCANS = Path("/scratch/indrisch/data/ScanNet/scans")
SCRIPT_DIR = Path(__file__).resolve().parent
MISSING_IDS_FILE = SCRIPT_DIR / "missing_scene_ids.txt"
HF_REPO = "cvis-tmu/Scene30k"
OUTPUT_DIR = SCRIPT_DIR / "Scene30k_with_images"
NUM_WORKERS = 64


def build_lookup() -> dict[str, list[str]]:
    """Build video -> images_every_24 lookup from SQA3Devery24 parquet files."""
    parquet_files = sorted(SQAD_DIR.glob("*.parquet"))
    print(f"Reading {len(parquet_files)} parquet file(s) from {SQAD_DIR}")

    lookup: dict[str, list[str]] = {}
    for pf in parquet_files:
        table = pq.read_table(pf, columns=["video", "images_every_24"])
        videos = table.column("video").to_pylist()
        images = table.column("images_every_24").to_pylist()
        for vid, img_list in zip(videos, images):
            if vid not in lookup:
                lookup[vid] = list(img_list) if img_list is not None else []

    print(f"Lookup built: {len(lookup)} unique video entries")
    return lookup


def _parse_line(line: str) -> dict:
    return json.loads(line)


def load_scene30k(path: Path) -> list[dict]:
    """Load Scene-30K.jsonl using multiprocessing for JSON parsing."""
    print(f"Reading {path}")
    with open(path, "r") as f:
        lines = f.readlines()

    print(f"Parsing {len(lines)} lines with {NUM_WORKERS} workers")
    rows: list[dict] = [None] * len(lines)  # type: ignore[list-item]
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_parse_line, line): i for i, line in enumerate(lines)}
        for future in as_completed(futures):
            rows[futures[future]] = future.result()

    return rows


def enrich_rows(rows: list[dict], lookup: dict[str, list[str]]) -> list[str]:
    """Add images_every_24 to each row. Returns list of missing scene_ids."""
    missing = set()
    for row in rows:
        scene_id = row["scene_id"]
        if scene_id in lookup:
            row["images_every_24"] = lookup[scene_id]
        else:
            row["images_every_24"] = []
            missing.add(scene_id)
    return sorted(missing)


def add_column_with_image_counts(rows: list[dict]) -> None:
    """Add a column that contains the number of images for that row."""
    for row in rows:
        row["num_images_every_24"] = len(row["images_every_24"])
        
        
def add_question_with_image_tags(rows: list[dict]) -> None:
    """Add a column that is equal to 'question' followed by a space, followed by '<image>' repeated for the number of images."""
    for row in rows:
        row["question_with_image_tags"] = row["question"] + " " + "<image>" * row["num_images_every_24"]


def write_missing_ids(missing: list[str]) -> None:
    print(f"Writing {len(missing)} missing scene_id(s) to {MISSING_IDS_FILE}")
    with open(MISSING_IDS_FILE, "w") as f:
        for sid in missing:
            f.write(sid + "\n")


def verify_missing_scenes() -> list[str]:
    """Check that every scene in missing_scene_ids.txt exists in ScanNet with images.

    Returns the list of verified scene_ids. Raises SystemExit if any are absent
    or have an empty color/ directory.
    """
    if not MISSING_IDS_FILE.exists():
        raise SystemExit(f"ERROR: {MISSING_IDS_FILE} not found")

    scene_ids = [
        line.strip() for line in MISSING_IDS_FILE.read_text().splitlines() if line.strip()
    ]
    print(f"Verifying {len(scene_ids)} missing scene(s) against {SCANNET_SCANS}")

    errors: list[str] = []
    for sid in scene_ids:
        color_dir = SCANNET_SCANS / sid / "color"
        if not color_dir.is_dir():
            errors.append(f"  {sid}: directory not found ({color_dir})")
        elif not any(color_dir.iterdir()):
            errors.append(f"  {sid}: color/ directory is empty ({color_dir})")

    if errors:
        raise SystemExit(
            "ERROR: Some missing scenes are not available in ScanNet:\n"
            + "\n".join(errors)
        )

    print(f"All {len(scene_ids)} missing scene(s) verified: present with populated color/")
    return scene_ids


def _build_image_list_for_scene(scene_id: str) -> list[str]:
    """Build a sorted images_every_24 list for a scene from its ScanNet color/ dir."""
    color_dir = SCANNET_SCANS / scene_id / "color"
    frame_numbers = []
    for f in color_dir.iterdir():
        if f.suffix == ".jpg":
            try:
                n = int(f.stem)
            except ValueError:
                continue
            if n % 24 == 0:
                frame_numbers.append(n)
    frame_numbers.sort()
    return [f"ScanNet/scans/{scene_id}/color/{n}.jpg" for n in frame_numbers]


def fill_empty_image_lists(rows: list[dict]) -> int:
    """Populate empty images_every_24 lists from ScanNet on disk.

    Builds a lookup for only the scene_ids that need it, then applies to all
    matching rows. Returns the number of rows that were filled.
    """
    scenes_needing_fill: set[str] = set()
    for row in rows:
        if not row["images_every_24"]:
            scenes_needing_fill.add(row["scene_id"])

    if not scenes_needing_fill:
        print("No empty image lists to fill")
        return 0

    print(f"Building image lists from ScanNet for {len(scenes_needing_fill)} scene(s)")
    scannet_lookup: dict[str, list[str]] = {}
    for sid in sorted(scenes_needing_fill):
        scannet_lookup[sid] = _build_image_list_for_scene(sid)

    filled = 0
    for row in rows:
        if not row["images_every_24"] and row["scene_id"] in scannet_lookup:
            row["images_every_24"] = scannet_lookup[row["scene_id"]]
            filled += 1

    print(f"Filled {filled} row(s) across {len(scenes_needing_fill)} scene(s)")
    return filled


def push_to_hub(ds: Dataset) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is not set")
    print(f"Pushing to {HF_REPO}")
    ds.push_to_hub(HF_REPO, token=token)
    print("Upload complete")


def save_and_upload(rows: list[dict], online: bool) -> None:
    print(f"Creating Dataset from {len(rows)} rows")
    ds = Dataset.from_list(rows)
    print(ds)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset locally to {OUTPUT_DIR}")
    ds.save_to_disk(str(OUTPUT_DIR))
    print(f"Local save complete: {OUTPUT_DIR}")

    if online:
        push_to_hub(ds)
    else:
        print("ONLINE=false, skipping HuggingFace upload")


def upload_only() -> None:
    if not OUTPUT_DIR.exists():
        raise SystemExit(
            f"ERROR: Local dataset not found at {OUTPUT_DIR}\n"
            "Run the full pipeline first (without --upload-only) to create it."
        )
    print(f"Loading existing dataset from {OUTPUT_DIR}")
    ds = Dataset.load_from_disk(str(OUTPUT_DIR))
    print(ds)
    push_to_hub(ds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--online", action="store_true", default=False,
        help="Also push the dataset to HuggingFace (default: local only)",
    )
    group.add_argument(
        "--upload-only", action="store_true", default=False,
        help="Skip dataset creation; upload the existing local dataset to HuggingFace",
    )
    args = parser.parse_args()

    if args.upload_only:
        upload_only()
        return

    lookup = build_lookup()
    rows = load_scene30k(SCENE30K_JSONL)
    missing = enrich_rows(rows, lookup)
    write_missing_ids(missing)

    verify_missing_scenes()
    fill_empty_image_lists(rows)
    
    add_column_with_image_counts(rows)
    add_question_with_image_tags(rows)

    matched = len(rows) - sum(1 for r in rows if not r["images_every_24"])
    print(f"Rows with images: {matched}, rows without: {len(rows) - matched}")

    save_and_upload(rows, online=args.online)


if __name__ == "__main__":
    main()
