# Copyright 2025 the LlamaFactory team.
"""Prepare CoT annotations once; verify and transfer the resulting offline bundle."""

import argparse
import copy
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


FORMAT_VERSION = "cot-v1"
INSTRUCTION = (
    "Your output must be formatted as '<think>thought process as you decide on an answer</think>"
    "<answer>final answer</answer>'. Start your response with '<think>', write your reasoning, "
    "close it with '</think>', then put only the final answer inside '<answer>' and '</answer>'."
)
MC_INSTRUCTION = INSTRUCTION + " Select ONE correct option, including its label and text, e.g. 'A. Above'."
REGISTRY = {
    "Scene30k": {
        "file_name": "Scene30k.parquet",
        "formatting": "alpaca",
        "columns": {
            "system": "formatting_instruction",
            "prompt": "question_with_image_tags",
            "response": "cot",
            "images": "images_every_24",
            "question_id": "question_id",
        },
    },
    "SpatialSSRL_coldstart": {
        "file_name": "SpatialSSRL.json",
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "images": "images",
            "question_id": "question_id",
        },
    },
    "3DThinker10k": {
        "file_name": "Thinker10k.jsonl",
        "formatting": "alpaca",
        "columns": {
            "system": "system",
            "prompt": "instruction",
            "response": "output",
            "images": "images",
            "question_id": "question_id",
        },
    },
}


def normalize_scene30k(record: dict) -> dict:
    result = copy.deepcopy(record)
    question = record["question_with_image_tags"]
    tags = re.findall(r"<image>", question)
    if not tags:
        raise ValueError("question has no image tags")
    result["question_with_image_tags"] = "".join(tags) + " " + question.replace("<image>", "").strip()
    result["formatting_instruction"] = INSTRUCTION
    return result


def normalize_spatialssrl(record: dict) -> dict:
    result = copy.deepcopy(record)
    question = record["instruction"]
    question = re.sub(r"You FIRST .*?\\+boxed\{\}['\"]?\.", "", question, flags=re.DOTALL)
    result["input"] = question.replace("image<image>", "image <image>").strip()
    result["instruction"] = INSTRUCTION
    match = re.fullmatch(r"(.*?)\\+boxed\{(.*)\}.*", record["output"], flags=re.DOTALL)
    if not match:
        raise ValueError("output has no boxed answer")
    reasoning, answer = match.groups()
    answer = answer.strip()
    if answer in "ABCD" and len(answer) == 1:
        choices = dict(
            re.findall(r"(?:^|\s)([A-D])\.\s*(.*?)(?=\s+[A-D]\.\s|\.(?:\s|$)|$)", result["input"], flags=re.DOTALL)
        )
        if answer not in choices:
            raise ValueError(f"boxed option {answer} has no matching choice")
        answer = f"{answer}. {choices[answer].strip()}"
    result["output"] = f"<think>{reasoning.strip()}</think><answer>{answer}</answer>"
    return result


def normalize_thinker10k(record: dict) -> dict:
    result = copy.deepcopy(record)
    system = record["system"]
    if "[Answer Instruction]" not in system or "[Question]" not in system:
        raise ValueError("system is missing answer instruction or question section")
    task = system.split("[Answer Instruction]", 1)[0]
    tags = re.findall(r"<image>", task)
    if not tags:
        raise ValueError("system has no image tags")
    result["system"] = task.replace("<image>", "").strip() + "\n[Answer Instruction]\n" + MC_INSTRUCTION
    result["instruction"] = "".join(tags) + " " + record["instruction"]
    output = record["output"].removeprefix("<output_3D>").strip()
    match = re.fullmatch(r"(<think>.*?</think>).*?(<answer>.*?</answer>)\s*", output, flags=re.DOTALL)
    if not match:
        raise ValueError("output is missing think/answer blocks")
    result["output"] = "".join(match.groups())
    return result


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_footer(path: Path) -> bytes:
    with path.open("rb") as handle:
        handle.seek(-8, 2)
        tail = handle.read(8)
        if tail[4:] != b"PAR1":
            raise ValueError(f"invalid parquet: {path}")
        size = int.from_bytes(tail[:4], "little")
        if size > path.stat().st_size - 8:
            raise ValueError(f"invalid parquet footer: {path}")
        handle.seek(-8 - size, 2)
        return handle.read(size)


def read_records(path: Path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
    elif path.suffix == ".json":
        yield from json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def verify_bundle(root: Path, full: bool = True) -> dict:
    """Full transfer check, or a stdlib-only submission check without reformatting."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    registry_path = root / "dataset_info.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("incompatible CoT bundle version; prepare a cot-v1 bundle")
    if registry != REGISTRY or set(manifest.get("datasets", {})) != set(REGISTRY):
        raise ValueError("bundle registry has incompatible datasets or field mappings")
    if checksum(registry_path) != manifest["registry_sha256"]:
        raise ValueError("registry checksum mismatch")
    for name, entry in REGISTRY.items():
        path = root / entry["file_name"]
        expected = manifest["datasets"][name]
        if path.is_symlink():
            raise ValueError(f"{name}: bundle annotation must be a regular file, not a symlink")
        if path.stat().st_size != expected["size"]:
            raise ValueError(f"{name}: size mismatch; transfer the complete bundle")
        if not isinstance(expected["rows"], int) or expected["rows"] < 1:
            raise ValueError(f"{name}: invalid row count")
        required = set(entry["columns"].values())
        if not full and path.suffix == ".parquet":
            footer = parquet_footer(path)
            if any(field.encode() not in footer for field in required):
                raise ValueError(f"{name}: missing mapped parquet columns")
        else:
            rows = 0
            for row in read_records(path):
                rows += 1
                if required - row.keys():
                    raise ValueError(f"{name}: missing mapped fields {sorted(required - row.keys())}")
            if rows != expected["rows"]:
                raise ValueError(f"{name}: row count mismatch")
        if full and checksum(path) != expected["sha256"]:
            raise ValueError(f"{name}: checksum mismatch; transfer the complete bundle")
    return manifest


def prepare_bundle(scene30k: Path, spatialssrl: Path, thinker10k: Path, output_dir: Path) -> dict:
    """Explicit preparation; leave existing bundles untouched and publish only a verified bundle."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}; use a new bundle directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    sources = dict(zip(REGISTRY, map(Path, (scene30k, spatialssrl, thinker10k))))
    normalizers = dict(zip(REGISTRY, (normalize_scene30k, normalize_spatialssrl, normalize_thinker10k)))
    manifest = {
        "schema_version": 1,
        "format_version": FORMAT_VERSION,
        "preparation_script_sha256": checksum(Path(__file__)),
        "datasets": {},
    }
    try:
        manifest["preparation_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["preparation_commit"] = None
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=".cot-prepare-") as scratch:
        root = Path(scratch) / "bundle"
        root.mkdir()
        for name, source in sources.items():
            records = []
            seen = set()
            for row in read_records(source):
                identity = row.get("question_id")
                if not identity or identity in seen:
                    raise ValueError(f"{name}: missing or duplicate question_id {identity!r}")
                seen.add(identity)
                try:
                    records.append(normalizers[name](row))
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{name} {identity}: {exc}") from exc
            if not records:
                raise ValueError(f"{name}: no records")
            path = root / REGISTRY[name]["file_name"]
            if path.suffix == ".parquet":
                pq.write_table(pa.Table.from_pylist(records), path)
            elif path.suffix == ".json":
                path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            else:
                with path.open("w", encoding="utf-8") as handle:
                    for row in records:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest["datasets"][name] = {
                "rows": len(records),
                "size": path.stat().st_size,
                "sha256": checksum(path),
                "source_name": source.name,
                "source_sha256": checksum(source),
            }
        registry_path = root / "dataset_info.json"
        registry_path.write_text(json.dumps(REGISTRY, indent=2) + "\n", encoding="utf-8")
        manifest["registry_sha256"] = checksum(registry_path)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        verify_bundle(root)
        root.rename(output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene30k", type=Path)
    parser.add_argument("--spatialssrl", type=Path)
    parser.add_argument("--thinker10k", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path, help="verify a copied bundle; no formatting")
    parser.add_argument("--preflight", type=Path, help="lightweight stdlib-only validation")
    args = parser.parse_args()
    try:
        if args.verify or args.preflight:
            root = args.verify or args.preflight
            manifest = verify_bundle(root, full=bool(args.verify))
        else:
            if not all((args.scene30k, args.spatialssrl, args.thinker10k, args.output_dir)):
                parser.error("preparation requires --scene30k, --spatialssrl, --thinker10k, --output-dir")
            root = args.output_dir
            manifest = prepare_bundle(args.scene30k, args.spatialssrl, args.thinker10k, root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"CoT bundle: {exc}\n")
    print(f"CoT bundle OK: {root} ({manifest['format_version']})")
    for name, details in manifest["datasets"].items():
        print(f"  {name}: {details['rows']} records")


if __name__ == "__main__":
    main()
