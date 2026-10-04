# Copyright 2025 the LlamaFactory team.

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_cot_annotations.py"


def load_module():
    assert SCRIPT.is_file(), "prepared annotation implementation is missing"
    spec = importlib.util.spec_from_file_location("prepare_cot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_records():
    return {
        "Scene30k": {
            "question_id": "s0",
            "question_with_image_tags": "Where? <image><image>",
            "cot": "<think>Look.</think><answer>Left.</answer>",
            "images_every_24": ["a.jpg", "b.jpg"],
        },
        "SpatialSSRL_coldstart": {
            "question_id": "p0",
            "instruction": "An image<image>. A. Left B. Right C. Above D. Below. You FIRST provide a DETAILED analysis and put your final answer in '\\boxed{}'.",
            "input": "",
            "output": "Look left. \\boxed{A}",
            "images": ["a.jpg"],
        },
        "3DThinker10k": {
            "question_id": "t0",
            "system": "<image>\n[Task]\nLook. Keep punctuation.\n[Answer Instruction]\nOnly answer.\n[Question]\nWhere? A. Left B. Right",
            "instruction": "Where? A. Left B. Right",
            "output": "<output_3D>\n<think>Look.</think>\nextra\n<answer>A. Left</answer>",
            "images": ["a.jpg"],
        },
    }


def test_normalizers_preserve_content_and_map_questions():
    m = load_module()
    records = original_records()
    untouched = json.loads(json.dumps(records))
    s = m.normalize_scene30k(records["Scene30k"])
    assert s["question_with_image_tags"] == "<image><image> Where?"
    assert s["cot"] == records["Scene30k"]["cot"]
    assert "<think>" in s["formatting_instruction"]
    p = m.normalize_spatialssrl(records["SpatialSSRL_coldstart"])
    assert "A. Left" in p["input"] and "You FIRST" not in p["input"]
    assert p["output"] == "<think>Look left.</think><answer>A. Left</answer>"
    t = m.normalize_thinker10k(records["3DThinker10k"])
    assert "Keep punctuation." in t["system"]
    assert "A. Above" in t["system"]
    assert "[Question]" not in t["system"] and "<image>" not in t["system"]
    assert t["instruction"].startswith("<image> Where?")
    assert t["output"] == "<think>Look.</think><answer>A. Left</answer>"
    assert records == untouched
    for original, normalized in [
        (records["Scene30k"], s),
        (records["SpatialSSRL_coldstart"], p),
        (records["3DThinker10k"], t),
    ]:
        assert original["question_id"] == normalized["question_id"]
        key = "images_every_24" if "images_every_24" in original else "images"
        assert original[key] == normalized[key]


def test_malformed_answer_reports_error():
    m = load_module()
    p = original_records()["SpatialSSRL_coldstart"]
    p["output"] = "No boxed answer."
    with pytest.raises(ValueError, match="boxed"):
        m.normalize_spatialssrl(p)


def make_bundle(tmp_path):
    m = load_module()
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    r = original_records()
    scene = tmp_path / "scene.parquet"
    spatial = tmp_path / "spatial.json"
    thinker = tmp_path / "thinker.jsonl"
    pq.write_table(pa.Table.from_pylist([r["Scene30k"]]), scene)
    spatial.write_text(json.dumps([r["SpatialSSRL_coldstart"]]))
    thinker.write_text(json.dumps(r["3DThinker10k"]) + "\n")
    dest = tmp_path / "bundle"
    m.prepare_bundle(scene, spatial, thinker, dest)
    return m, dest


def test_bundle_relocates_and_detects_corruption(tmp_path):
    m, dest = make_bundle(tmp_path)
    moved = tmp_path / "different-root" / "bundle"
    shutil.copytree(dest, moved)
    m.verify_bundle(moved)
    info = json.loads((moved / "dataset_info.json").read_text())
    assert info["SpatialSSRL_coldstart"]["columns"]["query"] == "input"
    assert info["SpatialSSRL_coldstart"]["columns"]["prompt"] == "instruction"
    assert info["Scene30k"]["columns"]["system"] == "formatting_instruction"
    assert all(not Path(v["file_name"]).is_absolute() for v in info.values())
    manifest = json.loads((moved / "manifest.json").read_text())
    assert all(e["rows"] == 1 for e in manifest["datasets"].values())
    with (moved / "Thinker10k.jsonl").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="checksum|size"):
        m.verify_bundle(moved)


@pytest.mark.parametrize("damage", ["version", "mapping", "missing", "count", "schema"])
def test_preflight_rejects_incompatible_bundle(tmp_path, damage):
    m, dest = make_bundle(tmp_path)
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "version":
        manifest["format_version"] = "unknown"
    elif damage == "count":
        manifest["datasets"]["3DThinker10k"]["rows"] = 2
    elif damage == "missing":
        (dest / "SpatialSSRL.json").unlink()
    elif damage == "mapping":
        p = dest / "dataset_info.json"
        info = json.loads(p.read_text())
        info["SpatialSSRL_coldstart"]["columns"].pop("query")
        p.write_text(json.dumps(info))
    else:
        p = dest / "SpatialSSRL.json"
        records = json.loads(p.read_text())
        del records[0]["input"]
        p.write_text(json.dumps(records))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        m.verify_bundle(dest, full=False)


def test_portable_body_uses_prepared_bundle_after_relocation(tmp_path):
    _, bundle = make_bundle(tmp_path)
    repo = SCRIPT.parents[1]
    moved = tmp_path / "renamed-checkout"
    files = [
        "scripts/utils/portable_env.sh",
        "scripts/sysconfigtool.py",
        "scripts/sysconfig.json",
        "scripts/prepare_cot_annotations.py",
        "examples/deepspeed/ds_z2_config.json",
        "examples/train_lora/portable_qwen2_5vl_lora_sft_CoT_traineval.yaml",
        "models/qwen2_5vl_lora_sft_CoT/portable_body_qwen2_5vl_lora_sft_CoT_traineval.sh",
    ]
    for name in files:
        target = moved / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / name, target)
    (moved / "src/llamafactory").mkdir(parents=True)
    (moved / "pyproject.toml").touch()
    dest = moved / "data/annotations/cot-v1"
    shutil.copytree(bundle, dest)
    for name in [
        ".cache/huggingface/models--Qwen--Qwen2.5-VL-7B-Instruct",
        "data/h5/ScanNet_h5/scans",
        "data/h5/Spatial-SSRL_images_h5",
        "data/h5/3DThinker10K_images_h5",
        ".venv/bin",
    ]:
        (moved / name).mkdir(parents=True, exist_ok=True)
    binary = moved / ".venv/bin/llamafactory-cli"
    binary.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$TRAIN_ARGS"\n')
    binary.chmod(0o755)
    (moved / ".venv/bin/activate").write_text(f'export PATH="{binary.parent}:$PATH"\n')
    body = moved / files[-1]
    env = dict(
        os.environ,
        LFS_PROJECT_DIR=str(moved),
        CLUSTER="PORTABLE",
        RUNNING_MODE="VENV",
        PORTABLE_SKIP_SITE_ENV="1",
        TRAIN_ARGS=str(tmp_path / "args"),
    )
    for key in [
        "PROJECT_DIR",
        "PORTABLE_COT_BUNDLE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "MEDIA_DIR",
        "SCANNET_H5_DIR",
        "SPATIALSSRL_H5_DIR",
        "THINKER10K_H5_DIR",
        "VENV_LLAMAFACTORY",
    ]:
        env.pop(key, None)
    result = subprocess.run(["bash", str(body), "max_steps=1"], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    args = (tmp_path / "args").read_text()
    assert f"dataset_dir={dest}" in args
    assert str(repo) not in args
    registry = (dest / "dataset_info.json").read_bytes()
    result = subprocess.run(["bash", str(body)], env=dict(env, PORTABLE_STAGE="1"), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (dest / "dataset_info.json").read_bytes() == registry
    (dest / "manifest.json").unlink()
    result = subprocess.run(["bash", str(body)], env=dict(env, PREFLIGHT="1"), capture_output=True, text=True)
    assert result.returncode != 0
    assert "manifest" in result.stdout + result.stderr
