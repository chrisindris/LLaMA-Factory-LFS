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

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PKG = REPO_ROOT / "debug" / "logging_analysis"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from _selftest import run_self_tests  # noqa: E402
from analyze import main  # noqa: E402


def test_self_tests() -> None:
    run_self_tests()


def test_cli_self_test() -> None:
    assert main(["--self-test"]) == 0


def test_end_to_end_tiny_log(tmp_path: Path) -> None:
    dataset_info = {
        "Scene30k": {
            "file_name": "scene.jsonl",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "response": "output"},
        },
        "SpatialSSRL_coldstart": {
            "file_name": "ssrl.json",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "response": "output"},
        },
        "3DThinker10k": {
            "file_name": "thinker.jsonl",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "response": "output"},
        },
    }
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset_info.json").write_text(json.dumps(dataset_info), encoding="utf-8")
    (data_dir / "scene.jsonl").write_text(
        json.dumps(
            {
                "question_id": "Scene30k_0",
                "instruction": "What is left of the table?",
                "output": "<think>look</think><answer>chair</answer>",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "ssrl.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "SpatialSSRL_coldstart_0",
                    "instruction": "Where is region 1?",
                    "output": "Analysis:\nleft.\n\\boxed{B}",
                }
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "thinker.jsonl").write_text(
        json.dumps(
            {
                "question_id": "3DThinker10k_0",
                "idx": 5,
                "instruction": "Which way?",
                "output": "<output_3D>\n<think>geom</think>\n<answer>D. right</answer>",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    log = {
        "/tmp/cot_stage/annotations/Scene30k.parquet_0": {
            "10": "<think>look</think><answer>chair</answer>",
            "20": "chair chair chair chair chair chair",
        },
        "/tmp/cot_stage/annotations/SpatialSSRL_coldstart.json_0": {
            "10": "The camera sees left. B",
        },
        "/tmp/cot_stage/annotations/3dthinker10k_cot.jsonl_0": {
            "10": "<think>geom</think><answer>D. right</answer>",
        },
        "not_a_known_dataset.json_9": {"10": "hello"},
    }
    log_path = tmp_path / "train_predictions_ep1.json"
    log_path.write_text(json.dumps(log), encoding="utf-8")
    out = tmp_path / "out"
    rc = main(
        [
            "--log",
            str(log_path),
            "--dataset-info",
            str(data_dir / "dataset_info.json"),
            "--dataset-dir",
            str(data_dir),
            "--output-dir",
            str(out),
            "--no-plots",
        ]
    )
    assert rc == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Canonical-format" in report or "canonical" in report.lower()
    summary = json.loads((out / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summary["integrity"]["n_prediction_observations"] == 5
    detailed = (out / "detailed_predictions.csv").read_text(encoding="utf-8")
    assert "Scene30k" in detailed
    assert "3DThinker10k" in detailed
    assert "UNKNOWN" in detailed
    overlap = (out / "question_overlap.csv").read_text(encoding="utf-8")
    assert "jaccard" in overlap
    assert (out / "health_by_step.csv").exists()
    assert (out / "warnings.csv").exists()
