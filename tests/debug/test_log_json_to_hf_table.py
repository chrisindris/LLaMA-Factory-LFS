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

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PKG = REPO_ROOT / "debug" / "logging_analysis"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from log_json_to_hf_table import (  # noqa: E402
    PredictionDumpError,
    extract_dataset_and_qid,
    frames_from_dir,
    rows_from_prediction_file,
)


def test_extract_dataset_and_qid():
    assert extract_dataset_and_qid("Scene30k_5") == ("Scene30k", "5")
    assert extract_dataset_and_qid("SpatialSSRL_coldstart_18") == ("SpatialSSRL_coldstart", "18")
    assert extract_dataset_and_qid("3DThinker10k_5696") == ("3DThinker10k", "5696")


def test_flattens_eval_and_train_with_integer_step(tmp_path: Path):
    eval_ep0 = {
        "Scene30k_5": {"0": "a", "124": "b"},
        "SpatialSSRL_coldstart_18": {"0": "c"},
    }
    eval_ep1 = {"Scene30k_5": {"248": "d"}}
    train = {"3DThinker10k_5696": {"124": "t"}}
    (tmp_path / "eval_predictions_ep0.json").write_text(json.dumps(eval_ep0), encoding="utf-8")
    (tmp_path / "eval_predictions_ep1.json").write_text(json.dumps(eval_ep1), encoding="utf-8")
    (tmp_path / "train_predictions_ep1.json").write_text(json.dumps(train), encoding="utf-8")
    (tmp_path / "eval_predictions_ep1.json.bak").write_text("{}", encoding="utf-8")
    nested = tmp_path / "checkpoint-309"
    nested.mkdir()
    (nested / "eval_predictions_ep9.json").write_text(
        json.dumps({"Scene30k_9": {"9": "skip"}}),
        encoding="utf-8",
    )

    frames = frames_from_dir(tmp_path)

    assert set(frames) == {"eval_predictions", "train_predictions"}
    eval_frame = frames["eval_predictions"]
    assert list(eval_frame.columns) == ["ID", "dataset", "qid", "prediction", "step"]
    assert len(eval_frame) == 4
    assert str(eval_frame["step"].dtype) == "int64"
    assert sorted(int(step) for step in eval_frame["step"].tolist()) == [0, 0, 124, 248]
    scene = eval_frame.loc[eval_frame["ID"] == "Scene30k_5"].sort_values("step")
    assert scene["prediction"].tolist() == ["a", "b", "d"]
    assert scene["dataset"].iloc[0] == "Scene30k"
    assert scene["qid"].iloc[0] == "5"
    spatial = eval_frame.loc[eval_frame["ID"] == "SpatialSSRL_coldstart_18"].iloc[0]
    assert spatial["dataset"] == "SpatialSSRL_coldstart"
    assert spatial["qid"] == "18"

    train_frame = frames["train_predictions"]
    assert train_frame["step"].tolist() == [124]
    assert str(train_frame["step"].dtype) == "int64"
    assert train_frame["dataset"].iloc[0] == "3DThinker10k"
    assert train_frame["qid"].iloc[0] == "5696"
    assert train_frame["prediction"].iloc[0] == "t"


def test_non_integer_step_raises(tmp_path: Path):
    path = tmp_path / "eval_predictions.json"
    path.write_text(json.dumps({"Scene30k_5": {"step0": "a"}}), encoding="utf-8")
    with pytest.raises(PredictionDumpError, match="not an integer"):
        rows_from_prediction_file(path)


def test_bare_string_raises(tmp_path: Path):
    path = tmp_path / "train_predictions_ep1.json"
    path.write_text(json.dumps({"Scene30k_5": "hello"}), encoding="utf-8")
    with pytest.raises(PredictionDumpError, match="bare string"):
        rows_from_prediction_file(path)


def test_other_prediction_filename_uses_stem(tmp_path: Path):
    path = tmp_path / "generated_predictions.json"
    path.write_text(json.dumps({"Scene30k_1": {"3": "x"}}), encoding="utf-8")
    frames = frames_from_dir(tmp_path)
    assert set(frames) == {"generated_predictions"}
    assert int(frames["generated_predictions"]["step"].iloc[0]) == 3


def test_missing_dir_raises(tmp_path: Path):
    with pytest.raises(PredictionDumpError, match="Not a directory"):
        frames_from_dir(tmp_path / "missing")
