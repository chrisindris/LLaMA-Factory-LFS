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

import pytest
from datasets import Dataset

from llamafactory.data.data_utils import split_dataset
from llamafactory.hparams import DataArguments


def _row(question_id: str, prompt: str, response: str) -> dict:
    return {
        "_question_id": question_id,
        "_prompt": [{"role": "user", "content": prompt}],
        "_response": [{"role": "assistant", "content": response}],
    }


def _split(train_rows, eval_rows, **kwargs):
    data_args = DataArguments(**kwargs)
    train = Dataset.from_list(train_rows)
    if isinstance(eval_rows, dict):
        eval_dataset = {name: Dataset.from_list(rows) for name, rows in eval_rows.items()}
    else:
        eval_dataset = Dataset.from_list(eval_rows)
    return split_dataset(train, eval_dataset, data_args, seed=42)


@pytest.mark.runs_on(["cpu", "mps"])
def test_exclude_overlapping_question_ids():
    train_dict, eval_dict = _split(
        [
            _row("Scene30k_5", "q5", "a5"),
            _row("Scene30k_6", "q6", "a6"),
            _row("Scene30k_7", "q7", "a7"),
        ],
        [_row("Scene30k_5", "q5", "a5")],
        dataset="Scene30k",
        eval_dataset="Scene30k_eval16",
    )
    assert eval_dict["validation"]["_question_id"] == ["Scene30k_5"]
    assert train_dict["train"]["_question_id"] == ["Scene30k_6", "Scene30k_7"]


@pytest.mark.runs_on(["cpu", "mps"])
def test_exclude_overlapping_contents_with_different_ids():
    train_dict, eval_dict = _split(
        [
            _row("Scene30k_train_5", "same question", "same answer"),
            _row("Scene30k_6", "other question", "other answer"),
        ],
        [_row("Scene30k_5", "same question", "same answer")],
        dataset="Scene30k",
        eval_dataset="Scene30k_eval16",
    )
    assert eval_dict["validation"]["_question_id"] == ["Scene30k_5"]
    assert train_dict["train"]["_question_id"] == ["Scene30k_6"]


@pytest.mark.runs_on(["cpu", "mps"])
def test_no_overlap_keeps_train_length():
    train_dict, _eval_dict = _split(
        [_row("Scene30k_1", "q1", "a1"), _row("Scene30k_2", "q2", "a2")],
        [_row("Scene30k_9", "q9", "a9")],
        dataset="Scene30k",
        eval_dataset="Scene30k_eval16",
    )
    assert train_dict["train"]["_question_id"] == ["Scene30k_1", "Scene30k_2"]


@pytest.mark.runs_on(["cpu", "mps"])
def test_exclude_eval_from_train_false_keeps_overlap():
    train_dict, _eval_dict = _split(
        [_row("Scene30k_5", "q5", "a5"), _row("Scene30k_6", "q6", "a6")],
        [_row("Scene30k_5", "q5", "a5")],
        dataset="Scene30k",
        eval_dataset="Scene30k_eval16",
        exclude_eval_from_train=False,
    )
    assert train_dict["train"]["_question_id"] == ["Scene30k_5", "Scene30k_6"]


@pytest.mark.runs_on(["cpu", "mps"])
def test_same_dataset_names_skip_exclusion():
    train_dict, _eval_dict = _split(
        [_row("tiny_0", "q0", "a0"), _row("tiny_1", "q1", "a1")],
        [_row("tiny_0", "q0", "a0")],
        dataset="tiny",
        eval_dataset="tiny",
    )
    assert train_dict["train"]["_question_id"] == ["tiny_0", "tiny_1"]


@pytest.mark.runs_on(["cpu", "mps"])
def test_eval_on_each_dataset_dict_excludes_union():
    train_dict, eval_dict = _split(
        [
            _row("Scene30k_5", "q5", "a5"),
            _row("SpatialSSRL_coldstart_18", "q18", "a18"),
            _row("3DThinker10k_11", "q11", "a11"),
            _row("Scene30k_6", "q6", "a6"),
        ],
        {
            "Scene30k_eval16": [_row("Scene30k_5", "q5", "a5")],
            "SpatialSSRL_eval16": [_row("SpatialSSRL_coldstart_18", "q18", "a18")],
            "3DThinker10k_eval16": [_row("3DThinker10k_11", "q11", "a11")],
        },
        dataset="Scene30k,SpatialSSRL_coldstart,3DThinker10k",
        eval_dataset="Scene30k_eval16,SpatialSSRL_eval16,3DThinker10k_eval16",
    )
    assert set(eval_dict) == {
        "validation_Scene30k_eval16",
        "validation_SpatialSSRL_eval16",
        "validation_3DThinker10k_eval16",
    }
    assert train_dict["train"]["_question_id"] == ["Scene30k_6"]


def _n_rows(n: int) -> list[dict]:
    return [_row(f"id_{i}", f"q{i}", f"a{i}") for i in range(n)]


@pytest.mark.runs_on(["cpu", "mps"])
def test_val_size_equivalent_matches_val_size_minus_overlap():
    train_rows = _n_rows(100)
    eval_rows = [_row("id_0", "q0", "a0"), _row("id_1", "q1", "a1")]
    seed = 42
    val_train, val_eval = split_dataset(
        Dataset.from_list(train_rows),
        None,
        DataArguments(dataset="Scene30k", val_size=0.1),
        seed=seed,
    )
    equiv_train, equiv_eval = _split(
        train_rows,
        eval_rows,
        dataset="Scene30k",
        eval_dataset="Scene30k_eval16",
        val_size_equivalent=0.1,
    )
    eval_ids = {"id_0", "id_1"}
    expected = [qid for qid in val_train["train"]["_question_id"] if qid not in eval_ids]
    assert list(equiv_train["train"]["_question_id"]) == expected
    assert len(equiv_eval["validation"]) == 2
    assert len(val_eval["validation"]) == 10
    x = len(val_train["train"]) - len(expected)
    assert len(equiv_train["train"]) == len(val_train["train"]) - x


@pytest.mark.runs_on(["cpu", "mps"])
def test_val_size_equivalent_no_overlap_matches_val_size_train_length():
    train_rows = _n_rows(100)
    eval_rows = [_row("holdout_a", "qa", "aa"), _row("holdout_b", "qb", "ab")]
    seed = 42
    val_train, _val_eval = split_dataset(
        Dataset.from_list(train_rows),
        None,
        DataArguments(dataset="Scene30k", val_size=0.1),
        seed=seed,
    )
    equiv_train, equiv_eval = _split(
        train_rows,
        eval_rows,
        dataset="Scene30k",
        eval_dataset="Scene30k_eval16",
        val_size_equivalent=0.1,
    )
    assert len(equiv_train["train"]) == len(val_train["train"])
    assert list(equiv_train["train"]["_question_id"]) == list(val_train["train"]["_question_id"])
    assert len(equiv_eval["validation"]) == 2


@pytest.mark.runs_on(["cpu", "mps"])
def test_val_size_equivalent_without_exclude_keeps_raw_size():
    train_rows = _n_rows(100)
    eval_rows = [_row("id_0", "q0", "a0"), _row("id_1", "q1", "a1")]
    seed = 42
    val_train, _val_eval = split_dataset(
        Dataset.from_list(train_rows),
        None,
        DataArguments(dataset="Scene30k", val_size=0.1),
        seed=seed,
    )
    equiv_train, _equiv_eval = _split(
        train_rows,
        eval_rows,
        dataset="Scene30k",
        eval_dataset="Scene30k_eval16",
        val_size_equivalent=0.1,
        exclude_eval_from_train=False,
    )
    assert len(equiv_train["train"]) == len(val_train["train"])
    assert list(equiv_train["train"]["_question_id"]) == list(val_train["train"]["_question_id"])


@pytest.mark.runs_on(["cpu", "mps"])
def test_val_size_and_val_size_equivalent_are_mutually_exclusive():
    with pytest.raises(ValueError, match="both `val_size` and `val_size_equivalent`"):
        DataArguments(dataset="Scene30k", val_size=0.1, val_size_equivalent=0.1)


@pytest.mark.runs_on(["cpu", "mps"])
def test_val_size_equivalent_requires_eval_dataset():
    with pytest.raises(ValueError, match="val_size_equivalent"):
        DataArguments(dataset="Scene30k", val_size_equivalent=0.1)
