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

import json
from pathlib import Path

from llamafactory.hparams.finetuning_args import FinetuningArguments
from llamafactory.train.prediction_dump import (
    IGNORE_INDEX,
    PredictionDumpStore,
    decode_teacher_forced_batch,
    flatten_gathered_pairs,
    normalize_question_ids,
    should_record_train_prediction,
)


def test_train_records_keyed_by_question_id_and_step(tmp_path: Path):
    store = PredictionDumpStore(train_path=str(tmp_path / "train_predictions.json"))
    added = store.add_train_records(10, [("q1", "hello"), ("q2", "world"), ("", "skip")])

    assert added == 2
    assert store.train_data["q1"]["10"] == "hello"
    assert store.train_data["q2"]["10"] == "world"
    assert "" not in store.train_data
    assert store.train_record_count == 2

    store.flush_train()
    dumped = json.loads((tmp_path / "train_predictions.json").read_text(encoding="utf-8"))
    assert dumped == {"q1": {"10": "hello"}, "q2": {"10": "world"}}


def test_train_overwrite_same_qid_step_does_not_inflate_cap():
    store = PredictionDumpStore(max_train_samples=2)
    assert store.add_train_records(1, [("q1", "first")]) == 1
    assert store.add_train_records(1, [("q1", "second")]) == 0
    assert store.train_data["q1"]["1"] == "second"
    assert store.train_record_count == 1

    assert store.add_train_records(2, [("q2", "other")]) == 1
    assert store.train_full()
    assert store.add_train_records(3, [("q3", "rejected")]) == 0
    assert "q3" not in store.train_data


def test_eval_records_keyed_by_question_id(tmp_path: Path):
    store = PredictionDumpStore(eval_path=str(tmp_path / "eval_predictions.json"))
    assert store.add_eval_records([("q1", "a"), ("q1", "b"), ("", "skip")]) == 2
    assert store.eval_data == {"q1": "b"}

    store.flush_eval()
    dumped = json.loads((tmp_path / "eval_predictions.json").read_text(encoding="utf-8"))
    assert dumped == {"q1": "b"}


def test_normalize_question_ids_flattens_packed_and_pads():
    assert normalize_question_ids(None, 2) == ["", ""]
    assert normalize_question_ids(["a", ["b", "c"]], 3) == ["a", "b", ""]
    assert normalize_question_ids("only", 2) == ["only", "only"]


def test_should_record_train_prediction_once_per_step_and_synced_cap():
    kwargs = dict(interval=1, last_dumped_step=-1)
    assert should_record_train_prediction(dump_full=False, global_step=1, **kwargs)
    # second microbatch of the same optimizer step
    assert not should_record_train_prediction(dump_full=False, global_step=1, interval=1, last_dumped_step=1)
    # cap is a synced flag; local store.train_full() must not be used instead
    assert not should_record_train_prediction(dump_full=True, global_step=2, interval=1, last_dumped_step=1)
    assert should_record_train_prediction(dump_full=False, global_step=2, interval=1, last_dumped_step=1)
    assert not should_record_train_prediction(dump_full=False, global_step=0, interval=1, last_dumped_step=-1)
    assert not should_record_train_prediction(dump_full=False, global_step=3, interval=2, last_dumped_step=-1)
    assert should_record_train_prediction(dump_full=False, global_step=4, interval=2, last_dumped_step=2)


class _IdTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return ",".join(str(int(i)) for i in ids)


def test_decode_teacher_forced_batch_greedy_on_response_positions():
    import torch

    # logits[:, t] predicts labels[:, t+1]. Prompt positions stay IGNORE_INDEX.
    logits = torch.zeros(1, 5, 4)
    logits[0, 1, 1] = 10.0
    logits[0, 2, 2] = 10.0
    logits[0, 3, 3] = 10.0
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 1, 2, 3]])
    assert decode_teacher_forced_batch(logits, labels, _IdTokenizer()) == ["1,2,3"]


def test_flatten_gathered_pairs_keeps_empty_rank_chunks():
    assert flatten_gathered_pairs(None) == []
    assert flatten_gathered_pairs([[], [], []]) == []
    assert flatten_gathered_pairs([[], [("q1", "a")], [], [("q2", "b")]]) == [("q1", "a"), ("q2", "b")]
    assert flatten_gathered_pairs([("q1", "a"), ("q2", "b")]) == [("q1", "a"), ("q2", "b")]


def test_finetuning_args_accept_logging_and_resume_fields():
    args = FinetuningArguments(
        save_train_predictions=True,
        train_prediction_interval=4,
        save_eval_predictions=True,
        allow_warm_start_resume=False,
        require_resume_bundle=True,
        resume_bundle_dir="/tmp/resume_bundle",
        stop_at_global_step=1240,
    )
    assert args.save_train_predictions
    assert args.train_prediction_interval == 4
    assert args.save_eval_predictions
    assert args.allow_warm_start_resume is False
    assert args.require_resume_bundle
    assert args.resume_bundle_dir == "/tmp/resume_bundle"
    assert args.stop_at_global_step == 1240
