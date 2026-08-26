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
    format_epoch_name,
    normalize_question_ids,
    resolve_prediction_dump_path,
    should_record_train_prediction,
)


def test_format_epoch_name():
    assert format_epoch_name(1.0) == "1"
    assert format_epoch_name(2.0) == "2"
    assert format_epoch_name(1) == "1"
    assert format_epoch_name("2.0") == "2"
    assert format_epoch_name("3") == "3"
    assert format_epoch_name(0.0) == "0"
    assert format_epoch_name(None, is_training=True) == "1"
    assert format_epoch_name(None, is_training=False) == "0"
    # During training steps:
    assert format_epoch_name(0.2, is_training=True) == "1"
    assert format_epoch_name(1.0, is_training=True) == "1"
    assert format_epoch_name(1.2, is_training=True) == "2"
    assert format_epoch_name(2.0, is_training=True) == "2"


def test_resolve_prediction_dump_path():
    assert resolve_prediction_dump_path(None, "1", "train_predictions", "/tmp/out") == "/tmp/out/train_predictions_ep1.json"
    assert resolve_prediction_dump_path(None, "2", "eval_predictions", "/tmp/out") == "/tmp/out/eval_predictions_ep2.json"
    assert resolve_prediction_dump_path("/custom/train_{epoch}.json", "1", "train_predictions") == "/custom/train_1.json"
    assert resolve_prediction_dump_path("/custom/train_predictions.json", "2", "train_predictions") == "/custom/train_predictions_ep2.json"
    assert resolve_prediction_dump_path("/custom/train_predictions_ep1.json", "1", "train_predictions") == "/custom/train_predictions_ep1.json"


def test_train_records_per_epoch_and_flush(tmp_path: Path):
    store = PredictionDumpStore(
        train_path_template=str(tmp_path / "train_predictions_ep{epoch}.json"),
        output_dir=str(tmp_path),
    )
    # Epoch 1
    added_ep1 = store.add_train_records(1, [("q1", "ep1_s1"), ("q2", "ep1_s1")], epoch=1.0)
    assert added_ep1 == 2
    assert store.get_train_record_count("1") == 2

    # Epoch 2
    added_ep2 = store.add_train_records(6, [("q1", "ep2_s6"), ("q3", "ep2_s6")], epoch=2.0)
    assert added_ep2 == 2
    assert store.get_train_record_count("2") == 2

    store.flush_train(epoch="1")
    store.flush_train(epoch="2")

    dumped_ep1 = json.loads((tmp_path / "train_predictions_ep1.json").read_text(encoding="utf-8"))
    dumped_ep2 = json.loads((tmp_path / "train_predictions_ep2.json").read_text(encoding="utf-8"))

    assert dumped_ep1 == {"q1": {"1": "ep1_s1"}, "q2": {"1": "ep1_s1"}}
    assert dumped_ep2 == {"q1": {"6": "ep2_s6"}, "q3": {"6": "ep2_s6"}}


def test_train_capping_is_per_epoch():
    # max_train_samples = 2 applies per epoch
    store = PredictionDumpStore(max_train_samples=2)

    # Epoch 1: fill to cap
    assert store.add_train_records(1, [("q1", "a")], epoch="1") == 1
    assert store.add_train_records(2, [("q2", "b")], epoch="1") == 1
    assert store.train_full(epoch="1")
    assert store.add_train_records(3, [("q3", "c")], epoch="1") == 0
    assert store.get_train_record_count("1") == 2

    # Epoch 2: cap starts fresh
    assert not store.train_full(epoch="2")
    assert store.add_train_records(6, [("q1", "d")], epoch="2") == 1
    assert store.add_train_records(7, [("q2", "e")], epoch="2") == 1
    assert store.train_full(epoch="2")
    assert store.add_train_records(8, [("q3", "f")], epoch="2") == 0
    assert store.get_train_record_count("2") == 2


def test_eval_records_per_epoch_and_flush(tmp_path: Path):
    store = PredictionDumpStore(
        eval_path_template=str(tmp_path / "eval_predictions_ep{epoch}.json"),
        output_dir=str(tmp_path),
    )
    # Eval after epoch 1
    assert store.add_eval_records([("q1", "eval1_q1"), ("q2", "eval1_q2")], epoch=1.0) == 2
    store.flush_eval(epoch=1.0)

    # Eval after epoch 2
    assert store.add_eval_records([("q1", "eval2_q1"), ("q2", "eval2_q2")], epoch=2.0) == 2
    store.flush_eval(epoch=2.0)

    dumped_ep1 = json.loads((tmp_path / "eval_predictions_ep1.json").read_text(encoding="utf-8"))
    dumped_ep2 = json.loads((tmp_path / "eval_predictions_ep2.json").read_text(encoding="utf-8"))

    assert dumped_ep1 == {"q1": "eval1_q1", "q2": "eval1_q2"}
    assert dumped_ep2 == {"q1": "eval2_q1", "q2": "eval2_q2"}


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

