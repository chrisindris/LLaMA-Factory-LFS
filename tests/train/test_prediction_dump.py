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
from types import SimpleNamespace

import torch

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
    assert (
        resolve_prediction_dump_path(None, "1", "train_predictions", "/tmp/out")
        == "/tmp/out/train_predictions_ep1.json"
    )
    assert (
        resolve_prediction_dump_path(None, "2", "eval_predictions", "/tmp/out") == "/tmp/out/eval_predictions_ep2.json"
    )
    assert (
        resolve_prediction_dump_path("/custom/train_{epoch}.json", "1", "train_predictions") == "/custom/train_1.json"
    )
    assert (
        resolve_prediction_dump_path("/custom/train_predictions.json", "2", "train_predictions")
        == "/custom/train_predictions_ep2.json"
    )
    assert (
        resolve_prediction_dump_path("/custom/train_predictions_ep1.json", "1", "train_predictions")
        == "/custom/train_predictions_ep1.json"
    )


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
    assert store.add_eval_records([("q1", "eval1_q1"), ("q2", "eval1_q2")], epoch=1.0, step=620) == 2
    store.flush_eval(epoch=1.0)

    # Eval after epoch 2
    assert store.add_eval_records([("q1", "eval2_q1"), ("q2", "eval2_q2")], epoch=2.0, step=1240) == 2
    store.flush_eval(epoch=2.0)

    dumped_ep1 = json.loads((tmp_path / "eval_predictions_ep1.json").read_text(encoding="utf-8"))
    dumped_ep2 = json.loads((tmp_path / "eval_predictions_ep2.json").read_text(encoding="utf-8"))

    assert dumped_ep1 == {"q1": {"620": "eval1_q1"}, "q2": {"620": "eval1_q2"}}
    assert dumped_ep2 == {"q1": {"1240": "eval2_q1"}, "q2": {"1240": "eval2_q2"}}


def test_eval_records_keep_multiple_steps_in_same_epoch(tmp_path: Path):
    store = PredictionDumpStore(
        eval_path_template=str(tmp_path / "eval_predictions_ep{epoch}.json"),
        output_dir=str(tmp_path),
    )
    assert store.add_eval_records([("Scene30k_1", "t0")], epoch=0.0, step=0) == 1
    assert store.add_eval_records([("Scene30k_1", "t10")], epoch=0.02, step=10) == 1
    store.flush_eval(epoch="0")
    dumped = json.loads((tmp_path / "eval_predictions_ep0.json").read_text(encoding="utf-8"))
    assert dumped == {"Scene30k_1": {"0": "t0", "10": "t10"}}


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
    pad_token_id = 0
    unk_token_id = None
    im_start_id = 7

    def decode(self, ids, skip_special_tokens=True):
        return ",".join(str(int(i)) for i in ids)

    def convert_tokens_to_ids(self, token):
        if token == "<|im_start|>":
            return self.im_start_id
        return 0


def test_decode_teacher_forced_batch_greedy_on_response_positions():
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


class _CaptureGenerateModel:
    def __init__(self):
        self.training = False
        self.generate_kwargs = None
        self.rope_deltas_at_generate = "missing"

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        self.rope_deltas_at_generate = getattr(self, "rope_deltas", "missing")
        input_ids = kwargs["input_ids"]
        extra = input_ids.new_full((input_ids.size(0), 1), 9)
        return torch.cat([input_ids, extra], dim=1)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


def _make_generate_dump_trainer(*, max_new_tokens: int = 2048, eval_dump_max_new_tokens: int = 256):
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    trainer = CustomSeq2SeqTrainer.__new__(CustomSeq2SeqTrainer)
    trainer.processing_class = _IdTokenizer()
    trainer._dump_skip_special_tokens = True
    trainer._gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": True, "temperature": 0.95}
    trainer.finetuning_args = SimpleNamespace(eval_dump_max_new_tokens=eval_dump_max_new_tokens)
    return trainer


def test_texts_from_generate_strips_indices():
    trainer = _make_generate_dump_trainer()
    model = _CaptureGenerateModel()
    labels = torch.tensor([[IGNORE_INDEX, 1, 2]])
    inputs = {
        "input_ids": torch.tensor([[10, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": labels,
        "pixel_values": torch.ones(1, 2, 2),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "position_ids": torch.zeros(4, 1, 3, dtype=torch.long),
        "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(1, 3, dtype=torch.long),
        "cache_position": torch.arange(3),
        "_indices": torch.tensor([42]),
        "question_ids": ["q1"],
        "debug_samples": [{"sample_idx": 0}],
    }
    texts = trainer._texts_from_generate(model, inputs, labels)

    assert model.generate_kwargs is not None
    assert "_indices" not in model.generate_kwargs
    assert "question_ids" not in model.generate_kwargs
    assert "debug_samples" not in model.generate_kwargs
    assert "labels" not in model.generate_kwargs
    assert "position_ids" not in model.generate_kwargs
    assert "rope_deltas" not in model.generate_kwargs
    assert "mm_token_type_ids" not in model.generate_kwargs
    assert "cache_position" not in model.generate_kwargs
    assert "pixel_values" in model.generate_kwargs
    assert "image_grid_thw" in model.generate_kwargs
    assert list(model.generate_kwargs["input_ids"].shape) == [1, 1]
    assert texts == ["9"]


def test_texts_from_generate_omits_empty_vision():
    trainer = _make_generate_dump_trainer()
    model = _CaptureGenerateModel()
    labels = torch.tensor([[IGNORE_INDEX, 1, 2]])
    inputs = {
        "input_ids": torch.tensor([[10, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": labels,
        "pixel_values": torch.zeros(0, 1176),
        "image_grid_thw": torch.zeros(0, 3, dtype=torch.long),
        "pixel_values_videos": torch.zeros(0, 1176),
        "video_grid_thw": torch.zeros(0, 3, dtype=torch.long),
        "position_ids": torch.zeros(4, 1, 3, dtype=torch.long),
    }
    texts = trainer._texts_from_generate(model, inputs, labels)

    assert model.generate_kwargs is not None
    assert "pixel_values" not in model.generate_kwargs
    assert "image_grid_thw" not in model.generate_kwargs
    assert "pixel_values_videos" not in model.generate_kwargs
    assert "video_grid_thw" not in model.generate_kwargs
    assert "position_ids" not in model.generate_kwargs
    assert texts == ["9"]


def test_texts_from_generate_clears_stale_rope_deltas():
    trainer = _make_generate_dump_trainer()
    model = _CaptureGenerateModel()
    model.rope_deltas = torch.tensor([[99]])
    model.model = type("Inner", (), {})()
    model.model.rope_deltas = torch.tensor([[7]])
    labels = torch.tensor([[IGNORE_INDEX, 1]])
    inputs = {
        "input_ids": torch.tensor([[10, 1]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "labels": labels,
    }
    trainer._texts_from_generate(model, inputs, labels)
    assert model.rope_deltas_at_generate is None
    assert int(model.rope_deltas.item()) == 99
    assert int(model.model.rope_deltas.item()) == 7
    assert "rope_deltas" not in model.generate_kwargs


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
    assert args.eval_dump_max_new_tokens == 256
    assert args.allow_warm_start_resume is False
    assert args.require_resume_bundle
    assert args.resume_bundle_dir == "/tmp/resume_bundle"
    assert args.stop_at_global_step == 1240


def test_texts_from_generate_is_greedy_capped_and_suppresses_im_start():
    trainer = _make_generate_dump_trainer(max_new_tokens=2048, eval_dump_max_new_tokens=256)
    model = _CaptureGenerateModel()
    labels = torch.tensor([[IGNORE_INDEX, 1, 2]])
    inputs = {
        "input_ids": torch.tensor([[10, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": labels,
    }
    texts = trainer._texts_from_generate(model, inputs, labels, question_ids=["q1"])

    assert texts == ["9"]
    kwargs = model.generate_kwargs
    assert kwargs is not None
    assert kwargs["do_sample"] is False
    assert kwargs["max_new_tokens"] == 256
    assert "max_length" not in kwargs
    assert _IdTokenizer.im_start_id in kwargs["suppress_tokens"]
    assert [_IdTokenizer.im_start_id] in kwargs["bad_words_ids"]


def test_eval_dump_max_new_tokens_zero_keeps_configured_cap():
    trainer = _make_generate_dump_trainer(max_new_tokens=128, eval_dump_max_new_tokens=0)
    kwargs = trainer._build_dump_generate_kwargs(trainer.processing_class)
    assert kwargs["max_new_tokens"] == 128
    assert kwargs["do_sample"] is False


def test_prediction_step_generate_mode_does_not_call_generate(monkeypatch, tmp_path):
    from transformers import Seq2SeqTrainer

    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    trainer = CustomSeq2SeqTrainer.__new__(CustomSeq2SeqTrainer)
    trainer.finetuning_args = SimpleNamespace(
        save_eval_predictions=True,
        eval_prediction_mode="generate",
        eval_dump_max_new_tokens=256,
    )
    trainer.args = SimpleNamespace(predict_with_generate=False)
    trainer.prediction_dump = PredictionDumpStore(eval_path_template=str(tmp_path / "eval_predictions.json"))
    trainer._eval_pred_buffer = []
    trainer._pred_dump_warned_missing_qid = False
    trainer.processing_class = _IdTokenizer()
    trainer._dump_skip_special_tokens = True
    trainer._gen_kwargs = {"max_new_tokens": 2048, "do_sample": True}

    generate_calls = {"n": 0}
    parent_logits = torch.zeros(1, 4, 8)
    parent_labels = torch.ones(1, 4, dtype=torch.long)

    def fake_parent_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None, **kwargs):
        return torch.tensor(0.5), parent_logits, parent_labels

    monkeypatch.setattr(Seq2SeqTrainer, "prediction_step", fake_parent_step)

    class BoomGenerate:
        training = False

        def generate(self, **kwargs):
            generate_calls["n"] += 1
            raise AssertionError("generate should not run in prediction_step")

    inputs = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": parent_labels.clone(),
        "question_ids": ["q1"],
    }
    loss, logits, labels = CustomSeq2SeqTrainer.prediction_step(
        trainer, model=BoomGenerate(), inputs=inputs, prediction_loss_only=True
    )

    assert generate_calls["n"] == 0
    assert trainer._eval_pred_buffer == []
    assert logits is None
    assert labels is None
    assert float(loss) == 0.5


def test_deferred_eval_generate_pass_records_pairs_and_empty_flush_gathers(tmp_path):
    trainer = _make_generate_dump_trainer()
    trainer.finetuning_args = SimpleNamespace(
        save_eval_predictions=True,
        eval_prediction_mode="generate",
        eval_dump_max_new_tokens=256,
    )
    trainer.args = SimpleNamespace(
        predict_with_generate=False,
        output_dir=str(tmp_path),
        report_to=[],
    )
    trainer.prediction_dump = PredictionDumpStore(
        eval_path_template=str(tmp_path / "eval_predictions_ep{epoch}.json"),
        output_dir=str(tmp_path),
    )
    trainer._eval_pred_buffer = []
    trainer._pred_dump_warned_missing_qid = False
    trainer.state = SimpleNamespace(epoch=0.0, global_step=30)
    trainer._distributed_world_size = lambda: 1
    trainer.is_world_process_zero = lambda: True

    model = _CaptureGenerateModel()
    trainer.model = model
    batch = {
        "input_ids": torch.tensor([[10, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": torch.tensor([[IGNORE_INDEX, 1, 2]]),
        "question_ids": ["q1"],
    }
    trainer.eval_dataset = object()
    trainer.get_eval_dataloader = lambda eval_dataset=None: [batch]
    trainer._prepare_inputs = lambda x: dict(x)

    trainer._dump_eval_generate_pass()
    assert trainer._eval_pred_buffer == [("q1", "9")]
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 256

    trainer._eval_pred_buffer = []
    gathered = {"called": False}

    def fake_gather(pairs):
        gathered["called"] = True
        assert pairs == []
        return []

    trainer._gather_prediction_pairs = fake_gather
    trainer._flush_eval_predictions()
    assert gathered["called"] is True
    assert trainer._eval_pred_buffer == []
