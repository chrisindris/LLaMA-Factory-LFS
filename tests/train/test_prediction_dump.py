# Copyright 2025 the LlamaFactory team.
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

from types import SimpleNamespace

import torch
from transformers import Seq2SeqTrainer

from llamafactory.train.prediction_dump import PredictionDumpStore
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


def _make_dump_trainer(tmp_path, *, eval_prediction_mode: str = "teacher_forced"):
    trainer = CustomSeq2SeqTrainer.__new__(CustomSeq2SeqTrainer)
    trainer.finetuning_args = SimpleNamespace(
        save_eval_predictions=True,
        eval_prediction_mode=eval_prediction_mode,
    )
    trainer.args = SimpleNamespace(predict_with_generate=False)
    trainer.prediction_dump = PredictionDumpStore(eval_path=str(tmp_path / "eval_predictions.json"))
    trainer._eval_pred_buffer = []
    trainer._pred_dump_warned_missing_qid = False
    return trainer


def test_eval_dump_loss_only_does_not_return_logits(monkeypatch, tmp_path):
    trainer = _make_dump_trainer(tmp_path)
    captured: dict[str, object] = {}
    parent_logits = torch.zeros(1, 4, 32)

    def fake_parent_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None, **kwargs):
        captured["prediction_loss_only"] = prediction_loss_only
        return torch.tensor(0.5), parent_logits, inputs.get("labels")

    monkeypatch.setattr(Seq2SeqTrainer, "prediction_step", fake_parent_step)
    trainer._texts_from_teacher_forced = lambda *args, **kwargs: ["dumped text"]

    inputs = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
        "question_ids": ["q1"],
    }
    loss, logits, labels = CustomSeq2SeqTrainer.prediction_step(
        trainer, model=object(), inputs=inputs, prediction_loss_only=True
    )

    assert captured["prediction_loss_only"] is True
    assert logits is None
    assert labels is None
    assert float(loss) == 0.5
    assert trainer._eval_pred_buffer == [("q1", "dumped text")]


def test_eval_dump_keeps_logits_when_metrics_need_them(monkeypatch, tmp_path):
    trainer = _make_dump_trainer(tmp_path)
    parent_logits = torch.zeros(1, 4, 32)
    parent_labels = torch.ones(1, 4, dtype=torch.long)

    def fake_parent_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None, **kwargs):
        captured_flag.append(prediction_loss_only)
        return torch.tensor(0.25), parent_logits, parent_labels

    captured_flag: list[bool] = []
    monkeypatch.setattr(Seq2SeqTrainer, "prediction_step", fake_parent_step)
    trainer._texts_from_teacher_forced = lambda *args, **kwargs: ["dumped text"]

    inputs = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": parent_labels,
        "question_ids": ["q2"],
    }
    loss, logits, labels = CustomSeq2SeqTrainer.prediction_step(
        trainer, model=object(), inputs=inputs, prediction_loss_only=False
    )

    assert captured_flag == [False]
    assert logits is parent_logits
    assert labels is parent_labels
    assert float(loss) == 0.25
    assert trainer._eval_pred_buffer == [("q2", "dumped text")]
