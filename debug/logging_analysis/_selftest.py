"""Standalone checks used by ``--self-test`` and pytest."""

from __future__ import annotations

import json
from pathlib import Path

from _annotations import (
    expand_dataset_path,
    extract_boxed_answers,
    extract_ground_truth_text,
    index_annotations,
    load_annotation_records,
    lookup_annotation,
    mapped_column,
)
from _evaldirs import (
    AnalysisRun,
    assign_run_steps,
    collect_trainer_tables,
    discover_eval_dir,
    infer_run_name,
    load_trainer_log,
    parse_named_path,
    resolve_analysis_runs,
    summarize_trainer_log,
)
from _logparse import (
    UNKNOWN_DATASET,
    WarningRecord,
    build_matchers,
    flatten_prediction_log,
    identify_dataset,
    parse_question_index,
)
from _metrics import analyze_repetition, analyze_tags, simple_tokenize


def run_self_tests() -> None:
    _test_key_parsing()
    _test_thinker_alias()
    _test_canonical_tags()
    _test_missing_answer_tag()
    _test_empty_reasoning()
    _test_malformed_order()
    _test_token_repetition()
    _test_phrase_repetition()
    _test_multiple_steps()
    _test_unknown_dataset()
    _test_eval_dump()
    _test_step_keyed_eval_dump()
    _test_named_path_and_run_name()
    _test_trainer_log_json_and_jsonl()
    _test_multi_eval_dirs_distinct_steps()
    _test_bare_eval_prediction_logs_do_not_collapse()
    _test_hf_hub_cache_expand()
    _test_gt_field_from_columns()
    _test_boxed_extraction()
    _test_annotation_index()


def _matchers():
    info = {
        "Scene30k": {"file_name": "Scene30k.parquet", "columns": {"response": "cot"}},
        "SpatialSSRL_coldstart": {"file_name": "SFT-coldstart.json", "columns": {"response": "output"}},
        "3DThinker10k": {"file_name": "3dthinker10k_cot.with_question_id.jsonl", "columns": {"response": "output"}},
    }
    return build_matchers(info), info


def _test_key_parsing() -> None:
    matchers, _info = _matchers()
    key = "/tmp/cot_stage/annotations/SpatialSSRL_coldstart.json_1661"
    dataset, alias, others = identify_dataset(key, matchers)
    assert dataset == "SpatialSSRL_coldstart", dataset
    assert parse_question_index(key) == 1661
    assert others == []
    assert alias


def _test_thinker_alias() -> None:
    matchers, _info = _matchers()
    key = "/tmp/cot_stage/annotations/3dthinker10k_cot.jsonl_9178"
    dataset, _alias, _others = identify_dataset(key, matchers)
    assert dataset == "3DThinker10k", dataset
    assert dataset != UNKNOWN_DATASET
    assert dataset != "3dthinker10k"


def _test_canonical_tags() -> None:
    text = "<think>There is a chair left of the table.</think>\n<answer>chair</answer>"
    tags = analyze_tags(text)
    assert tags["canonical_format"] is True
    assert tags["usable_format"] is True
    assert tags["think_text"] == "There is a chair left of the table."
    assert tags["answer_text"] == "chair"


def _test_missing_answer_tag() -> None:
    text = "<think>reasoning</think>\nchair"
    tags = analyze_tags(text)
    assert tags["has_complete_answer_pair"] is False
    assert tags["canonical_format"] is False
    assert tags["has_complete_think_pair"] is True


def _test_empty_reasoning() -> None:
    text = "<think></think><answer>chair</answer>"
    tags = analyze_tags(text)
    assert tags["think_is_empty"] is True
    assert tags["canonical_format"] is False
    assert tags["has_complete_think_pair"] is True


def _test_malformed_order() -> None:
    text = "<answer>chair</answer><think>reasoning</think>"
    tags = analyze_tags(text)
    assert tags["canonical_format"] is False
    assert tags["proper_tag_order"] is False
    assert tags["think_before_answer"] is False


def _test_token_repetition() -> None:
    text = "chair chair chair chair chair chair"
    tokens = simple_tokenize(text)
    assert tokens == ["chair"] * 6
    stats = analyze_repetition(text)
    assert stats["max_identical_token_run"] == 6, stats["max_identical_token_run"]


def _test_phrase_repetition() -> None:
    text = "left of chair left of chair left of chair"
    stats = analyze_repetition(text)
    assert stats["max_identical_token_run"] == 1
    assert stats["ngram3_max_frequency"] >= 3
    assert stats["ngram3_repeated_fraction"] > 0.5


def _test_multiple_steps() -> None:
    matchers, _info = _matchers()
    warnings: list[WarningRecord] = []
    data = {
        "/tmp/cot_stage/annotations/Scene30k.parquet_1": {
            "10": "<think>a</think><answer>b</answer>",
            "20": "<think>c</think><answer>d</answer>",
        }
    }
    rows = flatten_prediction_log(data, "train_predictions_ep1.json", matchers, warnings)
    assert len(rows) == 2
    steps = sorted(row.step for row in rows)
    assert steps == [10, 20]
    assert rows[0].dataset == "Scene30k"
    assert rows[0].question_id == "Scene30k_1"


def _test_unknown_dataset() -> None:
    matchers, _info = _matchers()
    warnings: list[WarningRecord] = []
    data = {"totally_unrelated_file.json_3": {"5": "hello"}}
    rows = flatten_prediction_log(data, "train.json", matchers, warnings)
    assert rows[0].dataset == UNKNOWN_DATASET
    assert any(rec.code == "unknown_dataset" for rec in warnings)


def _test_eval_dump() -> None:
    matchers, _info = _matchers()
    warnings: list[WarningRecord] = []
    data = {"/tmp/cot_stage/annotations/Scene30k.parquet_19": "<think>x</think><answer>y</answer>"}
    rows = flatten_prediction_log(data, "eval_predictions_ep2.json", matchers, warnings)
    assert len(rows) == 1
    assert rows[0].source_kind == "eval"
    assert rows[0].step == 2
    assert rows[0].dataset == "Scene30k"


def _test_step_keyed_eval_dump() -> None:
    matchers, _info = _matchers()
    warnings: list[WarningRecord] = []
    data = {
        "Scene30k_1": {"0": "<think>a</think><answer>b</answer>", "10": "<think>c</think><answer>d</answer>"},
        "SpatialSSRL_coldstart_2": {"0": "x", "10": "y"},
    }
    rows = flatten_prediction_log(data, "eval_predictions_ep0.json", matchers, warnings)
    assert len(rows) == 4
    assert {row.question_id for row in rows} == {"Scene30k_1", "SpatialSSRL_coldstart_2"}
    assert sorted({row.step for row in rows}) == [0, 10]
    assert not any(rec.code == "multiple_steps" for rec in warnings)


def _test_hf_hub_cache_expand() -> None:
    expanded = expand_dataset_path("${HF_HUB_CACHE}/foo.json", hf_hub_cache="/tmp/hub")
    assert expanded.replace("\\", "/").endswith("/foo.json")
    assert "hub" in expanded


def _test_gt_field_from_columns() -> None:
    spec = {"columns": {"response": "cot", "prompt": "question_with_image_tags"}}
    assert mapped_column(spec, "response") == "cot"
    record = {"cot": "<think>t</think><answer>a</answer>", "output": "other"}
    text, field, _extra = extract_ground_truth_text(record, preferred_field="cot")
    assert field == "cot"
    assert text.startswith("<think>")


def _test_boxed_extraction() -> None:
    text = r"Therefore the camera sees left. \(\boxed{D}\)"
    boxed = extract_boxed_answers(text)
    assert boxed == ["D"], boxed


def _test_annotation_index(tmp_path: Path | None = None) -> None:
    root = Path(tmpfile_dir()) if tmp_path is None else tmp_path
    jsonl = root / "toy.jsonl"
    jsonl.write_text(
        json.dumps({"question_id": "Toy_0", "output": "alpha", "idx": 99})
        + "\n"
        + json.dumps({"question_id": "Toy_1", "output": "beta", "idx": 100})
        + "\n",
        encoding="utf-8",
    )
    records = load_annotation_records(jsonl)
    index = index_annotations(records, "Toy", path=jsonl)
    assert lookup_annotation(index, "Toy_1", 1)["output"] == "beta"
    # File-order index 0, not the annotation `idx` field.
    assert lookup_annotation(index, "missing", 0)["output"] == "alpha"


def tmpfile_dir() -> str:
    import tempfile

    return tempfile.mkdtemp(prefix="log_analyzer_selftest_")


def _write_eval_dir(root: Path, name: str, prediction: str, eval_loss: float, jsonl: bool = True) -> Path:
    folder = root / name / "lora" / "eval"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"/tmp/cot_stage/annotations/Scene30k.parquet_19": prediction}
    (folder / "eval_predictions.json").write_text(json.dumps(payload), encoding="utf-8")
    (folder / "eval_results.json").write_text(
        json.dumps({"eval_loss": eval_loss, "eval_runtime": 1.5}), encoding="utf-8"
    )
    progress = [
        {"current_steps": 5, "total_steps": 10, "percentage": 50.0, "elapsed_time": "0:00:01"},
        {
            "current_steps": 0,
            "total_steps": 10,
            "eval_loss": eval_loss,
            "percentage": 0.0,
            "elapsed_time": "0:00:02",
            "remaining_time": "0:00:00",
        },
    ]
    if jsonl:
        (folder / "trainer_log.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in progress), encoding="utf-8"
        )
    else:
        (folder / "trainer_log.json").write_text(json.dumps(progress), encoding="utf-8")
    return folder


def _test_named_path_and_run_name() -> None:
    name, path = parse_named_path("base=/tmp/foo/lora/eval")
    assert name == "base"
    assert path == Path("/tmp/foo/lora/eval")
    name, path = parse_named_path("/tmp/foo/lora/eval")
    assert name is None
    assert path == Path("/tmp/foo/lora/eval")
    assert infer_run_name("/tmp/models/Qwen2.5-VL-7B-Instruct/lora/eval") == "Qwen2.5-VL-7B-Instruct"
    assert (
        infer_run_name(
            "/tmp/models/qwen2_5vl-7b-lora-sft-CoT_traineval_1epochs_merged/lora/eval/eval_predictions.json"
        )
        == "qwen2_5vl-7b-lora-sft-CoT_traineval_1epochs_merged"
    )


def _test_trainer_log_json_and_jsonl() -> None:
    root = Path(tmpfile_dir())
    jsonl_dir = _write_eval_dir(root, "jsonl_run", "<think>a</think><answer>b</answer>", 0.9, jsonl=True)
    json_dir = _write_eval_dir(root, "json_run", "<think>c</think><answer>d</answer>", 0.8, jsonl=False)
    jsonl_rows = load_trainer_log(jsonl_dir / "trainer_log.jsonl")
    json_rows = load_trainer_log(json_dir / "trainer_log.json")
    assert len(jsonl_rows) == 2
    assert len(json_rows) == 2
    assert summarize_trainer_log(jsonl_rows)["eval_loss"] == 0.9
    assert summarize_trainer_log(json_rows)["eval_loss"] == 0.8
    files = discover_eval_dir(jsonl_dir)
    assert files.predictions and files.predictions[0].name == "eval_predictions.json"
    assert files.trainer_log is not None


def _test_multi_eval_dirs_distinct_steps() -> None:
    from _aggregate import question_trajectories

    matchers, _info = _matchers()
    root = Path(tmpfile_dir())
    base = _write_eval_dir(root, "Qwen2.5-VL-7B-Instruct", "no tags yet", 1.67)
    ep1 = _write_eval_dir(
        root,
        "qwen2_5vl-7b-lora-sft-CoT_traineval_1epochs_merged",
        "<think>x</think><answer>y</answer>",
        0.95,
    )
    warnings: list[WarningRecord] = []
    runs = resolve_analysis_runs(
        None,
        [f"base={base}", f"ep1={ep1}"],
        ["base=0", "ep1=620"],
        warnings,
    )
    assert [run.run_name for run in runs] == ["base", "ep1"]
    assert [run.step for run in runs] == [0, 620]
    rows = []
    for run in runs:
        data = json.loads(run.prediction_path.read_text(encoding="utf-8"))
        rows.extend(
            flatten_prediction_log(
                data,
                run.prediction_path,
                matchers,
                warnings,
                run_name=run.run_name,
                step_override=run.step,
            )
        )
    assert {row.run_name for row in rows} == {"base", "ep1"}
    assert sorted(row.step for row in rows) == [0, 620]
    assert len({(row.question_id, row.step) for row in rows}) == 2
    summary_rows, history_rows = collect_trainer_tables(runs, warnings)
    assert [row["eval_loss"] for row in summary_rows] == [1.67, 0.95]
    assert len(history_rows) == 4
    import pandas as pd

    frame = pd.DataFrame(
        {
            "question_id": [row.question_id for row in rows],
            "step": [row.step for row in rows],
            "run_name": [row.run_name for row in rows],
            "dataset": [row.dataset for row in rows],
            "prediction": [row.prediction for row in rows],
            "think_text": ["", "x"],
            "answer_text": ["", "y"],
            "canonical_format": [False, True],
            "tag_presence_score": [0.0, 1.0],
            "think_token_count": [0, 1],
            "answer_token_count": [0, 1],
            "repetition_score": [0.0, 0.0],
            "normalized_exact_match": [False, False],
        }
    )
    traj = question_trajectories(frame)
    assert len(traj) == 2
    assert set(traj["run_name"]) == {"base", "ep1"}


def _test_bare_eval_prediction_logs_do_not_collapse() -> None:
    matchers, _info = _matchers()
    root = Path(tmpfile_dir())
    a = _write_eval_dir(root, "run_a", "first", 1.1)
    b = _write_eval_dir(root, "run_b", "second", 1.0)
    warnings: list[WarningRecord] = []
    runs = resolve_analysis_runs(
        [str(a / "eval_predictions.json"), str(b / "eval_predictions.json")],
        None,
        None,
        warnings,
    )
    assert len(runs) == 2
    assert runs[0].run_name != runs[1].run_name
    assert runs[0].step != runs[1].step
    rows = []
    for run in runs:
        data = json.loads(run.prediction_path.read_text(encoding="utf-8"))
        rows.extend(
            flatten_prediction_log(
                data,
                run.prediction_path,
                matchers,
                warnings,
                run_name=run.run_name,
                step_override=run.step,
            )
        )
    assert len(rows) == 2
    assert rows[0].step != rows[1].step
    assert {row.prediction for row in rows} == {"first", "second"}
    # Default epoch-less eval dumps used to share step=0.
    colliding = [
        AnalysisRun(run_name="x", prediction_path=a / "eval_predictions.json", eval_dir=a),
        AnalysisRun(run_name="y", prediction_path=b / "eval_predictions.json", eval_dir=b),
    ]
    assign_run_steps(colliding, None, warnings)
    assert colliding[0].step != colliding[1].step
