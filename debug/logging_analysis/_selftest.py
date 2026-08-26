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
