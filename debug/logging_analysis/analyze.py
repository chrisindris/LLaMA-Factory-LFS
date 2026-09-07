#!/usr/bin/env python3
r"""Deterministic analyzer for LLaMA-Factory train/eval prediction dumps.

Purpose
-------
Watch tag adherence, length, repetition, surface match, and (optionally)
LanguageTool / checkpoint probe loss while a Qwen2.5-VL model is SFT'd on the
CoT mix (Scene30k, SpatialSSRL_coldstart, 3DThinker10k).

This is **not** an LLM-as-a-judge. It never calls an external AI API, never
embeds text, and never treats grammar or exact match as semantic correctness.

Expected log structure
----------------------
Train dumps written by ``src/llamafactory/train/prediction_dump.py``::

    { "<QUESTION_ID>": { "<global_step>": "<model_output>" }, ... }

In this fork QUESTION_ID is often a staged annotation path plus a file-order
index, because ``dataset_info.json`` does not map ``columns.question_id``::

    /tmp/cot_stage/annotations/SpatialSSRL_coldstart.json_1661

Eval dumps are ``{ "<QUESTION_ID>": "<model_output>" }`` and are treated as a
single synthetic step (epoch from ``*_epN.json`` when present).

The YAML in this repo often uses ``train_prediction_mode: teacher_forced``:
dumped strings are argmax next-token decodes of response positions, **not**
``model.generate`` output. Metrics still describe those strings.

Dataset resolution
------------------
Canonical names match ``data/dataset_info.json`` keys. Aliases include
``3dthinker10k_cot`` → ``3DThinker10k``. Annotation ``file_name`` values may
contain ``${HF_HUB_CACHE}``; pass ``--hf-hub-cache`` or export the env var.
Do not use 3DThinker's ``idx`` field as the question index.

What runs without a model
    All log parsing, annotation joins, tag/length/repetition/vocab/punctuation
    diagnostics, plots, and the markdown report.

What requires LanguageTool
    ``--grammar`` (optional JRE + ``language_tool_python``). Diagnostic only.

What requires checkpoints / LLaMA-Factory
    ``--compute-loss``. This is **reference-target teacher-forced CE** at a
    checkpoint ("checkpoint probe loss"), **not** the trainer's original batch
    loss. Prediction strings alone cannot reconstruct SFT loss.

Example:
-------
::

    python debug/logging_analysis/analyze.py \\
        --log saves/qwen2_5vl-7b/lora/sft/CoT_traineval_resume_ep1/train_predictions_ep1.json \\
        --dataset-info data/dataset_info.json \\
        --output-dir debug/logging_analysis/out/CoT_traineval_resume_ep1

    python debug/logging_analysis/analyze.py \\
        --eval-dir base=models/Qwen2.5-VL-7B-Instruct/lora/eval \\
        --eval-dir ep1=models/qwen2_5vl-7b-lora-sft-CoT_traineval_1epochs_merged/lora/eval \\
        --eval-dir ep2=models/qwen2_5vl-7b-lora-sft-CoT_traineval_2epochs_merged/lora/eval \\
        --eval-dir ep3=models/qwen2_5vl-7b-lora-sft-CoT_traineval_3epochs_merged/lora/eval \\
        --output-dir debug/logging_analysis/out/eval_compare
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any


_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from _aggregate import (  # noqa: E402
    add_previous_prediction_flags,
    add_row_flags,
    apply_matched_questions,
    correlation_tables,
    health_by_step,
    interesting_examples,
    overlap_table,
    question_ids_by_step,
    question_trajectories,
    step_change_flags,
    summarize_frame,
)
from _annotations import (  # noqa: E402
    extract_boxed_answers,
    extract_ground_truth_text,
    extract_prompt_text,
    load_annotation_indices,
    load_dataset_info,
    lookup_annotation,
    parse_kv_overrides,
)
from _evaldirs import (  # noqa: E402
    collect_trainer_tables,
    needs_synthetic_step,
    resolve_analysis_runs,
)
from _logparse import (  # noqa: E402
    UNKNOWN_DATASET,
    WarningRecord,
    build_matchers,
    flatten_prediction_log,
    load_prediction_log,
)
from _metrics import per_prediction_metrics  # noqa: E402
from _plots import generate_plots  # noqa: E402
from _report import build_analysis_summary, build_report, dump_json  # noqa: E402


logger = logging.getLogger("train_log_analyzer")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_INFO = REPO_ROOT / "data" / "dataset_info.json"
DEFAULT_DATASET_DIR = REPO_ROOT / "data"

LONG_TEXT_COLUMNS = (
    "prediction",
    "ground_truth",
    "think_text",
    "answer_text",
    "gt_think_text",
    "gt_answer_text",
    "text_before_think",
    "text_between_think_and_answer",
    "text_after_answer",
    "prompt_text",
    "best_reference",
    "longest_repeated_span",
    "most_repeated_span",
    "flags",
)


def _tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic diagnostics for LLaMA-Factory prediction dumps "
            "(tag adherence, length, repetition, surface match). "
            "Does not call an LLM. Dump strings alone cannot reconstruct SFT loss. "
            "Teacher-forced dumps are argmax next-token decodes, not generate()."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log",
        action="append",
        dest="logs",
        default=None,
        help=(
            "Prediction JSON or eval save folder (repeatable). "
            "Train: D[qid][step]=text. Eval: D[qid]=text. Directories are treated as --eval-dir."
        ),
    )
    parser.add_argument(
        "--eval-dir",
        action="append",
        dest="eval_dirs",
        default=None,
        help=(
            "Eval save folder (repeatable). Form [NAME=]PATH. Discovers eval_predictions.json "
            "and trainer_log.jsonl / trainer_log.json."
        ),
    )
    parser.add_argument(
        "--eval-step",
        action="append",
        dest="eval_steps",
        default=None,
        help="Map a run onto the numeric step axis: NAME=INT (repeatable).",
    )
    parser.add_argument(
        "--dataset-info", type=Path, default=DEFAULT_DATASET_INFO if DEFAULT_DATASET_INFO.exists() else None
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--hf-hub-cache", type=str, default=None, help="Expands ${HF_HUB_CACHE} in dataset_info file_name."
    )
    parser.add_argument("--annotation-file", action="append", default=None, help="DATASET=/path override. Repeatable.")
    parser.add_argument(
        "--gt-field", action="append", default=None, help="DATASET=column override for ground truth. Repeatable."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"))
    parser.add_argument("--tokenizer", type=Path, default=None, help="Local tokenizer dir. Never downloaded.")
    parser.add_argument("--grammar", action="store_true", help="Optional LanguageTool diagnostic (not accuracy).")
    parser.add_argument("--language", default="en-US")
    parser.add_argument(
        "--matched-questions",
        action="store_true",
        help="Restrict step trends to IDs seen at every step when possible.",
    )
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rolling-window", type=int, default=1)
    parser.add_argument("--compute-loss", action="store_true")
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--llamafactory-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--training-config", type=Path, default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--loss-batch-size", type=int, default=1)
    parser.add_argument("--loss-max-samples-per-dataset", type=int, default=16)
    parser.add_argument("--nearest-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--short-think-tokens", type=int, default=8)
    parser.add_argument("--long-think-tokens", type=int, default=4000)
    parser.add_argument("--high-repetition", type=float, default=0.35)
    parser.add_argument("--high-ngram-repetition", type=float, default=0.5)
    parser.add_argument("--canonical-drop", type=float, default=0.20)
    parser.add_argument("--think-growth", type=float, default=3.0)
    return parser.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def dependency_versions() -> dict[str, str | None]:
    names = [
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "pyarrow",
        "torch",
        "transformers",
        "llamafactory",
        "language_tool_python",
        "rapidfuzz",
        "tqdm",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def load_optional_tokenizer(path: Path | None) -> Any:
    if path is None:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
    except Exception as exc:
        logger.warning("Could not load local tokenizer from %s (%s). Using whitespace tokens.", path, exc)
        return None


def observations_to_frame(
    observations,
    indices,
    gt_field_overrides: dict[str, str],
    tokenizer,
    grammar,
    warnings: list[WarningRecord],
    args: argparse.Namespace,
):
    import pandas as pd
    from _grammar import grammar_columns_for_text, strip_xml_like_tags

    rows: list[dict[str, Any]] = []
    lookup_ok = 0
    lookup_fail = 0
    for obs in _tqdm(observations, desc="metrics"):
        index = indices.get(obs.dataset)
        record = lookup_annotation(index, obs.question_id, obs.question_index)
        preferred = gt_field_overrides.get(obs.dataset) or (index.response_field if index is not None else None)
        gt_text, gt_field, extra_refs = extract_ground_truth_text(
            record, preferred_field=preferred, warnings=warnings, log_key=obs.log_key
        )
        extra_refs = list(extra_refs) + extract_boxed_answers(gt_text)
        prompt_text = extract_prompt_text(record, index.prompt_field if index is not None else None)
        annotation_found = record is not None
        if annotation_found:
            lookup_ok += 1
        else:
            lookup_fail += 1
            if obs.dataset != UNKNOWN_DATASET:
                warnings.append(
                    WarningRecord(
                        code="annotation_lookup_failed",
                        message=f"No annotation for {obs.question_id}",
                        log_key=obs.log_key,
                        step=obs.step,
                    )
                )
        metrics = per_prediction_metrics(
            obs.prediction,
            tokenizer=tokenizer,
            ground_truth=gt_text,
            extra_references=extra_refs,
            short_think_tokens=args.short_think_tokens,
            long_think_tokens=args.long_think_tokens,
            high_repetition=args.high_repetition,
            high_ngram_repetition=args.high_ngram_repetition,
        )
        if grammar is not None:
            metrics.update(grammar_columns_for_text(grammar, strip_xml_like_tags(obs.prediction), "", tokenizer))
            metrics.update(grammar_columns_for_text(grammar, metrics.get("think_text"), "think_", tokenizer))
            metrics.update(grammar_columns_for_text(grammar, metrics.get("answer_text"), "answer_", tokenizer))
        rows.append(
            {
                "dataset": obs.dataset,
                "question_index": obs.question_index,
                "question_id": obs.question_id,
                "log_key": obs.log_key,
                "step": obs.step,
                "run_name": obs.run_name,
                "epoch": obs.epoch,
                "source_kind": obs.source_kind,
                "source_log": obs.source_log,
                "prediction": obs.prediction,
                "ground_truth": gt_text,
                "ground_truth_field": gt_field,
                "prompt_text": prompt_text,
                "annotation_found": annotation_found,
                "matched_alias": obs.matched_alias,
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["step"] = pd.to_numeric(frame["step"], errors="coerce").astype("Int64")
        frame["question_index"] = pd.to_numeric(frame["question_index"], errors="coerce").astype("Int64")
        sort_cols = [col for col in ("step", "run_name", "dataset", "question_id") if col in frame.columns]
        frame = frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return frame, lookup_ok, lookup_fail


def save_table(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame is None or frame.empty:
        frame = frame if frame is not None else __import__("pandas").DataFrame()
    frame.to_csv(path, index=False)


def save_detailed(frame, output_dir: Path) -> None:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "detailed_predictions.parquet"
    jsonl_path = output_dir / "detailed_predictions.jsonl"
    csv_path = output_dir / "detailed_predictions.csv"
    if frame is None or frame.empty:
        empty = frame if frame is not None else pd.DataFrame()
        empty.to_csv(csv_path, index=False)
        return
    try:
        frame.to_parquet(parquet_path, index=False)
    except Exception as exc:
        logger.warning("Parquet write failed (%s); writing JSONL instead.", exc)
        frame.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    csv_frame = frame.copy()
    for col in LONG_TEXT_COLUMNS:
        if col in csv_frame.columns:
            csv_frame[col] = csv_frame[col].map(lambda value: "" if pd.isna(value) else str(value)[:500])
    csv_frame.to_csv(csv_path, index=False)


def integrity_stats(observations, frame, warnings: list[WarningRecord]) -> dict[str, Any]:
    outer_keys = {obs.log_key for obs in observations}
    datasets = sorted({obs.dataset for obs in observations})
    known = [name for name in datasets if name != UNKNOWN_DATASET]
    steps = sorted({obs.step for obs in observations if obs.step is not None})
    dup = 0
    if not frame.empty:
        dup_cols = [col for col in ("run_name", "question_id", "step") if col in frame.columns]
        dup = int(frame.duplicated(dup_cols, keep=False).sum())
    codes = {}
    for rec in warnings:
        codes[rec.code] = codes.get(rec.code, 0) + 1
    return {
        "n_outer_records": len(outer_keys),
        "n_prediction_observations": len(observations),
        "datasets_recognized": known,
        "datasets_unrecognized": [UNKNOWN_DATASET] if UNKNOWN_DATASET in datasets else [],
        "step_min": steps[0] if steps else None,
        "step_max": steps[-1] if steps else None,
        "n_distinct_steps": len(steps),
        "n_unique_questions": int(frame["question_id"].nunique()) if not frame.empty else 0,
        "duplicate_question_step_rows": dup,
        "warning_counts": codes,
    }


def _attach_trainer_eval_loss(summary, trainer_summary):
    if summary is None or summary.empty or trainer_summary is None or trainer_summary.empty:
        return summary
    if "eval_loss" not in trainer_summary.columns:
        return summary
    if "run_name" in summary.columns and "run_name" in trainer_summary.columns:
        on = ["run_name"]
    elif "step" in summary.columns and "step" in trainer_summary.columns:
        on = ["step"]
    else:
        return summary
    right = (
        trainer_summary[on + ["eval_loss"]]
        .drop_duplicates(subset=on)
        .rename(columns={"eval_loss": "trainer_eval_loss"})
    )
    return summary.merge(right, on=on, how="left")


def run_analysis(args: argparse.Namespace) -> int:
    if not args.logs and not args.eval_dirs:
        raise SystemExit("Pass one or more --log files or --eval-dir folders (or --self-test).")
    warnings: list[WarningRecord] = []
    dataset_info: dict[str, Any] = {}
    if args.dataset_info and Path(args.dataset_info).exists():
        dataset_info = load_dataset_info(args.dataset_info)
    elif args.dataset_info:
        warnings.append(WarningRecord(code="dataset_info_missing", message=f"{args.dataset_info} not found"))
        logger.warning("dataset_info not found: %s", args.dataset_info)

    annotation_overrides = parse_kv_overrides(args.annotation_file)
    gt_field_overrides = parse_kv_overrides(args.gt_field)
    matchers = build_matchers(dataset_info)
    runs = resolve_analysis_runs(args.logs, args.eval_dirs, args.eval_steps, warnings)
    trainer_summary_rows, trainer_history_rows = collect_trainer_tables(runs, warnings)
    observations = []
    for run in runs:
        if run.prediction_path is None:
            continue
        if not Path(run.prediction_path).is_file():
            warnings.append(
                WarningRecord(
                    code="eval_predictions_missing",
                    message=f"Prediction file not found: {run.prediction_path}",
                    extra=str(run.prediction_path),
                )
            )
            continue
        data = load_prediction_log(run.prediction_path)
        observations.extend(
            flatten_prediction_log(
                data,
                run.prediction_path,
                matchers,
                warnings,
                run_name=run.run_name,
                step_override=run.step if needs_synthetic_step(run) else None,
            )
        )
    if not observations and not trainer_summary_rows:
        logger.error("No observations parsed from logs=%s eval_dirs=%s", args.logs, args.eval_dirs)
        return 2
    if not observations:
        logger.warning("No prediction observations; writing trainer_log tables only.")

    import pandas as pd

    datasets = sorted({obs.dataset for obs in observations if obs.dataset != UNKNOWN_DATASET})
    indices = load_annotation_indices(
        datasets,
        dataset_info,
        hf_hub_cache=args.hf_hub_cache,
        dataset_dir=args.dataset_dir,
        annotation_overrides=annotation_overrides,
        gt_field_overrides=gt_field_overrides,
        warnings=warnings,
    )

    tokenizer = load_optional_tokenizer(args.tokenizer)
    grammar = None
    if args.grammar:
        from _grammar import GrammarAnalyzer

        grammar = GrammarAnalyzer(language=args.language)

    lookup_ok = 0
    lookup_fail = 0
    if observations:
        try:
            frame, lookup_ok, lookup_fail = observations_to_frame(
                observations, indices, gt_field_overrides, tokenizer, grammar, warnings, args
            )
        finally:
            if grammar is not None:
                grammar.close()
    else:
        if grammar is not None:
            grammar.close()
        frame = pd.DataFrame()

    if not frame.empty:
        frame = add_previous_prediction_flags(frame)
        frame = add_row_flags(frame)

    trend_frame, matched_strategy, matched_ids = apply_matched_questions(frame, args.matched_questions)
    step_group = ["step", "run_name"] if (not trend_frame.empty and "run_name" in trend_frame.columns) else ["step"]
    dataset_step_group = (
        ["dataset", "step", "run_name"]
        if (not trend_frame.empty and "run_name" in trend_frame.columns)
        else ["dataset", "step"]
    )
    step_summary = summarize_frame(trend_frame, step_group)
    dataset_step = summarize_frame(trend_frame, dataset_step_group)
    dataset_summary = summarize_frame(frame, ["dataset"])
    run_summary = (
        summarize_frame(frame, ["run_name", "step"] if "step" in frame.columns else ["run_name"])
        if (not frame.empty and "run_name" in frame.columns)
        else pd.DataFrame()
    )
    dataset_run_summary = (
        summarize_frame(frame, ["dataset", "run_name"])
        if (not frame.empty and "run_name" in frame.columns)
        else pd.DataFrame()
    )
    trainer_summary = pd.DataFrame(trainer_summary_rows)
    trainer_history = pd.DataFrame(trainer_history_rows)
    step_summary = _attach_trainer_eval_loss(step_summary, trainer_summary)
    run_summary = _attach_trainer_eval_loss(run_summary, trainer_summary)
    if run_summary.empty and not trainer_summary.empty:
        run_summary = trainer_summary.copy()
        if "eval_loss" in run_summary.columns and "trainer_eval_loss" not in run_summary.columns:
            run_summary = run_summary.rename(columns={"eval_loss": "trainer_eval_loss"})
    overlap = overlap_table(question_ids_by_step(frame))
    if not overlap.empty and not frame.empty and "run_name" in frame.columns:
        run_by_step = frame.dropna(subset=["step"]).drop_duplicates(subset=["step"])[["step", "run_name"]]
        overlap = overlap.merge(run_by_step, on="step", how="left")
    trajectories = question_trajectories(frame)
    examples = interesting_examples(trajectories)
    health = health_by_step(step_summary, dataset_step)
    step_flags = step_change_flags(
        step_summary, canonical_drop=args.canonical_drop, think_growth=args.think_growth, warnings=warnings
    )
    corr = correlation_tables(trend_frame)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_frame = None
    loss_summary = None
    if args.compute_loss:
        from _loss import compute_probe_losses, summarize_probe_loss

        if args.checkpoint_root is None:
            raise SystemExit("--compute-loss requires --checkpoint-root")
        loss_frame = compute_probe_losses(
            frame=frame,
            annotation_indices=indices,
            dataset_info=dataset_info,
            checkpoint_root=args.checkpoint_root,
            training_config=args.training_config,
            model_name_or_path=args.model_name_or_path,
            llamafactory_root=args.llamafactory_root,
            loss_batch_size=args.loss_batch_size,
            max_samples_per_dataset=args.loss_max_samples_per_dataset,
            seed=args.seed,
            nearest_checkpoint=args.nearest_checkpoint,
            output_dir=output_dir,
            warnings=warnings,
        )
        if not loss_frame.empty:
            merge_cols = [
                "question_id",
                "dataset",
                "step",
                "probe_loss",
                "probe_token_nll",
                "target_token_count",
                "skip_reason",
            ]
            present = [col for col in merge_cols if col in loss_frame.columns]
            on = ["question_id", "dataset", "step"]
            if "run_name" in loss_frame.columns and "run_name" in frame.columns:
                on.append("run_name")
                if "run_name" not in present:
                    present = ["run_name", *present]
            frame = frame.merge(loss_frame[present], on=on, how="left")
            # Recompute dataset×step so probe_loss_* stats appear.
            trend_frame, matched_strategy, matched_ids = apply_matched_questions(frame, args.matched_questions)
            step_summary = summarize_frame(trend_frame, step_group)
            dataset_step = summarize_frame(trend_frame, dataset_step_group)
            run_summary = (
                summarize_frame(frame, ["run_name", "step"] if "step" in frame.columns else ["run_name"])
                if (not frame.empty and "run_name" in frame.columns)
                else run_summary
            )
            step_summary = _attach_trainer_eval_loss(step_summary, trainer_summary)
            run_summary = _attach_trainer_eval_loss(run_summary, trainer_summary)
            health = health_by_step(step_summary, dataset_step)
        loss_summary = summarize_probe_loss(loss_frame)
        save_table(loss_frame, output_dir / "loss_by_example.csv")
        save_table(loss_summary, output_dir / "loss_by_checkpoint_dataset.csv")

    save_detailed(frame, output_dir)
    save_table(step_summary, output_dir / "step_summary.csv")
    save_table(dataset_step, output_dir / "dataset_step_summary.csv")
    save_table(dataset_summary, output_dir / "dataset_summary.csv")
    save_table(run_summary, output_dir / "run_summary.csv")
    save_table(dataset_run_summary, output_dir / "dataset_run_summary.csv")
    save_table(trainer_summary, output_dir / "trainer_log_summary.csv")
    save_table(trainer_history, output_dir / "trainer_log_history.csv")
    save_table(overlap, output_dir / "question_overlap.csv")
    save_table(trajectories, output_dir / "question_trajectories.csv")
    save_table(health, output_dir / "health_by_step.csv")
    save_table(step_flags, output_dir / "step_flags.csv")
    if not corr["global"].empty:
        save_table(corr["global"], output_dir / "correlations_global.csv")
    if "per_dataset" in corr and not corr["per_dataset"].empty:
        save_table(corr["per_dataset"], output_dir / "correlations_by_dataset.csv")
    for name, table in examples.items():
        save_table(table, output_dir / f"examples_{name}.csv")

    warn_df = __import__("pandas").DataFrame([rec.to_dict() for rec in warnings])
    save_table(warn_df, output_dir / "warnings.csv")

    plots_written: list[str] = []
    if args.plots:
        plots_written = generate_plots(
            output_dir=output_dir,
            step_summary=step_summary,
            dataset_step=dataset_step,
            dataset_summary=dataset_summary,
            overlap=overlap,
            detailed=frame,
            grammar_enabled=args.grammar,
            rolling_window=args.rolling_window,
            run_summary=run_summary,
        )

    steps = sorted({obs.step for obs in observations if obs.step is not None})
    annotation_lookup = {
        "success": lookup_ok,
        "failure": lookup_fail,
        "rate": (lookup_ok / max(lookup_ok + lookup_fail, 1)),
        "files": {name: (str(idx.path) if idx.path else None) for name, idx in indices.items()},
    }
    integrity = integrity_stats(observations, frame, warnings)
    pred_names = [str(run.prediction_path) for run in runs if run.prediction_path is not None]
    teacher_forced_note = any("train" in Path(path).name.lower() for path in pred_names)
    eval_only = (not observations) or ({obs.source_kind for obs in observations} <= {"eval"})
    run_records = (
        trainer_summary.to_dict(orient="records")
        if not trainer_summary.empty
        else [
            {
                "run_name": run.run_name,
                "step": run.step,
                "eval_dir": str(run.eval_dir) if run.eval_dir else None,
                "eval_predictions": str(run.prediction_path) if run.prediction_path else None,
                "trainer_log": str(run.trainer_log_path) if run.trainer_log_path else None,
            }
            for run in runs
        ]
    )

    flag_note = "None."
    if not step_flags.empty and "flags" in step_flags.columns:
        nonempty = step_flags[step_flags["flags"].astype(str).str.len() > 0]
        if not nonempty.empty:
            flag_note = "; ".join(f"step {row.step}: {row.flags}" for row in nonempty.itertuples())

    report = build_report(
        log_paths=pred_names or [str(path) for path in (args.logs or [])],
        n_rows=len(frame),
        datasets=sorted({obs.dataset for obs in observations}),
        steps=steps,
        annotation_lookup=annotation_lookup,
        step_summary=step_summary,
        dataset_summary=dataset_summary,
        overlap=overlap,
        matched_strategy=matched_strategy,
        n_matched=len(matched_ids),
        n_trajectories=int(trajectories["question_id"].nunique()) if not trajectories.empty else 0,
        teacher_forced_note=teacher_forced_note,
        grammar_enabled=args.grammar,
        loss_enabled=args.compute_loss,
        flags_note=flag_note,
        warnings_n=len(warnings),
        run_records=run_records,
        run_summary=run_summary,
        eval_only=eval_only,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    summary = build_analysis_summary(
        {
            "input_logs": pred_names or [str(path) for path in (args.logs or [])],
            "eval_dirs": list(args.eval_dirs or []),
            "runs": run_records,
            "dataset_info": str(args.dataset_info) if args.dataset_info else None,
            "output_dir": str(output_dir),
            "cli": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
            "seed": args.seed,
            "dependency_versions": dependency_versions(),
            "integrity": integrity,
            "annotation_lookup": annotation_lookup,
            "matched_strategy": matched_strategy,
            "n_matched_questions": len(matched_ids),
            "thresholds": {
                "short_think_tokens": args.short_think_tokens,
                "long_think_tokens": args.long_think_tokens,
                "high_repetition": args.high_repetition,
                "high_ngram_repetition": args.high_ngram_repetition,
                "canonical_drop": args.canonical_drop,
                "think_growth": args.think_growth,
                "repetition_score": (
                    "0.4*adjacent_identical_fraction + 0.3*(1-distinct_3) "
                    "+ 0.2*(1-compression_ratio) + 0.1*most_common_token_fraction"
                ),
            },
            "plots": plots_written,
            "teacher_forced_note": teacher_forced_note,
        }
    )
    dump_json(output_dir / "analysis_summary.json", summary)
    logger.info("Wrote analysis to %s", output_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    if args.self_test:
        from _selftest import run_self_tests

        run_self_tests()
        print("self-test: ok")
        return 0
    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
