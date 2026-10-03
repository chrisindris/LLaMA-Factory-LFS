"""Deterministic markdown + JSON summaries. No AI interpretation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _first_last_step(step_summary: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    if step_summary.empty or "step" not in step_summary.columns:
        return None, None
    ordered = step_summary.sort_values("step")
    return ordered.iloc[0], ordered.iloc[-1]


def _run_metric_table(run_summary: pd.DataFrame) -> list[str]:
    if run_summary.empty or "run_name" not in run_summary.columns:
        return []
    cols = [
        ("run_name", "run"),
        ("step", "step"),
        ("n", "n"),
        ("canonical_format_fraction", "canonical"),
        ("usable_format_fraction", "usable"),
        ("normalized_exact_match_fraction", "norm EM"),
        ("repetition_score_median", "repetition"),
        ("think_token_count_median", "think tok"),
        ("trainer_eval_loss", "eval_loss"),
    ]
    present = [(col, label) for col, label in cols if col in run_summary.columns]
    if not present:
        return []
    ordered = run_summary.sort_values("step" if "step" in run_summary.columns else "run_name")
    header = "| " + " | ".join(label for _col, label in present) + " |"
    sep = "| " + " | ".join("---" for _ in present) + " |"
    lines = ["", header, sep]
    for _, row in ordered.iterrows():
        cells = []
        for col, _label in present:
            value = row.get(col)
            if col == "run_name":
                cells.append(f"`{value}`")
            elif col in {"canonical_format_fraction", "usable_format_fraction", "normalized_exact_match_fraction"}:
                cells.append(_pct(value))
            elif col in {"repetition_score_median", "trainer_eval_loss"}:
                cells.append(_num(value, 3))
            elif col == "think_token_count_median":
                cells.append(_num(value, 1))
            else:
                cells.append(
                    str(value) if value is not None and not (isinstance(value, float) and value != value) else "n/a"
                )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def build_report(
    *,
    log_paths: list[str],
    n_rows: int,
    datasets: list[str],
    steps: list[Any],
    annotation_lookup: dict[str, Any],
    step_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    overlap: pd.DataFrame,
    matched_strategy: str,
    n_matched: int,
    n_trajectories: int,
    teacher_forced_note: bool,
    grammar_enabled: bool,
    loss_enabled: bool,
    flags_note: str,
    warnings_n: int,
    run_records: list[dict[str, Any]] | None = None,
    run_summary: pd.DataFrame | None = None,
    eval_only: bool = False,
) -> str:
    first, last = _first_last_step(step_summary)
    n_runs = len(run_records or [])
    if eval_only and n_runs > 1:
        title = "# Eval prediction comparison"
    elif eval_only:
        title = "# Eval prediction log analysis"
    else:
        title = "# Train prediction log analysis"
    lines = [
        title,
        "",
        "This report is a **deterministic statistical summary**. It is not a judge of",
        "reasoning quality, semantic correctness, or training success.",
        "",
        "## Inputs",
        "",
    ]
    if run_records:
        for rec in run_records:
            name = rec.get("run_name") or rec.get("name")
            step = rec.get("step")
            pred = rec.get("eval_predictions") or rec.get("prediction_path")
            trainer = rec.get("trainer_log")
            loss = rec.get("eval_loss")
            folder = rec.get("eval_dir")
            bits = [f"**{name}**"]
            if step is not None:
                bits.append(f"step {step}")
            if loss is not None:
                bits.append(f"eval_loss={_num(loss, 4)}")
            lines.append(f"- {' · '.join(bits)}")
            if folder:
                lines.append(f"  - dir: `{folder}`")
            if pred:
                lines.append(f"  - predictions: `{pred}`")
            if trainer:
                lines.append(f"  - trainer_log: `{trainer}`")
    else:
        for path in log_paths:
            lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            f"- Predictions: **{n_rows}**",
            f"- Datasets: {', '.join(datasets) if datasets else 'none'}",
            f"- Steps: {steps[0] if steps else 'n/a'} … {steps[-1] if steps else 'n/a'} ({len(steps)} distinct)",
            f"- Annotation lookup: {json.dumps(annotation_lookup)}",
            f"- Grammar: {'enabled' if grammar_enabled else 'disabled'}",
            f"- Checkpoint probe loss: {'enabled' if loss_enabled else 'disabled'}",
            "",
        ]
    )
    if n_runs > 1:
        lines.extend(
            [
                "Each eval save folder is a separate run. Assigned steps keep them from being",
                "pooled. `--matched-questions` intersects IDs across those runs.",
                "",
            ]
        )
        lines.extend(_run_metric_table(run_summary if run_summary is not None else pd.DataFrame()))
    if teacher_forced_note:
        lines.extend(
            [
                "## Teacher-forced dumps",
                "",
                "The training YAML in this repo defaults to `train_prediction_mode: teacher_forced`.",
                "Those texts are **argmax next-token decodes of response positions**, not",
                "`model.generate` outputs. Tag / repetition / surface-match numbers still",
                "describe the dumped strings, but they are **not** free-form generation quality.",
                "",
            ]
        )
    lines.extend(
        ["## Sampling", "", f"Matched-question strategy: `{matched_strategy}` (matched IDs: {n_matched}).", ""]
    )
    if not overlap.empty and "jaccard" in overlap.columns:
        consec = overlap[overlap["step"].map(lambda value: value != "ALL")]
        if not consec.empty:
            mean_j = pd.to_numeric(consec["jaccard"], errors="coerce").mean()
            lines.append(f"Mean consecutive-step question-ID Jaccard: {_num(mean_j, 3)}.")
        all_row = overlap[overlap["step"].map(lambda value: value == "ALL")]
        if not all_row.empty:
            lines.append(
                f"Intersection of question IDs across **all** steps: {int(all_row.iloc[0]['intersection'])} "
                f"/ union {int(all_row.iloc[0]['union'])}."
            )
        lines.append("")
        lines.append("If Jaccard is ~0, step trends mix different questions. That can masquerade as learning.")
        lines.append("")
    if n_trajectories == 0:
        lines.extend(
            [
                "No question ID appears at more than one step, so per-question trajectories are empty.",
                "",
            ]
        )

    lines.extend(["## Tag adherence", ""])
    if first is not None and last is not None:
        col = "canonical_format_fraction"
        if col in first.index:
            lines.append(
                f"Canonical-format rate {_pct(first[col])} at step {first['step']} "
                f"vs {_pct(last[col])} at step {last['step']}."
            )
        if "tag_pair_score_mean" in first.index:
            lines.append(
                f"Mean tag-pair score {_num(first['tag_pair_score_mean'])} at step {first['step']} "
                f"vs {_num(last['tag_pair_score_mean'])} at step {last['step']}."
            )
        if "usable_format_fraction" in first.index:
            lines.append(
                f"Usable-format rate {_pct(first['usable_format_fraction'])} at step {first['step']} "
                f"vs {_pct(last['usable_format_fraction'])} at step {last['step']}."
            )
        lines.append("")
    if not dataset_summary.empty and "canonical_format_fraction" in dataset_summary.columns:
        lines.append("Canonical-format rate by dataset (all steps pooled):")
        lines.append("")
        for _, row in dataset_summary.iterrows():
            lines.append(f"- `{row.get('dataset')}`: {_pct(row.get('canonical_format_fraction'))} (n={row.get('n')})")
        lines.append("")

    lines.extend(["## Length", ""])
    if first is not None and last is not None:
        for label, col in (
            ("median think tokens", "think_token_count_median"),
            ("median answer tokens", "answer_token_count_median"),
            ("median total tokens", "token_count_median"),
        ):
            if col in first.index:
                lines.append(
                    f"{label}: {_num(first[col], 1)} at step {first['step']} vs {_num(last[col], 1)} at step {last['step']}."
                )
        if "empty_final_answer_fraction" in first.index:
            lines.append(
                f"Empty-answer rate: {_pct(first['empty_final_answer_fraction'])} at step {first['step']} "
                f"vs {_pct(last['empty_final_answer_fraction'])} at step {last['step']}."
            )
        lines.append("")

    lines.extend(["## Repetition", ""])
    if first is not None and last is not None:
        for label, col in (
            ("median repetition_score", "repetition_score_median"),
            ("median trigram repetition fraction", "ngram3_repeated_fraction_median"),
            ("median adjacent-identical fraction", "adjacent_identical_fraction_median"),
        ):
            if col in first.index:
                lines.append(
                    f"{label}: {_num(first[col], 3)} at step {first['step']} vs {_num(last[col], 3)} at step {last['step']}."
                )
        lines.append("")
        lines.append("`repetition_score` is an uncalibrated weighted mix of adjacent-token identity,")
        lines.append("trigram distinct-n, zlib compression, and unigram concentration. Components are in the tables.")
        lines.append("")

    if grammar_enabled:
        lines.extend(["## Grammar (LanguageTool diagnostic, not accuracy)", ""])
        if first is not None and last is not None and "grammar_issues_per_100_words_median" in first.index:
            lines.append(
                f"Median issues / 100 words: {_num(first['grammar_issues_per_100_words_median'])} at step {first['step']} "
                f"vs {_num(last['grammar_issues_per_100_words_median'])} at step {last['step']}."
            )
            lines.append("")
        lines.append("Short scene answers such as `left of the sofa` can trigger misleading grammar warnings.")
        lines.append("")

    lines.extend(["## Surface match vs reference (not semantic correctness)", ""])
    if first is not None and last is not None and "normalized_exact_match_fraction" in first.index:
        lines.append(
            f"Normalized exact-match rate: {_pct(first['normalized_exact_match_fraction'])} at step {first['step']} "
            f"vs {_pct(last['normalized_exact_match_fraction'])} at step {last['step']}."
        )
        lines.append("")

    if loss_enabled:
        lines.extend(
            [
                "## Checkpoint probe loss",
                "",
                "This is **reference-target teacher-forced cross-entropy** on recovered annotations,",
                "not the trainer's original batch loss at that step.",
                "",
            ]
        )

    lines.extend(
        [
            "## Combined views (read the tables/plots; no causal claims)",
            "",
            "- Formatting up + repetition up: the model may learn tags while degenerating.",
            "- Probe loss down + think tokens collapsing: optimization vs generation mismatch.",
            "- Think tokens up + normalized EM flat: longer reasoning without lexical match gains.",
            "- Per-dataset rows: one mix member can drive a global trend.",
            "",
            "## Flags",
            "",
            flags_note or "No step-level flags.",
            "",
            f"Warnings written: {warnings_n} (see `warnings.csv`).",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_analysis_summary(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload.setdefault("timestamp_utc", datetime.now(UTC).isoformat())
    return payload


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
