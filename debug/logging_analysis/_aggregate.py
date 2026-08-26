"""Step / dataset aggregation, overlap, trajectories, flags, correlations."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import pandas as pd
from _logparse import WarningRecord


logger = logging.getLogger(__name__)


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        if pd.isna(value):
            return math.nan
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _signed_delta(current: Any, previous: Any) -> float:
    cur = _as_float(current)
    old = _as_float(previous)
    if cur != cur or old != old:
        return math.nan
    return cur - old


BOOL_METRICS = (
    "canonical_format",
    "usable_format",
    "has_complete_think_pair",
    "has_complete_answer_pair",
    "has_both_tag_pairs",
    "think_is_empty",
    "answer_is_empty",
    "malformed_tags",
    "prediction_empty",
    "extremely_short_reasoning",
    "extremely_long_reasoning",
    "empty_final_answer",
    "length_explosion",
    "length_collapse",
    "very_high_token_repetition",
    "very_high_ngram_repetition",
    "exact_match",
    "case_insensitive_exact_match",
    "normalized_exact_match",
    "annotation_found",
    "answer_inside_think",
    "think_after_answer",
    "text_after_answer_nonempty",
    "same_as_previous_prediction",
)

NUMERIC_METRICS = (
    "tag_presence_score",
    "tag_pair_score",
    "token_count",
    "think_token_count",
    "answer_token_count",
    "char_count",
    "word_count",
    "think_char_count",
    "answer_char_count",
    "think_word_count",
    "answer_word_count",
    "think_token_fraction",
    "answer_token_fraction",
    "think_over_answer_tokens",
    "adjacent_identical_fraction",
    "max_identical_token_run",
    "ngram3_repeated_fraction",
    "distinct_1",
    "distinct_2",
    "distinct_3",
    "distinct_4",
    "repetition_score",
    "think_repetition_score",
    "answer_repetition_score",
    "compression_ratio",
    "most_common_token_fraction",
    "think_type_token_ratio",
    "think_unique_token_count",
    "token_f1",
    "jaccard",
    "normalized_levenshtein",
    "pred_minus_gt_think_tokens",
    "pred_minus_gt_answer_tokens",
    "pred_over_gt_think_tokens",
    "pred_over_gt_answer_tokens",
    "grammar_issue_count",
    "grammar_issues_per_100_words",
    "think_grammar_issues_per_100_words",
    "answer_grammar_issues_per_100_words",
    "probe_loss",
    "probe_token_nll",
)

HEALTH_NUMERIC = (
    "canonical_format",
    "usable_format",
    "think_token_count",
    "answer_token_count",
    "token_count",
    "empty_final_answer",
    "repetition_score",
    "grammar_issues_per_100_words",
    "normalized_exact_match",
    "n",
)


def _present_columns(frame: pd.DataFrame, names: Sequence[str]) -> list[str]:
    return [name for name in names if name in frame.columns]


def _bool_stats(series: pd.Series) -> dict[str, float]:
    valid = series.dropna()
    if valid.empty:
        return {"count": 0.0, "fraction": math.nan}
    values = valid.astype(bool)
    return {"count": float(values.sum()), "fraction": float(values.mean())}


def _numeric_stats(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "n": 0.0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "p25": math.nan,
            "p75": math.nan,
        }
    return {
        "n": float(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
    }


def summarize_frame(frame: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    bool_cols = _present_columns(frame, BOOL_METRICS)
    num_cols = _present_columns(frame, NUMERIC_METRICS)
    if group_cols:
        grouped = frame.groupby(group_cols, dropna=False, sort=True)
        rows = []
        for key, sub in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            row: dict[str, Any] = dict(zip(group_cols, key))
            row["n"] = int(len(sub))
            row["n_unique_questions"] = int(sub["question_id"].nunique()) if "question_id" in sub.columns else row["n"]
            for col in bool_cols:
                stats = _bool_stats(sub[col])
                row[f"{col}_count"] = stats["count"]
                row[f"{col}_fraction"] = stats["fraction"]
            for col in num_cols:
                stats = _numeric_stats(sub[col])
                for stat_name, value in stats.items():
                    row[f"{col}_{stat_name}"] = value
            rows.append(row)
        return pd.DataFrame(rows)
    row: dict[str, Any] = {"n": int(len(frame))}
    if "question_id" in frame.columns:
        row["n_unique_questions"] = int(frame["question_id"].nunique())
    for col in bool_cols:
        stats = _bool_stats(frame[col])
        row[f"{col}_count"] = stats["count"]
        row[f"{col}_fraction"] = stats["fraction"]
    for col in num_cols:
        stats = _numeric_stats(frame[col])
        for stat_name, value in stats.items():
            row[f"{col}_{stat_name}"] = value
    return pd.DataFrame([row])


def question_ids_by_step(frame: pd.DataFrame) -> dict[Any, set[str]]:
    out: dict[Any, set[str]] = {}
    if frame.empty or "step" not in frame.columns:
        return out
    for step, sub in frame.groupby("step", dropna=False):
        out[step] = set(sub["question_id"].astype(str))
    return out


def overlap_table(by_step: dict[Any, set[str]]) -> pd.DataFrame:
    steps = sorted(by_step, key=lambda value: (value is None, value))
    rows = []
    for i, step in enumerate(steps):
        current = by_step[step]
        row = {
            "step": step,
            "n_questions": len(current),
        }
        if i == 0:
            row.update(
                {
                    "prev_step": math.nan,
                    "intersection": math.nan,
                    "union": math.nan,
                    "jaccard": math.nan,
                }
            )
        else:
            prev_step = steps[i - 1]
            prev = by_step[prev_step]
            inter = current & prev
            union = current | prev
            row.update(
                {
                    "prev_step": prev_step,
                    "intersection": len(inter),
                    "union": len(union),
                    "jaccard": (len(inter) / len(union)) if union else math.nan,
                }
            )
        rows.append(row)
    if len(steps) >= 2:
        all_inter = set.intersection(*by_step.values()) if by_step else set()
        all_union = set.union(*by_step.values()) if by_step else set()
        rows.append(
            {
                "step": "ALL",
                "n_questions": len(all_union),
                "prev_step": "ALL_STEPS",
                "intersection": len(all_inter),
                "union": len(all_union),
                "jaccard": (len(all_inter) / len(all_union)) if all_union else math.nan,
            }
        )
    return pd.DataFrame(rows)


def matched_question_ids(by_step: dict[Any, set[str]]) -> set[str]:
    sets = [ids for ids in by_step.values() if ids is not None]
    if not sets:
        return set()
    return set.intersection(*sets)


def apply_matched_questions(
    frame: pd.DataFrame,
    enabled: bool,
) -> tuple[pd.DataFrame, str, set[str]]:
    """Return (frame_for_step_trends, strategy, matched_ids).

    If every plotted step shares a non-empty ID set, restrict to that set.
    Otherwise keep the full frame and document pairwise consecutive overlap.
    """
    if not enabled:
        return frame, "unmatched_all_observations", set()
    by_step = question_ids_by_step(frame)
    matched = matched_question_ids(by_step)
    if matched:
        filtered = frame[frame["question_id"].astype(str).isin(matched)].copy()
        return filtered, "intersection_of_all_steps", matched
    return frame, "pairwise_consecutive_only_no_global_intersection", set()


def add_previous_prediction_flags(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = frame.sort_values(["question_id", "step"], kind="mergesort").copy()
    prev_pred = ordered.groupby("question_id")["prediction"].shift(1)
    prev_think = ordered.groupby("question_id")["think_text"].shift(1) if "think_text" in ordered.columns else None
    prev_answer = ordered.groupby("question_id")["answer_text"].shift(1) if "answer_text" in ordered.columns else None
    ordered["same_as_previous_prediction"] = prev_pred.notna() & (ordered["prediction"] == prev_pred)
    if prev_think is not None:
        a = ordered["think_text"].fillna("").map(lambda value: " ".join(str(value).split()).casefold())
        b = prev_think.fillna("").map(lambda value: " ".join(str(value).split()).casefold())
        ordered["same_as_previous_think"] = prev_think.notna() & (a == b)
    if prev_answer is not None:
        a = ordered["answer_text"].fillna("").map(lambda value: " ".join(str(value).split()).casefold())
        b = prev_answer.fillna("").map(lambda value: " ".join(str(value).split()).casefold())
        ordered["same_as_previous_answer"] = prev_answer.notna() & (a == b)
    unique_counts = ordered.groupby("question_id")["prediction"].transform("nunique")
    ordered["n_unique_predictions_for_question"] = unique_counts
    frac = ordered.groupby("question_id")["same_as_previous_prediction"].transform(
        lambda series: float(series.fillna(False).mean()) if series.notna().any() else math.nan
    )
    ordered["fraction_of_observations_identical_to_previous"] = frac
    return ordered


def question_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    ordered = frame.sort_values(["question_id", "step"], kind="mergesort")
    rows = []
    metric_pairs = [
        ("tag_presence_score", "tag_score_delta"),
        ("think_token_count", "think_tokens_delta"),
        ("answer_token_count", "answer_tokens_delta"),
        ("repetition_score", "repetition_delta"),
        ("grammar_issues_per_100_words", "grammar_delta"),
        ("normalized_exact_match", "surface_match_delta"),
    ]
    for question_id, sub in ordered.groupby("question_id", sort=False):
        if len(sub) < 2:
            continue
        prev = None
        for _, row in sub.iterrows():
            record = {
                "question_id": question_id,
                "dataset": row.get("dataset"),
                "step": row.get("step"),
                "prediction": row.get("prediction"),
                "think_text": row.get("think_text"),
                "answer_text": row.get("answer_text"),
                "canonical_format": row.get("canonical_format"),
                "repetition_score": row.get("repetition_score"),
                "think_token_count": row.get("think_token_count"),
                "answer_token_count": row.get("answer_token_count"),
                "normalized_exact_match": row.get("normalized_exact_match"),
            }
            if prev is not None:
                record["prev_step"] = prev.get("step")
                for src, dest in metric_pairs:
                    if src in row.index and src in prev.index:
                        record[dest] = _signed_delta(row[src], prev[src])
                    else:
                        record[dest] = math.nan
                record["same_as_previous_prediction"] = row.get("prediction") == prev.get("prediction")
            else:
                record["prev_step"] = pd.NA
                for _src, dest in metric_pairs:
                    record[dest] = math.nan
                record["same_as_previous_prediction"] = pd.NA
            rows.append(record)
            prev = row
    return pd.DataFrame(rows)


def interesting_examples(trajectories: pd.DataFrame, k: int = 5) -> dict[str, pd.DataFrame]:
    if trajectories.empty:
        return {}
    deltas = (
        trajectories.dropna(subset=["prev_step"], how="any") if "prev_step" in trajectories.columns else trajectories
    )
    picks: dict[str, pd.DataFrame] = {}

    def _head_sorted(frame: pd.DataFrame, column: str, ascending: bool) -> pd.DataFrame:
        if column not in frame.columns:
            return pd.DataFrame()
        ordered = frame.sort_values(column, ascending=ascending, na_position="last")
        return ordered.head(k)

    picks["biggest_tag_improvement"] = _head_sorted(deltas, "tag_score_delta", False)
    picks["biggest_repetition_increase"] = _head_sorted(deltas, "repetition_delta", False)
    picks["largest_reasoning_explosion"] = _head_sorted(deltas, "think_tokens_delta", False)
    picks["largest_reasoning_collapse"] = _head_sorted(deltas, "think_tokens_delta", True)
    if "answer_token_count" in deltas.columns:
        empty_answer = deltas[(deltas["answer_token_count"].fillna(0) == 0)]
        picks["answer_became_empty"] = empty_answer.head(k)
    picks["largest_grammar_deterioration"] = _head_sorted(deltas, "grammar_delta", False)
    picks["largest_surface_match_improvement"] = _head_sorted(deltas, "surface_match_delta", False)
    return picks


def health_by_step(step_summary: pd.DataFrame, dataset_step: pd.DataFrame | None = None) -> pd.DataFrame:
    if step_summary.empty:
        return step_summary
    mapping = {
        "canonical_format_fraction": "canonical_format_pct",
        "think_token_count_median": "median_think_tokens",
        "answer_token_count_median": "median_answer_tokens",
        "empty_final_answer_fraction": "empty_answer_pct",
        "repetition_score_median": "repetition_metric",
        "grammar_issues_per_100_words_median": "grammar_issues_per_100_words",
        "normalized_exact_match_fraction": "normalized_exact_match_pct",
    }
    frame = step_summary.copy()
    out = pd.DataFrame()
    if "step" in frame.columns:
        out["step"] = frame["step"]
    out["n"] = frame["n"] if "n" in frame.columns else pd.NA
    for src, dest in mapping.items():
        if src in frame.columns:
            series = frame[src]
            if dest.endswith("_pct"):
                out[dest] = series * 100.0
            else:
                out[dest] = series
    if dataset_step is not None and not dataset_step.empty and "probe_loss_mean" in dataset_step.columns:
        loss_wide = dataset_step.pivot_table(
            index="step", columns="dataset", values="probe_loss_mean", aggfunc="first"
        )
        loss_wide.columns = [f"{col}_loss" for col in loss_wide.columns]
        out = out.merge(loss_wide.reset_index(), on="step", how="left")
    return out


def spearman_pearson(frame: pd.DataFrame, x_col: str, y_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    if x_col not in frame.columns:
        return pd.DataFrame()
    x = pd.to_numeric(frame[x_col], errors="coerce")
    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        spearmanr = None
        pearsonr = None
    for col in y_cols:
        if col not in frame.columns:
            continue
        y = pd.to_numeric(frame[col], errors="coerce")
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        row = {"metric": col, "n": n, "spearman": math.nan, "pearson": math.nan}
        if n >= 3:
            xv = x[mask].to_numpy()
            yv = y[mask].to_numpy()
            if spearmanr is not None:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        rho, _p = spearmanr(xv, yv)
                        row["spearman"] = float(rho)
                    except Exception:
                        pass
                    try:
                        r, _p = pearsonr(xv, yv)
                        row["pearson"] = float(r)
                    except Exception:
                        pass
            else:
                row["spearman"] = float(pd.Series(xv).corr(pd.Series(yv), method="spearman"))
                row["pearson"] = float(pd.Series(xv).corr(pd.Series(yv), method="pearson"))
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    y_cols = [
        "tag_presence_score",
        "tag_pair_score",
        "canonical_format",
        "think_token_count",
        "answer_token_count",
        "repetition_score",
        "grammar_issues_per_100_words",
        "normalized_exact_match",
        "probe_loss",
    ]
    tables = {"global": spearman_pearson(frame, "step", y_cols)}
    if "dataset" in frame.columns:
        parts = []
        for dataset, sub in frame.groupby("dataset"):
            table = spearman_pearson(sub, "step", y_cols)
            table.insert(0, "dataset", dataset)
            parts.append(table)
        tables["per_dataset"] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return tables


def add_row_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    flags = []
    for _, row in out.iterrows():
        names = []
        if row.get("prediction_empty"):
            names.append("prediction_empty")
        if not row.get("has_think_open") or not row.get("has_think_close"):
            names.append("missing_think_tags")
        if not row.get("has_answer_open") or not row.get("has_answer_close"):
            names.append("missing_answer_tags")
        if row.get("malformed_tags"):
            names.append("malformed_tags")
        if row.get("think_is_empty"):
            names.append("empty_think")
        if row.get("answer_is_empty"):
            names.append("empty_answer")
        if row.get("very_high_token_repetition"):
            names.append("very_high_token_repetition")
        if row.get("very_high_ngram_repetition"):
            names.append("very_high_ngram_repetition")
        if row.get("length_explosion"):
            names.append("length_explosion")
        if row.get("length_collapse"):
            names.append("length_collapse")
        if row.get("same_as_previous_prediction"):
            names.append("prediction_identical_to_previous_step")
        flags.append("|".join(names))
    out["flags"] = flags
    return out


def step_change_flags(
    step_summary: pd.DataFrame,
    *,
    canonical_drop: float = 0.20,
    think_growth: float = 3.0,
    warnings: list[WarningRecord] | None = None,
) -> pd.DataFrame:
    if step_summary.empty or "step" not in step_summary.columns:
        return pd.DataFrame()
    ordered = step_summary.sort_values("step").copy()
    rows = []
    prev = None
    for _, row in ordered.iterrows():
        record = {"step": row["step"], "flags": []}
        if prev is not None:
            cur_c = row.get("canonical_format_fraction")
            prev_c = prev.get("canonical_format_fraction")
            if pd.notna(cur_c) and pd.notna(prev_c) and (prev_c - cur_c) >= canonical_drop:
                record["flags"].append(f"canonical_format_rate_drop_{(prev_c - cur_c) * 100:.1f}pp")
            cur_t = row.get("think_token_count_median")
            prev_t = prev.get("think_token_count_median")
            if pd.notna(cur_t) and pd.notna(prev_t) and prev_t > 0 and cur_t / prev_t >= think_growth:
                record["flags"].append(f"median_think_tokens_x{cur_t / prev_t:.1f}")
            cur_r = row.get("repetition_score_median")
            prev_r = prev.get("repetition_score_median")
            if pd.notna(cur_r) and pd.notna(prev_r) and (cur_r - prev_r) >= 0.15:
                record["flags"].append("repetition_spike")
            cur_g = row.get("grammar_issues_per_100_words_median")
            prev_g = prev.get("grammar_issues_per_100_words_median")
            if pd.notna(cur_g) and pd.notna(prev_g) and prev_g >= 0 and (cur_g - prev_g) >= 5:
                record["flags"].append("grammar_spike")
        record["flags"] = "|".join(record["flags"])
        if record["flags"] and warnings is not None:
            warnings.append(
                WarningRecord(
                    code="step_flag",
                    message=record["flags"],
                    step=int(row["step"]) if pd.notna(row["step"]) else None,
                )
            )
        rows.append(record)
        prev = row
    return pd.DataFrame(rows)
