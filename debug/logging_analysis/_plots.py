"""Matplotlib plots for train-prediction diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def _try_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    fig.clf()


def _rolling(series: pd.Series, window: int) -> pd.Series:
    if window <= 1 or series.notna().sum() < window:
        return series
    return series.rolling(window=window, min_periods=max(1, window // 2)).mean()


def _line_with_band(
    ax,
    frame: pd.DataFrame,
    y_median: str,
    y_p25: str | None,
    y_p75: str | None,
    label: str,
    rolling_window: int,
) -> None:
    if y_median not in frame.columns:
        return
    ordered = frame.sort_values("step")
    x = ordered["step"]
    y = ordered[y_median]
    ax.plot(x, y, marker="o", linewidth=1.2, label=label)
    if rolling_window > 1:
        ax.plot(x, _rolling(y, rolling_window), linestyle="--", linewidth=1.0, alpha=0.8, label=f"{label} (rolling)")
    if y_p25 and y_p75 and y_p25 in ordered.columns and y_p75 in ordered.columns:
        ax.fill_between(x, ordered[y_p25], ordered[y_p75], alpha=0.15)


def plot_metric_by_step(
    step_summary: pd.DataFrame,
    dataset_step: pd.DataFrame,
    *,
    y: str,
    title: str,
    ylabel: str,
    path: Path,
    rolling_window: int = 1,
    use_fraction: bool = False,
) -> None:
    plt = _try_pyplot()
    fig, ax = plt.subplots(figsize=(9, 5))
    if not step_summary.empty and y in step_summary.columns:
        _line_with_band(ax, step_summary, y, None, None, "all", rolling_window)
    if dataset_step is not None and not dataset_step.empty and y in dataset_step.columns:
        for dataset, sub in dataset_step.groupby("dataset"):
            _line_with_band(ax, sub, y, None, None, str(dataset), rolling_window)
    ax.set_xlabel("global training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if use_fraction:
        ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save(fig, path)
    plt.close(fig)


def plot_length_with_iqr(
    step_summary: pd.DataFrame,
    *,
    median_col: str,
    p25_col: str,
    p75_col: str,
    title: str,
    ylabel: str,
    path: Path,
    rolling_window: int = 1,
) -> None:
    plt = _try_pyplot()
    fig, ax = plt.subplots(figsize=(9, 5))
    _line_with_band(ax, step_summary, median_col, p25_col, p75_col, "median + IQR", rolling_window)
    ax.set_xlabel("global training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save(fig, path)
    plt.close(fig)


def plot_samples_per_step(dataset_step: pd.DataFrame, path: Path) -> None:
    if dataset_step.empty or "n" not in dataset_step.columns:
        return
    plt = _try_pyplot()
    pivot = dataset_step.pivot_table(index="step", columns="dataset", values="n", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.sort_index().plot(kind="bar", stacked=True, ax=ax, width=0.9)
    ax.set_xlabel("global training step")
    ax.set_ylabel("number of predictions")
    ax.set_title("Predictions per training step (dataset breakdown)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate(rotation=45)
    _save(fig, path)
    plt.close(fig)


def plot_overlap(overlap: pd.DataFrame, path: Path) -> None:
    frame = overlap[overlap["step"].map(lambda value: value != "ALL")].copy() if "step" in overlap.columns else overlap
    if frame.empty or "jaccard" not in frame.columns:
        return
    plt = _try_pyplot()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame["step"], frame["jaccard"], marker="o")
    ax.set_xlabel("global training step")
    ax.set_ylabel("Jaccard overlap with previous step")
    ax.set_title("Question-set overlap across consecutive steps")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    _save(fig, path)
    plt.close(fig)


def plot_dataset_comparison(dataset_summary: pd.DataFrame, path: Path) -> None:
    if dataset_summary.empty:
        return
    plt = _try_pyplot()
    metrics = [
        ("canonical_format_fraction", "canonical format"),
        ("usable_format_fraction", "usable format"),
        ("normalized_exact_match_fraction", "normalized EM"),
        ("repetition_score_median", "median repetition"),
        ("think_token_count_median", "median think tokens"),
    ]
    present = [(col, label) for col, label in metrics if col in dataset_summary.columns]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 4), squeeze=False)
    for ax, (col, label) in zip(axes[0], present):
        ax.bar(dataset_summary["dataset"].astype(str), dataset_summary[col])
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Dataset comparison (all steps pooled)")
    _save(fig, path)
    plt.close(fig)


def plot_loss(dataset_step: pd.DataFrame, path: Path) -> None:
    if dataset_step.empty or "probe_loss_mean" not in dataset_step.columns:
        return
    plot_metric_by_step(
        pd.DataFrame(),
        dataset_step,
        y="probe_loss_mean",
        title="Checkpoint probe loss (reference-target CE) vs step",
        ylabel="mean probe loss",
        path=path,
    )


def plot_loss_vs_generation(frame: pd.DataFrame, path: Path) -> None:
    if "probe_loss" not in frame.columns or frame["probe_loss"].dropna().empty:
        return
    plt = _try_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(frame["probe_loss"], frame.get("repetition_score"), alpha=0.4)
    axes[0].set_xlabel("probe loss")
    axes[0].set_ylabel("repetition_score")
    axes[0].set_title("Loss vs repetition")
    axes[1].scatter(frame["probe_loss"], frame.get("tag_presence_score"), alpha=0.4)
    axes[1].set_xlabel("probe loss")
    axes[1].set_ylabel("tag_presence_score")
    axes[1].set_title("Loss vs tag presence")
    _save(fig, path)
    plt.close(fig)


def generate_plots(
    *,
    output_dir: Path,
    step_summary: pd.DataFrame,
    dataset_step: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    overlap: pd.DataFrame,
    detailed: pd.DataFrame,
    grammar_enabled: bool,
    rolling_window: int,
) -> list[str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def _record(name: str) -> Path:
        path = plots_dir / name
        written.append(str(path))
        return path

    try:
        plot_metric_by_step(
            step_summary,
            dataset_step,
            y="canonical_format_fraction",
            title="Canonical <think>/<answer> format rate vs step",
            ylabel="canonical_format fraction",
            path=_record("tag_canonical_format.png"),
            rolling_window=rolling_window,
            use_fraction=True,
        )
        plot_metric_by_step(
            step_summary,
            dataset_step,
            y="tag_pair_score_mean",
            title="Tag-pair score vs step",
            ylabel="mean tag_pair_score",
            path=_record("tag_pair_score.png"),
            rolling_window=rolling_window,
        )
        if "think_token_count_median" in step_summary.columns:
            plot_length_with_iqr(
                step_summary,
                median_col="think_token_count_median",
                p25_col="think_token_count_p25",
                p75_col="think_token_count_p75",
                title="Think-section length vs step",
                ylabel="think tokens",
                path=_record("think_length.png"),
                rolling_window=rolling_window,
            )
        if "answer_token_count_median" in step_summary.columns:
            plot_length_with_iqr(
                step_summary,
                median_col="answer_token_count_median",
                p25_col="answer_token_count_p25",
                p75_col="answer_token_count_p75",
                title="Answer-section length vs step",
                ylabel="answer tokens",
                path=_record("answer_length.png"),
                rolling_window=rolling_window,
            )
        if "token_count_median" in step_summary.columns:
            plot_length_with_iqr(
                step_summary,
                median_col="token_count_median",
                p25_col="token_count_p25",
                p75_col="token_count_p75",
                title="Total response length vs step",
                ylabel="tokens",
                path=_record("total_length.png"),
                rolling_window=rolling_window,
            )
        plot_metric_by_step(
            step_summary,
            dataset_step,
            y="repetition_score_median",
            title="Composite repetition_score vs step (not a quality score)",
            ylabel="median repetition_score",
            path=_record("repetition_score.png"),
            rolling_window=rolling_window,
        )
        plot_metric_by_step(
            step_summary,
            dataset_step,
            y="ngram3_repeated_fraction_median",
            title="Repeated trigram fraction vs step",
            ylabel="median 1 - unique_3 / total_3",
            path=_record("repetition_trigram.png"),
            rolling_window=rolling_window,
        )
        plot_metric_by_step(
            step_summary,
            dataset_step,
            y="adjacent_identical_fraction_median",
            title="Adjacent identical-token fraction vs step",
            ylabel="median adjacent identical fraction",
            path=_record("repetition_adjacent.png"),
            rolling_window=rolling_window,
        )
        if grammar_enabled:
            plot_metric_by_step(
                step_summary,
                dataset_step,
                y="grammar_issues_per_100_words_median",
                title="LanguageTool issues per 100 words vs step (diagnostic only)",
                ylabel="median issues / 100 words",
                path=_record("grammar.png"),
                rolling_window=rolling_window,
            )
        plot_metric_by_step(
            step_summary,
            dataset_step,
            y="normalized_exact_match_fraction",
            title="Normalized exact match vs reference answer (surface only)",
            ylabel="normalized EM fraction",
            path=_record("surface_match.png"),
            rolling_window=rolling_window,
            use_fraction=True,
        )
        plot_samples_per_step(dataset_step, _record("samples_per_step.png"))
        plot_overlap(overlap, _record("question_overlap.png"))
        plot_dataset_comparison(dataset_summary, _record("dataset_comparison.png"))
        plot_loss(dataset_step, _record("probe_loss.png"))
        plot_loss_vs_generation(detailed, _record("loss_vs_generation.png"))
    except Exception:
        logger.exception("Plot generation failed")
    return written
