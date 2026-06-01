"""Helpers for aggregating multi-seed RL evaluation results."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import re
import statistics
from typing import Any


BOT_ORDER: tuple[str, ...] = (
    "RandomBot",
    "StimBot",
    "DiscardLowestChipBot",
    "LookaheadDiscardBot",
    "PrunedSampledLookaheadBot",
    "RLQBot",
)
SEED_FILE_PATTERN = re.compile(r"ante_(?P<max_ante>\d+)_comparison_seed_(?P<seed>\d+)\.json$")
PLOT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("mean_win_rate_pct", "Win Rate (%)", "win_rate_summary.png"),
    ("mean_avg_rounds", "Average Rounds Passed", "avg_rounds_summary.png"),
    ("mean_avg_final_chips", "Average Final Chips", "avg_final_chips_summary.png"),
)


def default_seed_result_paths(repo_root: Path) -> dict[str, list[Path]]:
    """Return ante-grouped seed-result JSON files from results/rl_eval/, sorted."""

    eval_dir = repo_root / "results" / "rl_eval"
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for result_path in sorted(eval_dir.glob("ante_*_comparison_seed_*.json")):
        ante_label, _ = parse_seed_result_metadata(result_path)
        grouped_paths[ante_label].append(result_path)
    return dict(grouped_paths)


def parse_seed_result_metadata(result_path: Path) -> tuple[str, int]:
    """Extract the ante label and seed index from one saved comparison path."""

    match = SEED_FILE_PATTERN.search(result_path.name)
    if match is None:
        raise ValueError(
            f"Could not infer ante/seed from {result_path}. Expected 'ante_<N>_comparison_seed_<n>.json'."
        )
    return f"ante_{match.group('max_ante')}", int(match.group("seed"))


def load_seed_result_rows(result_path: Path) -> list[dict[str, Any]]:
    """Load one seed-specific comparison JSON into per-bot summary rows."""

    ante_label, seed = parse_seed_result_metadata(result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for bot_result in payload["bot_results"]:
        rows.append(
            {
                "ante_label": ante_label,
                "seed": seed,
                "bot": bot_result["bot_name"],
                "win_rate_pct": float(bot_result["win_rate"]) * 100.0,
                "avg_rounds": float(bot_result["average_rounds_passed"]),
                "avg_final_chips": float(bot_result["average_final_chips_scored"]),
                "std_final_chips": float(bot_result["final_chips_std_dev"]),
            }
        )
    return rows


def summarize_seed_result_paths(result_paths: list[Path]) -> list[dict[str, Any]]:
    """Aggregate per-seed JSON results into mean/std summary rows."""

    grouped_metric_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result_path in result_paths:
        for row in load_seed_result_rows(result_path):
            grouped_metric_rows[(row["ante_label"], row["bot"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (ante_label, bot_name), rows in sorted(
        grouped_metric_rows.items(),
        key=lambda item: (item[0][0], _bot_sort_key(item[0][1])),
    ):
        win_rate_values = [row["win_rate_pct"] for row in rows]
        avg_round_values = [row["avg_rounds"] for row in rows]
        avg_chip_values = [row["avg_final_chips"] for row in rows]
        summary_rows.append(
            {
                "ante_label": ante_label,
                "bot": bot_name,
                "num_seeds": len(rows),
                "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda row: row["seed"])),
                "mean_win_rate_pct": statistics.mean(win_rate_values),
                "std_win_rate_pct": _sample_std(win_rate_values),
                "mean_avg_rounds": statistics.mean(avg_round_values),
                "std_avg_rounds": _sample_std(avg_round_values),
                "mean_avg_final_chips": statistics.mean(avg_chip_values),
                "std_avg_final_chips": _sample_std(avg_chip_values),
            }
        )
    return summary_rows


def write_seed_summary_csv(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write the aggregated seed summary rows to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ante_label",
        "bot",
        "num_seeds",
        "seeds",
        "mean_win_rate_pct",
        "std_win_rate_pct",
        "mean_avg_rounds",
        "std_avg_rounds",
        "mean_avg_final_chips",
        "std_avg_final_chips",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_seed_summary(summary_rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Create ante-specific bar charts with seed-to-seed error bars."""

    mpl_config_dir = output_dir.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []
    ante_labels = sorted(set(row["ante_label"] for row in summary_rows))
    for ante_label in ante_labels:
        ante_rows = [row for row in summary_rows if row["ante_label"] == ante_label]
        if not ante_rows:
            continue

        bot_names = [row["bot"] for row in ante_rows]
        for metric_key, ylabel, file_suffix in PLOT_SPECS:
            error_key = metric_key.replace("mean_", "std_")
            values = [float(row[metric_key]) for row in ante_rows]
            errors = [float(row[error_key]) for row in ante_rows]

            plt.figure(figsize=(10, 5.5))
            plt.bar(bot_names, values, yerr=errors, capsize=4)
            plt.xticks(rotation=20, ha="right")
            plt.ylabel(ylabel)
            plt.title(f"{ante_label}: {ylabel} by bot")
            plt.tight_layout()

            output_path = output_dir / f"{ante_label}_{file_suffix}"
            plt.savefig(output_path, dpi=200)
            plt.close()
            created_paths.append(output_path)
    return created_paths


def format_summary_table(summary_rows: list[dict[str, Any]]) -> str:
    """Render a compact text table for CLI display."""

    headers = [
        "Ante",
        "Bot",
        "Mean Win %",
        "Std Win %",
        "Mean Rounds",
        "Std Rounds",
        "Mean Chips",
        "Std Chips",
    ]
    rows = [
        [
            row["ante_label"],
            row["bot"],
            f"{row['mean_win_rate_pct']:.2f}",
            f"{row['std_win_rate_pct']:.2f}",
            f"{row['mean_avg_rounds']:.2f}",
            f"{row['std_avg_rounds']:.2f}",
            f"{row['mean_avg_final_chips']:.2f}",
            f"{row['std_avg_final_chips']:.2f}",
        ]
        for row in summary_rows
    ]
    column_widths = [
        max(len(str(row[column_index])) for row in [headers] + rows)
        for column_index in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        return " | ".join(value.ljust(column_widths[index]) for index, value in enumerate(row))

    table_lines = [format_row(headers), "-+-".join("-" * width for width in column_widths)]
    table_lines.extend(format_row(row) for row in rows)
    return "\n".join(table_lines)


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def _ante_sort_key(ante_label: str) -> int:
    """Sort ante labels naturally (ante_1 < ante_2 < ... < ante_8)."""
    match = re.match(r"ante_(\d+)", ante_label)
    return int(match.group(1)) if match else 99


def _bot_sort_key(bot_name: str) -> int:
    if bot_name in BOT_ORDER:
        return BOT_ORDER.index(bot_name)
    return len(BOT_ORDER)


__all__ = [
    "BOT_ORDER",
    "default_seed_result_paths",
    "format_summary_table",
    "load_seed_result_rows",
    "parse_seed_result_metadata",
    "plot_seed_summary",
    "summarize_seed_result_paths",
    "write_seed_summary_csv",
]
