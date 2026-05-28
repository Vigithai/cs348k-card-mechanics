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
SEED_FILE_PATTERN = re.compile(r"(?P<preset>easy|hard)_comparison_seed_(?P<seed>\d+)\.json$")
PLOT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("mean_win_rate_pct", "Win Rate (%)", "win_rate_summary.png"),
    ("mean_avg_rounds", "Average Rounds Passed", "avg_rounds_summary.png"),
    ("mean_avg_final_chips", "Average Final Chips", "avg_final_chips_summary.png"),
)


def default_seed_result_paths(repo_root: Path) -> dict[str, list[Path]]:
    """Return the default easy/hard seed-result JSON files in sorted order."""

    summary_dir = repo_root / "results" / "rl_summary"
    grouped_paths = {"easy": [], "hard": []}
    for result_path in sorted(summary_dir.glob("*_comparison_seed_*.json")):
        preset_name, _ = parse_seed_result_metadata(result_path)
        grouped_paths[preset_name].append(result_path)
    return grouped_paths


def parse_seed_result_metadata(result_path: Path) -> tuple[str, int]:
    """Extract the preset name and seed index from one saved comparison path."""

    match = SEED_FILE_PATTERN.search(result_path.name)
    if match is None:
        raise ValueError(
            f"Could not infer preset/seed from {result_path}. Expected '*_comparison_seed_<n>.json'."
        )
    return match.group("preset"), int(match.group("seed"))


def load_seed_result_rows(result_path: Path) -> list[dict[str, Any]]:
    """Load one seed-specific comparison JSON into per-bot summary rows."""

    preset_name, seed = parse_seed_result_metadata(result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for bot_result in payload["bot_results"]:
        rows.append(
            {
                "preset": preset_name,
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
    """Aggregate a preset's per-seed JSON results into mean/std summary rows."""

    grouped_metric_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result_path in result_paths:
        for row in load_seed_result_rows(result_path):
            grouped_metric_rows[(row["preset"], row["bot"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (preset_name, bot_name), rows in sorted(
        grouped_metric_rows.items(),
        key=lambda item: (_preset_sort_key(item[0][0]), _bot_sort_key(item[0][1])),
    ):
        win_rate_values = [row["win_rate_pct"] for row in rows]
        avg_round_values = [row["avg_rounds"] for row in rows]
        avg_chip_values = [row["avg_final_chips"] for row in rows]
        summary_rows.append(
            {
                "preset": preset_name,
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
        "preset",
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
    """Create preset-specific bar charts with seed-to-seed error bars."""

    mpl_config_dir = output_dir.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []
    for preset_name in ("easy", "hard"):
        preset_rows = [row for row in summary_rows if row["preset"] == preset_name]
        if not preset_rows:
            continue

        bot_names = [row["bot"] for row in preset_rows]
        for metric_key, ylabel, file_suffix in PLOT_SPECS:
            error_key = metric_key.replace("mean_", "std_")
            values = [float(row[metric_key]) for row in preset_rows]
            errors = [float(row[error_key]) for row in preset_rows]

            plt.figure(figsize=(10, 5.5))
            plt.bar(bot_names, values, yerr=errors, capsize=4)
            plt.xticks(rotation=20, ha="right")
            plt.ylabel(ylabel)
            plt.title(f"{preset_name.title()} preset: {ylabel} by bot")
            plt.tight_layout()

            output_path = output_dir / f"{preset_name}_{file_suffix}"
            plt.savefig(output_path, dpi=200)
            plt.close()
            created_paths.append(output_path)
    return created_paths


def format_summary_table(summary_rows: list[dict[str, Any]]) -> str:
    """Render a compact text table for CLI display."""

    headers = [
        "Preset",
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
            row["preset"],
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


def _preset_sort_key(preset_name: str) -> int:
    return {"easy": 0, "hard": 1}.get(preset_name, 99)


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
