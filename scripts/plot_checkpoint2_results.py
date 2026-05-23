"""Plot checkpoint-2 evaluation results from saved JSON summaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = REPO_ROOT / "results" / "checkpoint2_eval_results.json"
DEFAULT_FIGURES_DIR = REPO_ROOT / "results" / "figures"
MPLCONFIGDIR = REPO_ROOT / "results" / ".mplconfig"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(results_path: Path) -> dict[str, object]:
    """Load the saved evaluation JSON file."""
    with results_path.open("r", encoding="utf-8") as results_file:
        return json.load(results_file)


def plot_average_final_chips(bot_results: list[dict[str, object]], figures_dir: Path) -> None:
    """Save a bar chart for average final chips scored by bot."""
    bot_names = [result["bot_name"] for result in bot_results]
    average_scores = [result["average_final_chips_scored"] for result in bot_results]

    plt.figure(figsize=(8, 5))
    plt.bar(bot_names, average_scores, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("Average Final Chips")
    plt.title("Checkpoint-2 Average Final Chips by Bot")
    plt.tight_layout()
    plt.savefig(figures_dir / "average_final_chips_by_bot.png", dpi=150)
    plt.close()


def plot_win_rate(bot_results: list[dict[str, object]], figures_dir: Path) -> None:
    """Save a bar chart for win rate by bot."""
    bot_names = [result["bot_name"] for result in bot_results]
    win_rates = [result["win_rate"] for result in bot_results]

    plt.figure(figsize=(8, 5))
    plt.bar(bot_names, win_rates, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("Win Rate")
    plt.ylim(0.0, 1.0)
    plt.title("Checkpoint-2 Win Rate by Bot")
    plt.tight_layout()
    plt.savefig(figures_dir / "win_rate_by_bot.png", dpi=150)
    plt.close()


def plot_hand_type_distribution(bot_results: list[dict[str, object]], figures_dir: Path) -> None:
    """Save a stacked bar chart for hand-type counts by bot."""
    bot_names = [result["bot_name"] for result in bot_results]
    hand_types = sorted(
        {
            hand_type
            for result in bot_results
            for hand_type in result["hand_type_counts"].keys()
        }
    )

    plt.figure(figsize=(10, 6))
    bottoms = [0] * len(bot_names)
    color_cycle = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#EECA3B",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
    ]

    for color_index, hand_type in enumerate(hand_types):
        counts = [result["hand_type_counts"].get(hand_type, 0) for result in bot_results]
        plt.bar(
            bot_names,
            counts,
            bottom=bottoms,
            label=hand_type,
            color=color_cycle[color_index % len(color_cycle)],
        )
        bottoms = [bottom + count for bottom, count in zip(bottoms, counts)]

    plt.ylabel("Total Hands Scored")
    plt.title("Checkpoint-2 Hand-Type Distribution by Bot")
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(figures_dir / "hand_type_distribution_by_bot.png", dpi=150)
    plt.close()


def main() -> None:
    """Load saved evaluation results and generate reproducible plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to the checkpoint evaluation JSON file.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory where figures should be written.",
    )
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

    results = load_results(args.input)
    bot_results = results["bot_results"]

    plot_average_final_chips(bot_results, args.figures_dir)
    plot_win_rate(bot_results, args.figures_dir)
    plot_hand_type_distribution(bot_results, args.figures_dir)

    print(f"Saved figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
