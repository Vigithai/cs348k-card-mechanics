"""Plot evaluation results across presets from saved JSON summaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATHS = {
    "hard": REPO_ROOT / "results" / "hard_eval_results.json",
    "easy": REPO_ROOT / "results" / "easy_eval_results.json",
}
DEFAULT_FIGURES_DIR = REPO_ROOT / "results" / "figures"
MPLCONFIGDIR = REPO_ROOT / "results" / ".mplconfig"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(results_path: Path) -> dict[str, object]:
    """Load one saved evaluation JSON file."""
    with results_path.open("r", encoding="utf-8") as results_file:
        return json.load(results_file)


def plot_metric_across_presets(
    results_payloads: list[dict[str, object]],
    *,
    metric_key: str,
    ylabel: str,
    title: str,
    output_name: str,
    figures_dir: Path,
    y_limit: tuple[float, float] | None = None,
) -> None:
    """Save a grouped bar chart comparing one metric across presets."""
    preset_names = [payload.get("ante_label", payload.get("preset", f"ante_{payload.get('max_ante', '?')}")) for payload in results_payloads]
    bot_names = [result["bot_name"] for result in results_payloads[0]["bot_results"]]
    x_positions = list(range(len(bot_names)))
    bar_width = 0.8 / max(1, len(results_payloads))
    color_cycle = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    plt.figure(figsize=(9, 5))
    for preset_index, payload in enumerate(results_payloads):
        offset = (preset_index - (len(results_payloads) - 1) / 2) * bar_width
        metric_values = [result[metric_key] for result in payload["bot_results"]]
        plt.bar(
            [position + offset for position in x_positions],
            metric_values,
            width=bar_width,
            label=preset_names[preset_index],
            color=color_cycle[preset_index % len(color_cycle)],
        )

    plt.xticks(x_positions, bot_names)
    plt.ylabel(ylabel)
    plt.title(title)
    if y_limit is not None:
        plt.ylim(*y_limit)
    plt.legend(title="Preset")
    plt.tight_layout()
    plt.savefig(figures_dir / output_name, dpi=150)
    plt.close()


def plot_hand_type_distribution(results_payload: dict[str, object], figures_dir: Path) -> None:
    """Save a stacked hand-type distribution chart for one preset."""
    preset_name = results_payload.get("ante_label", results_payload.get("preset", f"ante_{results_payload.get('max_ante', '?')}"))
    bot_results = results_payload["bot_results"]
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
    plt.title(f"Hand-Type Distribution by Bot ({preset_name})")
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(figures_dir / f"hand_type_distribution_by_bot_{preset_name}.png", dpi=150)
    plt.close()


def resolve_input_paths(input_paths: list[Path] | None) -> list[Path]:
    """Resolve explicit or default result input paths."""
    if input_paths:
        return input_paths

    resolved_paths = [path for path in DEFAULT_RESULTS_PATHS.values() if path.exists()]
    if not resolved_paths:
        raise FileNotFoundError(
            "No input result files were provided and no default preset result files were found."
        )
    return resolved_paths


def main() -> None:
    """Load saved evaluation results and generate reproducible comparison plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=None,
        help="One or more evaluation JSON files. Defaults to available hard/easy result files.",
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

    input_paths = resolve_input_paths(args.inputs)
    results_payloads = [load_results(input_path) for input_path in input_paths]

    plot_metric_across_presets(
        results_payloads,
        metric_key="average_final_chips_scored",
        ylabel="Average Final Chips",
        title="Average Final Chips by Bot Across Presets",
        output_name="average_final_chips_by_bot_across_presets.png",
        figures_dir=args.figures_dir,
    )
    plot_metric_across_presets(
        results_payloads,
        metric_key="win_rate",
        ylabel="Win Rate",
        title="Win Rate by Bot Across Presets",
        output_name="win_rate_by_bot_across_presets.png",
        figures_dir=args.figures_dir,
        y_limit=(0.0, 1.0),
    )
    plot_metric_across_presets(
        results_payloads,
        metric_key="average_rounds_passed",
        ylabel="Average Rounds Passed",
        title="Average Rounds Passed by Bot Across Presets",
        output_name="average_rounds_passed_by_bot_across_presets.png",
        figures_dir=args.figures_dir,
    )

    for results_payload in results_payloads:
        plot_hand_type_distribution(results_payload, args.figures_dir)

    print(f"Saved figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
