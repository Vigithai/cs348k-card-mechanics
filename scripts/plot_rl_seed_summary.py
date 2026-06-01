"""Generate preset-specific plots from the aggregated RL seed summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp.rl_summary import plot_seed_summary


def main() -> None:
    """Parse CLI arguments, load the seed summary CSV, and save comparison plots."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPO_ROOT / "results" / "rl_eval" / "seed_summary.csv",
        help="Aggregated RL seed summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "figures",
        help="Directory for the generated figures.",
    )
    args = parser.parse_args()

    summary_rows = []
    with args.input_csv.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            summary_rows.append(
                {
                    "ante_label": row["ante_label"],
                    "bot": row["bot"],
                    "mean_win_rate_pct": float(row["mean_win_rate_pct"]),
                    "std_win_rate_pct": float(row["std_win_rate_pct"]),
                    "mean_avg_rounds": float(row["mean_avg_rounds"]),
                    "std_avg_rounds": float(row["std_avg_rounds"]),
                    "mean_avg_final_chips": float(row["mean_avg_final_chips"]),
                    "std_avg_final_chips": float(row["std_avg_final_chips"]),
                }
            )

    created_paths = plot_seed_summary(summary_rows, args.output_dir)
    for created_path in created_paths:
        print(f"Saved {created_path}")


if __name__ == "__main__":
    main()
