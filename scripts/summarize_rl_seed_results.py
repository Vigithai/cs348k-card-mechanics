"""Aggregate multi-seed RL comparison JSON files into a slide-ready CSV summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp.rl_summary import (
    default_seed_result_paths,
    format_summary_table,
    summarize_seed_result_paths,
    write_seed_summary_csv,
)


def main() -> None:
    """Parse CLI arguments, summarize RL seed results, and save CSV output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="*",
        help=(
            "Optional explicit ante_N_comparison_seed_N.json files. "
            "If omitted, auto-discovers all matching files in results/rl_eval/."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "results" / "rl_eval" / "seed_summary.csv",
        help="CSV path for the aggregated summary.",
    )
    args = parser.parse_args()

    if args.inputs:
        result_paths = args.inputs
    else:
        discovered = default_seed_result_paths(REPO_ROOT)
        result_paths = [p for paths in discovered.values() for p in paths]

    if not result_paths:
        raise ValueError(
            "No seed comparison JSON files found. "
            "Run evaluate_rl_qbot.py first, or pass files explicitly."
        )

    summary_rows = summarize_seed_result_paths(list(result_paths))
    write_seed_summary_csv(summary_rows, args.output_csv)

    print(format_summary_table(summary_rows))
    print(f"\nSaved summary CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
