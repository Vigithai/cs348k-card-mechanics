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
    """Parse CLI arguments, summarize easy/hard seed results, and save CSV output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--easy",
        type=Path,
        nargs="*",
        default=None,
        help="Optional easy-preset comparison JSON files.",
    )
    parser.add_argument(
        "--hard",
        type=Path,
        nargs="*",
        default=None,
        help="Optional hard-preset comparison JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "results" / "rl_summary" / "seed_summary.csv",
        help="CSV path for the aggregated summary.",
    )
    args = parser.parse_args()

    default_paths = default_seed_result_paths(REPO_ROOT)
    easy_paths = args.easy if args.easy is not None and args.easy else default_paths["easy"]
    hard_paths = args.hard if args.hard is not None and args.hard else default_paths["hard"]
    if not easy_paths and not hard_paths:
        raise ValueError("No seed comparison JSON files were found to summarize.")

    summary_rows = summarize_seed_result_paths(list(easy_paths) + list(hard_paths))
    write_seed_summary_csv(summary_rows, args.output_csv)

    print(format_summary_table(summary_rows))
    print(f"\nSaved summary CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
