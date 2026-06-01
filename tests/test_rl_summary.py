from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp.rl_summary import parse_seed_result_metadata, summarize_seed_result_paths


class RLSeedSummaryTests(unittest.TestCase):
    def test_parse_seed_result_metadata_reads_ante_and_seed(self) -> None:
        ante_label, seed = parse_seed_result_metadata(Path("ante_2_comparison_seed_3.json"))

        self.assertEqual(ante_label, "ante_2")
        self.assertEqual(seed, 3)

    def test_summarize_seed_result_paths_aggregates_mean_and_std(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            ante2_seed_0 = temp_path / "ante_2_comparison_seed_0.json"
            ante2_seed_1 = temp_path / "ante_2_comparison_seed_1.json"
            ante8_seed_0 = temp_path / "ante_8_comparison_seed_0.json"

            self._write_payload(
                ante2_seed_0,
                rl_win_rate=0.40,
                rl_avg_rounds=1.20,
                rl_avg_final_chips=200.0,
            )
            self._write_payload(
                ante2_seed_1,
                rl_win_rate=0.50,
                rl_avg_rounds=1.40,
                rl_avg_final_chips=240.0,
            )
            self._write_payload(
                ante8_seed_0,
                rl_win_rate=0.02,
                rl_avg_rounds=0.35,
                rl_avg_final_chips=210.0,
            )

            summary_rows = summarize_seed_result_paths([ante2_seed_0, ante2_seed_1, ante8_seed_0])

        ante2_rl_row = next(
            row for row in summary_rows if row["ante_label"] == "ante_2" and row["bot"] == "RLQBot"
        )
        ante8_rl_row = next(
            row for row in summary_rows if row["ante_label"] == "ante_8" and row["bot"] == "RLQBot"
        )

        self.assertEqual(ante2_rl_row["num_seeds"], 2)
        self.assertEqual(ante2_rl_row["seeds"], "0,1")
        self.assertAlmostEqual(ante2_rl_row["mean_win_rate_pct"], 45.0)
        self.assertAlmostEqual(ante2_rl_row["std_win_rate_pct"], 7.0710678118654755)
        self.assertAlmostEqual(ante2_rl_row["mean_avg_rounds"], 1.3)
        self.assertAlmostEqual(ante2_rl_row["mean_avg_final_chips"], 220.0)

        self.assertEqual(ante8_rl_row["num_seeds"], 1)
        self.assertEqual(ante8_rl_row["seeds"], "0")
        self.assertAlmostEqual(ante8_rl_row["mean_win_rate_pct"], 2.0)
        self.assertAlmostEqual(ante8_rl_row["std_win_rate_pct"], 0.0)

    def _write_payload(
        self,
        output_path: Path,
        *,
        rl_win_rate: float,
        rl_avg_rounds: float,
        rl_avg_final_chips: float,
    ) -> None:
        payload = {
            "bot_results": [
                {
                    "bot_name": "RandomBot",
                    "win_rate": 0.0,
                    "average_rounds_passed": 0.0,
                    "average_final_chips_scored": 10.0,
                    "final_chips_std_dev": 1.0,
                },
                {
                    "bot_name": "RLQBot",
                    "win_rate": rl_win_rate,
                    "average_rounds_passed": rl_avg_rounds,
                    "average_final_chips_scored": rl_avg_final_chips,
                    "final_chips_std_dev": 20.0,
                },
            ],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
