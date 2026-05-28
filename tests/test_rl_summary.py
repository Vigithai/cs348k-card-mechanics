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
    def test_parse_seed_result_metadata_reads_preset_and_seed(self) -> None:
        preset_name, seed = parse_seed_result_metadata(Path("easy_comparison_seed_2.json"))

        self.assertEqual(preset_name, "easy")
        self.assertEqual(seed, 2)

    def test_summarize_seed_result_paths_aggregates_mean_and_std(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            easy_seed_0 = temp_path / "easy_comparison_seed_0.json"
            easy_seed_1 = temp_path / "easy_comparison_seed_1.json"
            hard_seed_0 = temp_path / "hard_comparison_seed_0.json"

            self._write_payload(
                easy_seed_0,
                preset_name="easy",
                rl_win_rate=0.40,
                rl_avg_rounds=1.20,
                rl_avg_final_chips=200.0,
            )
            self._write_payload(
                easy_seed_1,
                preset_name="easy",
                rl_win_rate=0.50,
                rl_avg_rounds=1.40,
                rl_avg_final_chips=240.0,
            )
            self._write_payload(
                hard_seed_0,
                preset_name="hard",
                rl_win_rate=0.02,
                rl_avg_rounds=0.35,
                rl_avg_final_chips=210.0,
            )

            summary_rows = summarize_seed_result_paths([easy_seed_0, easy_seed_1, hard_seed_0])

        easy_rl_row = next(
            row for row in summary_rows if row["preset"] == "easy" and row["bot"] == "RLQBot"
        )
        hard_rl_row = next(
            row for row in summary_rows if row["preset"] == "hard" and row["bot"] == "RLQBot"
        )

        self.assertEqual(easy_rl_row["num_seeds"], 2)
        self.assertEqual(easy_rl_row["seeds"], "0,1")
        self.assertAlmostEqual(easy_rl_row["mean_win_rate_pct"], 45.0)
        self.assertAlmostEqual(easy_rl_row["std_win_rate_pct"], 7.0710678118654755)
        self.assertAlmostEqual(easy_rl_row["mean_avg_rounds"], 1.3)
        self.assertAlmostEqual(easy_rl_row["mean_avg_final_chips"], 220.0)

        self.assertEqual(hard_rl_row["num_seeds"], 1)
        self.assertEqual(hard_rl_row["seeds"], "0")
        self.assertAlmostEqual(hard_rl_row["mean_win_rate_pct"], 2.0)
        self.assertAlmostEqual(hard_rl_row["std_win_rate_pct"], 0.0)

    def _write_payload(
        self,
        output_path: Path,
        *,
        preset_name: str,
        rl_win_rate: float,
        rl_avg_rounds: float,
        rl_avg_final_chips: float,
    ) -> None:
        payload = {
            "preset": preset_name,
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
