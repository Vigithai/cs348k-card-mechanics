"""Run checkpoint-2 seeded simulations for the baseline Balatro MVP bots."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import argparse
import random
import statistics
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
DEFAULT_RESULTS_PATH = REPO_ROOT / "results" / "checkpoint2_eval_results.json"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp import BalatroMVPEnvironment, DiscardLowestChipBot, RandomBot, StimBot


BotFactory = Callable[[random.Random], object]


def evaluate_bot(
    bot_name: str,
    bot_factory: BotFactory,
    *,
    num_games: int,
    base_seed: int,
) -> dict[str, Any]:
    """Run repeated seeded games for one bot and return summary metrics."""
    wins = 0
    total_rounds_passed = 0
    hand_type_counts: Counter[str] = Counter()
    final_chip_scores: list[int] = []

    for game_index in range(num_games):
        seed = base_seed + game_index
        env = BalatroMVPEnvironment(seed=seed)
        bot = bot_factory(random.Random(seed))

        done = False
        rounds_passed = 0
        while not done:
            observation = env.get_observation()
            legal_actions = env.get_legal_actions()
            action = bot.act(observation, legal_actions)
            _, _, done, info = env.step(action)

            hand_category = info.get("hand_category")
            if hand_category is not None:
                hand_type_counts[hand_category] += 1
            if info.get("round_result") == "round_win":
                rounds_passed += 1

        if env.state is None:
            raise RuntimeError("Environment state unexpectedly missing after simulation.")

        if env.state.result == "run_win":
            wins += 1
        total_rounds_passed += rounds_passed
        final_chip_scores.append(env.state.chips_scored)

    average_final_chips = sum(final_chip_scores) / num_games
    final_chips_std_dev = statistics.pstdev(final_chip_scores) if len(final_chip_scores) > 1 else 0.0

    return {
        "bot_name": bot_name,
        "num_games": num_games,
        "win_rate": wins / num_games,
        "average_rounds_passed": total_rounds_passed / num_games,
        "average_final_chips_scored": average_final_chips,
        "final_chips_std_dev": final_chips_std_dev,
        "hand_type_counts": dict(sorted(hand_type_counts.items())),
    }


def print_summary_table(bot_results: list[dict[str, Any]]) -> None:
    """Print a compact summary table for the evaluated bots."""
    headers = [
        "Bot",
        "Win Rate",
        "Avg Rounds",
        "Avg Final Chips",
        "Std Final Chips",
        "Hand Types",
    ]
    rows = [
        [
            result["bot_name"],
            f"{result['win_rate']:.2%}",
            f"{result['average_rounds_passed']:.2f}",
            f"{result['average_final_chips_scored']:.2f}",
            f"{result['final_chips_std_dev']:.2f}",
            json.dumps(result["hand_type_counts"], sort_keys=True),
        ]
        for result in bot_results
    ]
    column_widths = [
        max(len(str(row[column_index])) for row in [headers] + rows)
        for column_index in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        return " | ".join(value.ljust(column_widths[index]) for index, value in enumerate(row))

    print(format_row(headers))
    print("-+-".join("-" * width for width in column_widths))
    for row in rows:
        print(format_row(row))


def save_results(
    *,
    output_path: Path,
    num_games: int,
    base_seed: int,
    bot_results: list[dict[str, Any]],
) -> None:
    """Write checkpoint evaluation results to a machine-readable JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_payload = {
        "num_games": num_games,
        "base_seed": base_seed,
        "bot_results": bot_results,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results_payload, output_file, indent=2, sort_keys=True)
    print(f"\nSaved results to {output_path}")


def main() -> None:
    """Parse arguments and run the checkpoint-2 baseline bot simulations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=25, help="Number of seeded games to run per bot.")
    parser.add_argument("--base-seed", type=int, default=0, help="Starting seed for simulation runs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to the JSON file where summary results should be saved.",
    )
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be positive.")

    bot_factories: list[tuple[str, BotFactory]] = [
        ("RandomBot", lambda rng: RandomBot(rng=rng)),
        ("StimBot", lambda rng: StimBot(rng=rng)),
        ("DiscardLowestChipBot", lambda rng: DiscardLowestChipBot(rng=rng)),
    ]

    bot_results = []
    for bot_name, bot_factory in bot_factories:
        bot_results.append(
            evaluate_bot(bot_name, bot_factory, num_games=args.games, base_seed=args.base_seed)
        )

    print_summary_table(bot_results)
    save_results(
        output_path=args.output,
        num_games=args.games,
        base_seed=args.base_seed,
        bot_results=bot_results,
    )


if __name__ == "__main__":
    main()
