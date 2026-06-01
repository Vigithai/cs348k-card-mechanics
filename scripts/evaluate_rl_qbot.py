"""Evaluate RLQBot against the heuristic baselines."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp import (
    BalatroMVPEnvironment,
    DEFAULT_MAX_ANTE,
    DiscardLowestChipBot,
    LookaheadDiscardBot,
    PrunedSampledLookaheadBot,
    RLQBot,
    RandomBot,
    StimBot,
)
from balatro_mvp.rl import load_q_network_from_checkpoint


BotFactory = Callable[[random.Random], object]
RESULTS_DIR = REPO_ROOT / "results" / "rl_eval"


def default_output_path(max_ante: int, seed: int) -> Path:
    """Return the default comparison-results path."""

    return RESULTS_DIR / f"ante_{max_ante}_comparison_seed_{seed}.json"


def evaluate_bot(
    bot_name: str,
    bot_factory: BotFactory,
    *,
    num_games: int,
    base_seed: int,
    max_ante: int,
) -> dict[str, Any]:
    """Evaluate one bot over a reproducible seeded game batch."""

    wins = 0
    total_rounds_passed = 0
    hand_type_counts: Counter[str] = Counter()
    final_chip_scores: list[int] = []

    for game_index in range(num_games):
        seed = base_seed + game_index
        env = BalatroMVPEnvironment(seed=seed, max_ante=max_ante)
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
            raise RuntimeError("Environment state unexpectedly missing after evaluation.")
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


def print_summary_table(
    bot_results: list[dict[str, Any]],
    *,
    max_ante: int,
) -> None:
    """Print a compact summary table for the evaluated bots."""

    print(f"Max Ante: {max_ante} (Antes 1-{max_ante}, 3 blinds each)")

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
    checkpoint_path: Path,
    max_ante: int,
    num_games: int,
    base_seed: int,
    bot_results: list[dict[str, Any]],
) -> None:
    """Persist comparison results to a JSON summary."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "max_ante": max_ante,
                "checkpoint_path": str(checkpoint_path),
                "num_games": num_games,
                "base_seed": base_seed,
                "bot_results": bot_results,
            },
            output_file,
            indent=2,
            sort_keys=True,
        )
    print(f"\nSaved results to {output_path}")


def main() -> None:
    """Parse arguments and compare RLQBot against the heuristic baselines."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to an RLQBot checkpoint.")
    parser.add_argument("--games", type=int, default=25, help="Number of seeded games to run per bot.")
    parser.add_argument("--base-seed", type=int, default=0, help="Starting seed for simulation runs.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Training seed index used to name the output file (ante_N_comparison_seed_N.json).",
    )
    parser.add_argument(
        "--max-ante",
        type=int,
        default=DEFAULT_MAX_ANTE,
        help="Maximum ante to win the run (default 8).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use for RLQBot inference.",
    )
    parser.add_argument(
        "--pruned-sample-count",
        type=int,
        default=24,
        help="Redraw sample count for PrunedSampledLookaheadBot discard evaluation.",
    )
    parser.add_argument(
        "--pruned-discard-margin",
        type=float,
        default=10.0,
        help="Required discard-value margin over the best immediate play.",
    )
    parser.add_argument(
        "--pruned-candidate-pool-size",
        type=int,
        default=5,
        help="How many of the most discardable cards seed the pruned subset search.",
    )
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be positive.")
    if args.pruned_sample_count <= 0:
        raise ValueError("--pruned-sample-count must be positive.")
    if args.pruned_candidate_pool_size <= 0:
        raise ValueError("--pruned-candidate-pool-size must be positive.")

    max_ante = args.max_ante
    output_path = args.output if args.output is not None else default_output_path(max_ante, args.seed)
    rl_q_network, _ = load_q_network_from_checkpoint(args.checkpoint, device=args.device)

    bot_factories: list[tuple[str, BotFactory]] = [
        ("RandomBot", lambda rng: RandomBot(rng=rng)),
        ("StimBot", lambda rng: StimBot(rng=rng)),
        ("DiscardLowestChipBot", lambda rng: DiscardLowestChipBot(rng=rng)),
        ("LookaheadDiscardBot", lambda rng: LookaheadDiscardBot(rng=rng)),
        (
            "PrunedSampledLookaheadBot",
            lambda rng: PrunedSampledLookaheadBot(
                rng=rng,
                sample_count=args.pruned_sample_count,
                discard_margin=args.pruned_discard_margin,
                pruning_candidate_pool_size=args.pruned_candidate_pool_size,
            ),
        ),
        (
            "RLQBot",
            lambda rng: RLQBot(
                rl_q_network,
                rng=rng,
                epsilon=0.0,
                training=False,
                device=args.device,
            ),
        ),
    ]

    bot_results = [
        evaluate_bot(
            bot_name,
            bot_factory,
            num_games=args.games,
            base_seed=args.base_seed,
            max_ante=max_ante,
        )
        for bot_name, bot_factory in bot_factories
    ]
    print_summary_table(bot_results, max_ante=max_ante)
    save_results(
        output_path=output_path,
        checkpoint_path=args.checkpoint,
        max_ante=max_ante,
        num_games=args.games,
        base_seed=args.base_seed,
        bot_results=bot_results,
    )


if __name__ == "__main__":
    main()
