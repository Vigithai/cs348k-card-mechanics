"""Run preset-based seeded simulations for the Balatro MVP baseline bots."""

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
RESULTS_DIR = REPO_ROOT / "results"
TRACE_RESULTS_DIR = RESULTS_DIR / "traces" / "scripted"
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp import (
    Action,
    BalatroMVPEnvironment,
    Card,
    DEFAULT_MAX_ANTE,
    DiscardLowestChipBot,
    LookaheadDiscardBot,
    PrunedSampledLookaheadBot,
    RandomBot,
    StimBot,
)


BotFactory = Callable[[random.Random], object]


def default_output_path(max_ante: int) -> Path:
    """Return the default JSON results path."""
    return RESULTS_DIR / "scripted_eval" / f"ante_{max_ante}_eval_results.json"


def evaluate_bot(
    bot_name: str,
    bot_factory: BotFactory,
    *,
    num_games: int,
    base_seed: int,
    max_ante: int,
    save_traces: bool,
    trace_limit: int,
    trace_bot_name: str,
    trace_output_dir: Path,
) -> dict[str, Any]:
    """Run repeated seeded games for one bot and return summary metrics."""
    wins = 0
    total_blinds_cleared = 0
    hand_type_counts: Counter[str] = Counter()
    final_chip_scores: list[int] = []
    decision_debug_records: list[dict[str, Any]] = []

    for game_index in range(num_games):
        seed = base_seed + game_index
        env = BalatroMVPEnvironment(seed=seed, max_ante=max_ante)
        bot = bot_factory(random.Random(seed))

        should_save_trace = save_traces and bot_name == trace_bot_name and game_index < trace_limit
        trace_records: list[dict[str, Any]] = []
        done = False
        blinds_cleared = 0
        turn_index = 0

        while not done:
            observation = env.get_observation()
            legal_actions = env.get_legal_actions()
            action = bot.act(observation, legal_actions)
            agent_debug = bot.get_last_decision_info() if hasattr(bot, "get_last_decision_info") else None
            _, reward, done, info = env.step(action)

            if should_save_trace:
                trace_records.append(
                    {
                        "ante": observation["ante"],
                        "blind_type": observation["blind_type"],
                        "turn_index": turn_index,
                        "chips_needed": observation["chips_needed"],
                        "chips_scored_before_action": observation["chips_scored"],
                        "chosen_action": serialize_action(action),
                        "scored_hand_type": info.get("hand_category"),
                        "reward": reward,
                        "agent_debug": serialize_value(agent_debug),
                        "info": serialize_value(info),
                    }
                )

            if agent_debug is not None:
                decision_debug_records.append(serialize_value(agent_debug))

            hand_category = info.get("hand_category")
            if hand_category is not None:
                hand_type_counts[hand_category] += 1
            if info.get("round_result") == "round_win":
                blinds_cleared += 1

            turn_index += 1

        if env.state is None:
            raise RuntimeError("Environment state unexpectedly missing after simulation.")

        if should_save_trace:
            save_trace(
                trace_output_dir=trace_output_dir,
                max_ante=max_ante,
                bot_name=bot_name,
                seed=seed,
                trace_records=trace_records,
            )

        if env.state.result == "run_win":
            wins += 1
        total_blinds_cleared += blinds_cleared
        final_chip_scores.append(env.state.chips_scored)

    average_final_chips = sum(final_chip_scores) / num_games
    final_chips_std_dev = statistics.pstdev(final_chip_scores) if len(final_chip_scores) > 1 else 0.0

    bot_summary = {
        "bot_name": bot_name,
        "num_games": num_games,
        "win_rate": wins / num_games,
        "average_blinds_cleared": total_blinds_cleared / num_games,
        "average_final_chips_scored": average_final_chips,
        "final_chips_std_dev": final_chips_std_dev,
        "hand_type_counts": dict(sorted(hand_type_counts.items())),
    }
    if decision_debug_records:
        bot_summary["decision_debug_summary"] = summarize_decision_debug_records(decision_debug_records)
    return bot_summary


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
        "Avg Blinds",
        "Avg Final Chips",
        "Std Final Chips",
        "Hand Types",
    ]
    rows = [
        [
            result["bot_name"],
            f"{result['win_rate']:.2%}",
            f"{result['average_blinds_cleared']:.2f}",
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
    max_ante: int,
    num_games: int,
    base_seed: int,
    bot_results: list[dict[str, Any]],
) -> None:
    """Write evaluation results to a machine-readable JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_payload = {
        "max_ante": max_ante,
        "num_games": num_games,
        "base_seed": base_seed,
        "bot_results": bot_results,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results_payload, output_file, indent=2, sort_keys=True)
    print(f"\nSaved results to {output_path}")


def save_trace(
    *,
    trace_output_dir: Path,
    max_ante: int,
    bot_name: str,
    seed: int,
    trace_records: list[dict[str, Any]],
) -> None:
    """Write one sample run trace to JSON."""
    ante_trace_dir = trace_output_dir / f"ante_{max_ante}"
    ante_trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = ante_trace_dir / f"{bot_name}_seed_{seed}.json"
    trace_payload = {
        "max_ante": max_ante,
        "bot_name": bot_name,
        "seed": seed,
        "trace": trace_records,
    }
    with trace_path.open("w", encoding="utf-8") as trace_file:
        json.dump(trace_payload, trace_file, indent=2, sort_keys=True)


def summarize_decision_debug_records(decision_debug_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate optional bot decision debug info into a compact summary."""
    chosen_discard_size_counts: Counter[str] = Counter()
    raw_legal_discard_counts: list[int] = []
    pruned_discard_candidate_counts: list[int] = []
    redraw_sample_counts: list[int] = []

    for record in decision_debug_records:
        chosen_discard_size_counts[str(record.get("chosen_discard_size", 0))] += 1
        if "raw_legal_discard_count" in record:
            raw_legal_discard_counts.append(int(record["raw_legal_discard_count"]))
        if "pruned_discard_candidate_count" in record:
            pruned_discard_candidate_counts.append(int(record["pruned_discard_candidate_count"]))
        if "redraw_sample_count_used" in record:
            redraw_sample_counts.append(int(record["redraw_sample_count_used"]))

    return {
        "average_raw_legal_discard_count": (
            sum(raw_legal_discard_counts) / len(raw_legal_discard_counts) if raw_legal_discard_counts else 0.0
        ),
        "average_pruned_discard_candidate_count": (
            sum(pruned_discard_candidate_counts) / len(pruned_discard_candidate_counts)
            if pruned_discard_candidate_counts
            else 0.0
        ),
        "average_redraw_sample_count_used": (
            sum(redraw_sample_counts) / len(redraw_sample_counts) if redraw_sample_counts else 0.0
        ),
        "chosen_discard_size_counts": dict(sorted(chosen_discard_size_counts.items())),
    }


def serialize_action(action: Action) -> dict[str, Any]:
    """Convert an Action into a JSON-friendly representation."""
    return {
        "type": action.type,
        "card_indices": list(action.card_indices),
    }


def serialize_card(card: Card) -> dict[str, Any]:
    """Convert a Card into a JSON-friendly representation."""
    return {
        "rank": card.rank,
        "suit": card.suit,
        "chip_value": card.chip_value,
    }


def serialize_value(value: Any) -> Any:
    """Recursively convert values into JSON-friendly data."""
    if isinstance(value, Card):
        return serialize_card(value)
    if isinstance(value, Action):
        return serialize_action(value)
    if isinstance(value, dict):
        return {str(key): serialize_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    return value


def main() -> None:
    """Parse arguments and run ante/blind-based baseline bot simulations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=25, help="Number of seeded games to run per bot.")
    parser.add_argument("--base-seed", type=int, default=0, help="Starting seed for simulation runs.")
    parser.add_argument(
        "--max-ante",
        type=int,
        default=2,
        help="Maximum ante to win the run (default 2; full game is 8).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to results/ante_<N>_eval_results.json.",
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Save a few sample traces for the configured trace bot.",
    )
    parser.add_argument(
        "--trace-bot",
        default="PrunedSampledLookaheadBot",
        help="Bot name for sample trace output when --save-traces is enabled.",
    )
    parser.add_argument(
        "--trace-limit",
        type=int,
        default=3,
        help="Maximum number of sample traces to save for the trace bot.",
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
    if args.trace_limit < 0:
        raise ValueError("--trace-limit cannot be negative.")
    if args.pruned_sample_count <= 0:
        raise ValueError("--pruned-sample-count must be positive.")
    if args.pruned_candidate_pool_size <= 0:
        raise ValueError("--pruned-candidate-pool-size must be positive.")

    max_ante = args.max_ante
    output_path = args.output if args.output is not None else default_output_path(max_ante)

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
    ]

    bot_results = []
    for bot_name, bot_factory in bot_factories:
        bot_results.append(
            evaluate_bot(
                bot_name,
                bot_factory,
                num_games=args.games,
                base_seed=args.base_seed,
                max_ante=max_ante,
                save_traces=args.save_traces,
                trace_limit=args.trace_limit,
                trace_bot_name=args.trace_bot,
                trace_output_dir=TRACE_RESULTS_DIR,
            )
        )

    print_summary_table(bot_results, max_ante=max_ante)
    save_results(
        output_path=output_path,
        max_ante=max_ante,
        num_games=args.games,
        base_seed=args.base_seed,
        bot_results=bot_results,
    )


if __name__ == "__main__":
    main()
