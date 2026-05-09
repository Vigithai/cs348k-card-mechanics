"""Run small seeded simulations for the baseline Balatro MVP bots."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import random
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp import BalatroMVPEnvironment, RandomBot, StimBot


BotFactory = Callable[[random.Random], object]


def run_bot_games(bot_name: str, bot_factory: BotFactory, *, num_games: int, base_seed: int) -> None:
    """Run repeated seeded games for one bot and print summary metrics."""
    wins = 0
    total_rounds_passed = 0
    total_final_chips = 0
    hand_type_counts: Counter[str] = Counter()

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
        total_final_chips += env.state.chips_scored

    print(f"{bot_name}:")
    print(f"  win rate: {wins / num_games:.2%}")
    print(f"  average rounds passed: {total_rounds_passed / num_games:.2f}")
    print(f"  average final chips scored: {total_final_chips / num_games:.2f}")
    print(f"  hand types scored: {dict(sorted(hand_type_counts.items()))}")


def main() -> None:
    """Parse arguments and run the baseline bot simulations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=25, help="Number of seeded games to run per bot.")
    parser.add_argument("--base-seed", type=int, default=0, help="Starting seed for simulation runs.")
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be positive.")

    bot_factories: list[tuple[str, BotFactory]] = [
        ("RandomBot", lambda rng: RandomBot(rng=rng)),
        ("StimBot", lambda rng: StimBot(rng=rng)),
    ]

    for bot_name, bot_factory in bot_factories:
        run_bot_games(bot_name, bot_factory, num_games=args.games, base_seed=args.base_seed)


if __name__ == "__main__":
    main()
