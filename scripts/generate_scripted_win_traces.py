"""Generate rich per-step traces for scripted bot winning games.

Mirrors generate_rl_win_traces.py but runs a named scripted bot instead of
RLQBot. Only winning games are saved. Output format is identical so both
trace files can be fed into analyze_win_traces.py for comparison.

Usage
-----
python scripts/generate_scripted_win_traces.py \
    --bot PrunedSampledLookaheadBot \
    --max-ante 1 \
    --games 500 \
    --base-seed 0

Output
------
results/traces/scripted_wins/<bot>_ante<N>_seed<base_seed>_<W>wins.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp import (
    BalatroMVPEnvironment,
    DiscardLowestChipBot,
    LookaheadDiscardBot,
    PrunedSampledLookaheadBot,
    RandomBot,
    StimBot,
)

# ── Bot registry ───────────────────────────────────────────────────────────────

BOT_REGISTRY: dict[str, Any] = {
    "RandomBot": lambda rng: RandomBot(rng=rng),
    "StimBot": lambda rng: StimBot(rng=rng),
    "DiscardLowestChipBot": lambda rng: DiscardLowestChipBot(rng=rng),
    "LookaheadDiscardBot": lambda rng: LookaheadDiscardBot(rng=rng),
    "PrunedSampledLookaheadBot": lambda rng: PrunedSampledLookaheadBot(rng=rng),
}

# ── Card formatting ────────────────────────────────────────────────────────────

SUIT_SYMBOL: dict[str, str] = {
    "club":    "♣",
    "diamond": "♦",
    "heart":   "♥",
    "spade":   "♠",
}


def fmt_card(card: Any) -> str:
    if hasattr(card, "rank"):
        return f"{card.rank}{SUIT_SYMBOL[card.suit]}"
    return f"{card['rank']}{SUIT_SYMBOL[card['suit']]}"


def fmt_hand(cards: Any) -> list[str]:
    return [fmt_card(c) for c in cards]


# ── Trace helpers ──────────────────────────────────────────────────────────────

def cards_added(hand_before: list[str], hand_after: list[str]) -> list[str]:
    """Return cards that appear in hand_after but not hand_before."""
    before_counts: dict[str, int] = {}
    for c in hand_before:
        before_counts[c] = before_counts.get(c, 0) + 1
    added = []
    for c in hand_after:
        if before_counts.get(c, 0) > 0:
            before_counts[c] -= 1
        else:
            added.append(c)
    return added


# ── Core game runner ───────────────────────────────────────────────────────────

def run_one_game(
    bot: Any,
    seed: int,
    max_ante: int,
) -> dict[str, Any] | None:
    """Play one game; return a rich trace dict if it is a win, else None."""

    env = BalatroMVPEnvironment(seed=seed, max_ante=max_ante)
    done = False

    blinds_data: list[dict[str, Any]] = []
    current_blind_steps: list[dict[str, Any]] = []
    current_ante: int = 1
    current_blind_type: str = "small_blind"

    while not done:
        obs = env.get_observation()
        legal_actions = env.get_legal_actions()

        hand_before = fmt_hand(obs["hand"])
        chips_before = obs["chips_scored"]
        hands_left_before = obs["hands_left"]
        discards_left_before = obs["discards_left"]
        chips_needed = obs["chips_needed"]
        ante = obs["ante"]
        blind_type = obs["blind_type"]

        # Detect blind boundary.
        if ante != current_ante or blind_type != current_blind_type:
            blinds_data.append({
                "ante": current_ante,
                "blind_type": current_blind_type,
                "chips_needed": chips_needed,
                "steps": current_blind_steps,
            })
            current_blind_steps = []
            current_ante = ante
            current_blind_type = blind_type

        action = bot.act(obs, legal_actions)
        next_obs, _, done, info = env.step(action)

        hand_after = fmt_hand(next_obs["hand"]) if not done else []
        drawn = cards_added(
            [c for c in hand_before if c not in fmt_hand(
                [obs["hand"][i] for i in action.card_indices]
            )],
            hand_after,
        )

        step: dict[str, Any] = {
            "turn": len(current_blind_steps),
            "ante": ante,
            "blind_type": blind_type,
            "chips_scored": chips_before,
            "chips_needed": chips_needed,
            "chips_remaining": chips_needed - chips_before,
            "hands_left": hands_left_before,
            "discards_left": discards_left_before,
            "hand_before": hand_before,
            "action": {
                "type": action.type,
                "cards": fmt_hand(info["selected_cards"]),
            },
            "chips_gained": info["chips_gained"],
            "hand_type": info.get("hand_category"),
            "hand_after": hand_after,
            "drawn": drawn,
        }

        if info.get("round_result"):
            step["round_result"] = info["round_result"]
        if info.get("run_result"):
            step["run_result"] = info["run_result"]

        current_blind_steps.append(step)

        if info.get("next_round_started"):
            blinds_data.append({
                "ante": current_ante,
                "blind_type": current_blind_type,
                "chips_needed": chips_needed,
                "result": "round_win",
                "steps": current_blind_steps,
            })
            current_blind_steps = []
            current_ante = env.state.ante if env.state else current_ante
            current_blind_type = env.state.blind_type if env.state else current_blind_type

    # Close the final blind.
    final_result = info.get("run_result", "run_loss")  # type: ignore[possibly-undefined]
    blinds_data.append({
        "ante": current_ante,
        "blind_type": current_blind_type,
        "chips_needed": obs["chips_needed"],  # type: ignore[possibly-undefined]
        "result": info.get("round_result"),
        "steps": current_blind_steps,
    })

    if final_result != "run_win":
        return None

    return {
        "seed": seed,
        "max_ante": max_ante,
        "won": True,
        "blinds_passed": len([b for b in blinds_data if b.get("result") == "round_win"]),
        "final_chips": env.state.chips_scored if env.state else 0,
        "total_steps": sum(len(b["steps"]) for b in blinds_data),
        "blinds": blinds_data,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bot",
        choices=sorted(BOT_REGISTRY),
        default="PrunedSampledLookaheadBot",
        help="Which scripted bot to run.",
    )
    parser.add_argument(
        "--max-ante", type=int, default=1,
        help="Maximum ante to win the run (default 1 for trace generation).",
    )
    parser.add_argument(
        "--games", type=int, default=500,
        help="Total games to simulate (wins are a subset of this).",
    )
    parser.add_argument(
        "--base-seed", type=int, default=0,
        help="Starting seed; game i uses seed base_seed+i.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path. Auto-named if omitted.",
    )
    args = parser.parse_args()

    max_ante = args.max_ante
    bot_name = args.bot
    wins: list[dict[str, Any]] = []

    print(f"Running {args.games} games ({bot_name}, max_ante={max_ante}, base_seed={args.base_seed}) …")

    for game_index in range(args.games):
        seed = args.base_seed + game_index
        bot = BOT_REGISTRY[bot_name](random.Random(seed))
        trace = run_one_game(bot, seed=seed, max_ante=max_ante)
        if trace is not None:
            trace["game_index"] = game_index
            trace["bot"] = bot_name
            wins.append(trace)

        if (game_index + 1) % 100 == 0:
            print(f"  {game_index + 1}/{args.games} games — {len(wins)} wins so far")

    print(f"\nDone. {len(wins)} winning games out of {args.games} ({len(wins)/args.games:.1%}).")

    out_dir = REPO_ROOT / "results" / "traces" / "scripted_wins"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = args.output
    else:
        out_path = out_dir / f"{bot_name}_ante{max_ante}_seed{args.base_seed}_{len(wins)}wins.json"

    payload = {
        "bot": bot_name,
        "max_ante": max_ante,
        "num_games_run": args.games,
        "base_seed": args.base_seed,
        "num_wins": len(wins),
        "win_rate": len(wins) / args.games,
        "games": wins,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved to {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
