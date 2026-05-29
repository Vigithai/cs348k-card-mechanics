"""Generate rich per-step traces for RLQBot winning games.

Only winning games are saved. Each trace captures the full hand state before
every action, the action taken, chips gained, and the hand after redraw — giving
enough context to read a run and understand what the bot was doing and why.

Usage
-----
python scripts/generate_rl_win_traces.py \\
    --checkpoint results/rl/easy/seed_0/checkpoints/episode_0500.pt \\
    --preset easy \\
    --games 200 \\
    --base-seed 0

Output
------
results/traces/rl_wins/<preset>_seed<base_seed>_<N>wins.json
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
    ROUND_TARGET_PRESETS,
    RLQBot,
)
from balatro_mvp.rl import load_q_network_from_checkpoint

# ── Card formatting ────────────────────────────────────────────────────────────

SUIT_SYMBOL: dict[str, str] = {
    "club":    "♣",
    "diamond": "♦",
    "heart":   "♥",
    "spade":   "♠",
}


def fmt_card(card: Any) -> str:
    """Return a compact human-readable card string, e.g. 'A♥' or '10♠'."""
    # Accepts a Card dataclass or a plain dict (from serialised info dicts).
    if hasattr(card, "rank"):
        return f"{card.rank}{SUIT_SYMBOL[card.suit]}"
    return f"{card['rank']}{SUIT_SYMBOL[card['suit']]}"


def fmt_hand(cards: Any) -> list[str]:
    """Format a sequence of cards as a list of compact strings."""
    return [fmt_card(c) for c in cards]


# ── Trace helpers ──────────────────────────────────────────────────────────────

def cards_added(hand_before: list[str], hand_after: list[str]) -> list[str]:
    """Return cards that appear in hand_after but not hand_before (the redraw).

    Uses multiset subtraction so duplicates are handled correctly.
    """
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
    bot: RLQBot,
    seed: int,
    round_chip_targets: dict[int, int],
) -> dict[str, Any] | None:
    """Play one game; return a rich trace dict if it is a win, else None."""

    env = BalatroMVPEnvironment(seed=seed, round_chip_targets=round_chip_targets)
    done = False

    # We accumulate steps and split into rounds as we go.
    rounds_data: list[dict[str, Any]] = []
    current_round_steps: list[dict[str, Any]] = []
    current_round_index: int = 1

    while not done:
        obs = env.get_observation()
        legal_actions = env.get_legal_actions()

        # Snapshot state before the action.
        hand_before = fmt_hand(obs["hand"])
        chips_before = obs["chips_scored"]
        hands_left_before = obs["hands_left"]
        discards_left_before = obs["discards_left"]
        chips_needed = obs["chips_needed"]
        round_index = obs["round_index"]

        # Detect round boundary (env resets round_index on new round).
        if round_index != current_round_index:
            # Close out the previous round.
            rounds_data.append({
                "round_index": current_round_index,
                "chips_needed": chips_needed,   # will be overwritten below
                "steps": current_round_steps,
            })
            current_round_steps = []
            current_round_index = round_index

        action = bot.act(obs, legal_actions)
        next_obs, _, done, info = env.step(action)

        # Snapshot hand after the action (= after redraw if applicable).
        hand_after = fmt_hand(next_obs["hand"]) if not done else []
        drawn = cards_added(
            [c for c in hand_before if c not in fmt_hand(
                [obs["hand"][i] for i in action.card_indices]
            )],
            hand_after,
        )

        # Build the step record.
        step: dict[str, Any] = {
            "turn": obs.get("turn_index", len(current_round_steps)),  # fallback counter
            "round": round_index,
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
            "hand_type": info.get("hand_category"),   # None for discards
            "hand_after": hand_after,
            "drawn": drawn,
        }

        # Annotate round/run outcomes when they occur.
        if info.get("round_result"):
            step["round_result"] = info["round_result"]
        if info.get("run_result"):
            step["run_result"] = info["run_result"]

        current_round_steps.append(step)

        # If a new round started the env has already reset; next obs is round 2.
        if info.get("next_round_started"):
            # Finalise round 1 entry (chips_needed was round 1's target).
            rounds_data.append({
                "round_index": current_round_index,
                "chips_needed": chips_needed,
                "result": "round_win",
                "steps": current_round_steps,
            })
            current_round_steps = []
            current_round_index += 1

    # Close the final round.
    final_result = info.get("run_result", "run_loss")  # type: ignore[possibly-undefined]
    rounds_data.append({
        "round_index": current_round_index,
        "chips_needed": obs["chips_needed"],   # type: ignore[possibly-undefined]
        "result": info.get("round_result"),
        "steps": current_round_steps,
    })

    if final_result != "run_win":
        return None

    return {
        "seed": seed,
        "preset": None,    # filled in by caller
        "won": True,
        "rounds_passed": len([r for r in rounds_data if r.get("result") == "round_win"]),
        "final_chips": env.state.chips_scored if env.state else 0,
        "total_steps": sum(len(r["steps"]) for r in rounds_data),
        "rounds": rounds_data,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to an RLQBot .pt checkpoint file.",
    )
    parser.add_argument(
        "--preset", choices=sorted(ROUND_TARGET_PRESETS), default="easy",
        help="Named round-target preset.",
    )
    parser.add_argument(
        "--games", type=int, default=200,
        help="Total games to simulate (wins are a subset of this).",
    )
    parser.add_argument(
        "--base-seed", type=int, default=0,
        help="Starting seed; game i uses seed base_seed+i.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device for RLQBot inference.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path. Auto-named if omitted.",
    )
    args = parser.parse_args()

    round_chip_targets = dict(ROUND_TARGET_PRESETS[args.preset])
    rl_q_network, _ = load_q_network_from_checkpoint(args.checkpoint, device=args.device)

    wins: list[dict[str, Any]] = []

    print(f"Running {args.games} games (preset={args.preset}, base_seed={args.base_seed}) …")

    for game_index in range(args.games):
        seed = args.base_seed + game_index
        bot = RLQBot(rl_q_network, rng=random.Random(seed), epsilon=0.0, training=False, device=args.device)
        trace = run_one_game(bot, seed=seed, round_chip_targets=round_chip_targets)
        if trace is not None:
            trace["game_index"] = game_index
            trace["preset"] = args.preset
            wins.append(trace)

        if (game_index + 1) % 50 == 0:
            print(f"  {game_index + 1}/{args.games} games — {len(wins)} wins so far")

    print(f"\nDone. {len(wins)} winning games out of {args.games} ({len(wins)/args.games:.1%}).")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = REPO_ROOT / "results" / "traces" / "rl_wins"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = args.output
    else:
        out_path = out_dir / f"{args.preset}_seed{args.base_seed}_{len(wins)}wins.json"

    payload = {
        "checkpoint": str(args.checkpoint),
        "preset": args.preset,
        "round_chip_targets": round_chip_targets,
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
