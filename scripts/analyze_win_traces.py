"""Analyze winning game traces to surface strategy insights.

Answers four questions:
  1. What consecutive hand-type sequences appear most in winning games?
  2. How aggressive is the hunting strategy, and does it pay off?
  3. How does RLQBot differ from the best scripted bot in winning games?
  4. Do aggressive games (many discards) win with more margin than conservative ones?

Usage
-----
python scripts/analyze_win_traces.py \
    --rl   results/traces/rl_wins/ante_1_seed0_177wins.json \
    --scripted results/traces/scripted_wins/PrunedSampledLookaheadBot_ante1_seed0_84wins.json

Output
------
results/traces/analysis/insights.json          -- machine-readable summary
results/figures/analysis_*.png                 -- one figure per question
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"

PREMIUM_HANDS = {"flush", "straight", "full_house", "four_of_a_kind",
                 "straight_flush", "royal_flush"}
ALL_HAND_TYPES = [
    "high_card", "pair", "two_pair", "three_of_a_kind",
    "straight", "flush", "full_house", "four_of_a_kind",
    "straight_flush", "royal_flush",
]


# ── Trace loading ──────────────────────────────────────────────────────────────

def load_traces(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["games"]


def play_sequence(blind: dict[str, Any]) -> list[str]:
    """Return the ordered list of hand types played in one blind (discards excluded)."""
    return [
        s["hand_type"]
        for s in blind["steps"]
        if s["action"]["type"] == "play" and s["hand_type"]
    ]


def all_play_sequences(games: list[dict[str, Any]]) -> list[list[str]]:
    seqs = []
    for game in games:
        for blind in game["blinds"]:
            seq = play_sequence(blind)
            if seq:
                seqs.append(seq)
    return seqs


def consec_discards_before_play(blind: dict[str, Any]) -> list[dict[str, Any]]:
    """Return records of (n_discards, outcome_hand, chips_remaining_at_hunt_start)."""
    records = []
    steps = blind["steps"]
    i = 0
    while i < len(steps):
        if steps[i]["action"]["type"] == "discard":
            # Count consecutive discards
            j = i
            while j < len(steps) and steps[j]["action"]["type"] == "discard":
                j += 1
            n_discards = j - i
            chips_remaining = steps[i]["chips_remaining"]
            # Find the next play
            if j < len(steps) and steps[j]["action"]["type"] == "play":
                outcome = steps[j]["hand_type"] or "unknown"
                records.append({
                    "n_discards": n_discards,
                    "outcome_hand": outcome,
                    "chips_remaining": chips_remaining,
                    "chips_needed": blind["chips_needed"],
                    "urgency": chips_remaining / blind["chips_needed"],
                })
            i = j
        else:
            i += 1
    return records


def win_margin(game: dict[str, Any]) -> int:
    """Chips scored above the final blind's target (proxy for how comfortable the win was)."""
    last_blind = game["blinds"][-1]
    return game["final_chips"] - last_blind["chips_needed"]


def discard_rate(game: dict[str, Any]) -> float:
    """Fraction of all actions that were discards."""
    all_steps = [s for b in game["blinds"] for s in b["steps"]]
    if not all_steps:
        return 0.0
    return sum(1 for s in all_steps if s["action"]["type"] == "discard") / len(all_steps)


def premium_hand_rate(game: dict[str, Any]) -> float:
    """Fraction of played hands that are premium (flush, straight, full house, etc.)."""
    plays = [s for b in game["blinds"] for s in b["steps"] if s["action"]["type"] == "play"]
    if not plays:
        return 0.0
    return sum(1 for s in plays if s["hand_type"] in PREMIUM_HANDS) / len(plays)


# ── Q1: Hand sequence n-grams ─────────────────────────────────────────────────

def q1_ngrams(games: list[dict[str, Any]], n: int = 2) -> dict[str, int]:
    counts: Counter = Counter()
    for seq in all_play_sequences(games):
        for i in range(len(seq) - n + 1):
            counts[" → ".join(seq[i:i + n])] += 1
    return dict(counts.most_common(12))


# ── Q2: Hunt aggressiveness ───────────────────────────────────────────────────

def q2_hunt_aggressiveness(games: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for game in games:
        for blind in game["blinds"]:
            records.extend(consec_discards_before_play(blind))

    if not records:
        return {"hunt_count": 0, "by_n_discards": {}, "outcome_distribution": {}}

    by_n: Counter = Counter(r["n_discards"] for r in records)
    outcome_given_hunt: Counter = Counter(
        r["outcome_hand"] for r in records if r["n_discards"] >= 2
    )
    # Comfortable (urgency < 0.5) vs desperate (urgency >= 0.5) hunt outcomes
    comfortable = [r for r in records if r["n_discards"] >= 2 and r["urgency"] < 0.5]
    desperate = [r for r in records if r["n_discards"] >= 2 and r["urgency"] >= 0.5]
    comfortable_premium = (
        sum(1 for r in comfortable if r["outcome_hand"] in PREMIUM_HANDS) / len(comfortable)
        if comfortable else 0.0
    )
    desperate_premium = (
        sum(1 for r in desperate if r["outcome_hand"] in PREMIUM_HANDS) / len(desperate)
        if desperate else 0.0
    )

    return {
        "hunt_count": len(records),
        "multi_discard_hunts": len([r for r in records if r["n_discards"] >= 2]),
        "by_n_discards": {str(k): v for k, v in sorted(by_n.items())},
        "outcome_given_multi_hunt": dict(outcome_given_hunt.most_common()),
        "comfortable_premium_rate": round(comfortable_premium, 3),
        "desperate_premium_rate": round(desperate_premium, 3),
    }


# ── Q3: Bot comparison ────────────────────────────────────────────────────────

def q3_bot_comparison(
    rl_games: list[dict[str, Any]],
    scripted_games: list[dict[str, Any]],
) -> dict[str, Any]:
    def stats(games: list[dict[str, Any]]) -> dict[str, Any]:
        margins = [win_margin(g) for g in games]
        disc_rates = [discard_rate(g) for g in games]
        prem_rates = [premium_hand_rate(g) for g in games]
        return {
            "n_wins": len(games),
            "avg_win_margin": round(sum(margins) / len(margins), 1),
            "avg_discard_rate": round(sum(disc_rates) / len(disc_rates), 3),
            "avg_premium_hand_rate": round(sum(prem_rates) / len(prem_rates), 3),
            "hand_type_totals": dict(
                Counter(
                    s["hand_type"]
                    for g in games
                    for b in g["blinds"]
                    for s in b["steps"]
                    if s["action"]["type"] == "play" and s["hand_type"]
                ).most_common()
            ),
        }

    return {
        "RLQBot": stats(rl_games),
        "PrunedSampledLookaheadBot": stats(scripted_games),
    }


# ── Q4: Conservative vs aggressive ───────────────────────────────────────────

def q4_conservative_vs_aggressive(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Split games by total discards per blind and compare win margin."""
    conservative, aggressive = [], []
    for game in games:
        total_discards = sum(
            1 for b in game["blinds"] for s in b["steps"]
            if s["action"]["type"] == "discard"
        )
        total_blinds = len(game["blinds"])
        avg_discards_per_blind = total_discards / total_blinds if total_blinds else 0
        if avg_discards_per_blind <= 1:
            conservative.append(game)
        else:
            aggressive.append(game)

    def avg_margin(gs: list) -> float:
        return sum(win_margin(g) for g in gs) / len(gs) if gs else 0.0

    def avg_premium(gs: list) -> float:
        rates = [premium_hand_rate(g) for g in gs]
        return sum(rates) / len(rates) if rates else 0.0

    return {
        "conservative_count": len(conservative),
        "aggressive_count": len(aggressive),
        "conservative_avg_margin": round(avg_margin(conservative), 1),
        "aggressive_avg_margin": round(avg_margin(aggressive), 1),
        "conservative_avg_premium_rate": round(avg_premium(conservative), 3),
        "aggressive_avg_premium_rate": round(avg_premium(aggressive), 3),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def setup_mpl(figures_dir: Path) -> None:
    mpl_config = figures_dir.parent / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    # Match the Dark Botanical presentation palette
    matplotlib.rcParams.update({
        "figure.facecolor":  "#0f0f0f",
        "axes.facecolor":    "#161616",
        "axes.edgecolor":    "#2a2a2a",
        "axes.labelcolor":   "#9a9590",
        "xtick.color":       "#9a9590",
        "ytick.color":       "#9a9590",
        "text.color":        "#e8e4df",
        "grid.color":        "#2a2a2a",
        "grid.alpha":        0.4,
        "font.family":       "sans-serif",
    })


def plot_q1_ngrams(
    rl_ngrams: dict[str, int],
    scripted_ngrams: dict[str, int],
    figures_dir: Path,
) -> Path:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np

    # Top 8 by RL count
    top_keys = list(rl_ngrams.keys())[:8]
    rl_vals = [rl_ngrams.get(k, 0) for k in top_keys]
    sc_vals = [scripted_ngrams.get(k, 0) for k in top_keys]

    x = np.arange(len(top_keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, rl_vals, w, label="RLQBot", color="#d4a574")
    ax.bar(x + w / 2, sc_vals, w, label="PrunedSampledLookahead", color="#6a8fad")
    ax.set_xticks(x)
    ax.set_xticklabels(top_keys, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Frequency across winning games")
    ax.set_title("Q1 · Most common consecutive hand-type pairs in winning games")
    ax.legend()
    fig.tight_layout()
    out = figures_dir / "analysis_q1_ngrams.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_q2_hunt(hunt_data: dict[str, Any], figures_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    outcome_counts = hunt_data.get("outcome_given_multi_hunt", {})
    if not outcome_counts:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No multi-discard hunts found", ha="center")
        out = figures_dir / "analysis_q2_hunt.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        return out

    # Bar chart: outcome distribution after 2+ discards
    labels = list(outcome_counts.keys())
    values = list(outcome_counts.values())
    colors = ["#d4a574" if h in PREMIUM_HANDS else "#8a8a8a" for h in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(range(len(labels)), values, color=colors)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Hunt count")
    axes[0].set_title("What the bot hunts for\n(outcome after 2+ consecutive discards)")

    # Comfortable vs desperate hunt success
    cats = ["Comfortable\n(>50% chips left)", "Desperate\n(≤50% chips left)"]
    rates = [
        hunt_data["comfortable_premium_rate"] * 100,
        hunt_data["desperate_premium_rate"] * 100,
    ]
    axes[1].bar(cats, rates, color=["#d4a574", "#c0655a"])
    axes[1].set_ylabel("Premium hand rate after hunt (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Does pressure affect hunt success?")
    for i, v in enumerate(rates):
        axes[1].text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=11)

    fig.suptitle("Q2 · Hunt aggressiveness in winning games (RLQBot)", fontsize=12)
    fig.tight_layout()
    out = figures_dir / "analysis_q2_hunt.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_q3_comparison(comparison: dict[str, Any], figures_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    bots = list(comparison.keys())
    metrics = {
        "Avg win margin (chips)": [comparison[b]["avg_win_margin"] for b in bots],
        "Discard rate (%)": [comparison[b]["avg_discard_rate"] * 100 for b in bots],
        "Premium hand rate (%)": [comparison[b]["avg_premium_hand_rate"] * 100 for b in bots],
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    colors = ["#d4a574", "#6a8fad"]
    for ax, (label, vals) in zip(axes, metrics.items()):
        bars = ax.bar(range(len(bots)), vals, color=colors)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(len(bots)))
        ax.set_xticklabels(bots, rotation=15, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val:.1f}", ha="center", fontsize=10)

    fig.suptitle("Q3 · How do RL and scripted bots win differently?", fontsize=12)
    fig.tight_layout()
    out = figures_dir / "analysis_q3_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_q4_conservative_vs_aggressive(q4: dict[str, Any], figures_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    cats = [
        f"Conservative\n(≤1 discard/blind)\nn={q4['conservative_count']}",
        f"Aggressive\n(>1 discard/blind)\nn={q4['aggressive_count']}",
    ]
    margins = [q4["conservative_avg_margin"], q4["aggressive_avg_margin"]]
    premiums = [
        q4["conservative_avg_premium_rate"] * 100,
        q4["aggressive_avg_premium_rate"] * 100,
    ]
    colors = ["#6a8fad", "#d4a574"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, vals, label in zip(
        axes,
        [margins, premiums],
        ["Avg win margin (chips)", "Premium hand rate (%)"],
    ):
        bars = ax.bar(cats, vals, color=colors)
        ax.set_title(label)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val:.1f}", ha="center", fontsize=11)

    fig.suptitle("Q4 · Does aggressive discarding pay off in winning games?", fontsize=12)
    fig.tight_layout()
    out = figures_dir / "analysis_q4_conservative_vs_aggressive.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rl", type=Path,
        default=REPO_ROOT / "results" / "traces" / "rl_wins" / "ante_1_seed0_177wins.json",
        help="RLQBot winning traces JSON.",
    )
    parser.add_argument(
        "--scripted", type=Path,
        default=REPO_ROOT / "results" / "traces" / "scripted_wins" /
                "PrunedSampledLookaheadBot_ante1_seed0_84wins.json",
        help="Scripted bot winning traces JSON.",
    )
    parser.add_argument(
        "--figures-dir", type=Path,
        default=REPO_ROOT / "results" / "figures",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=REPO_ROOT / "results" / "traces" / "analysis" / "insights.json",
        help="Path for machine-readable insights JSON.",
    )
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    setup_mpl(args.figures_dir)

    print("Loading traces …")
    rl_games = load_traces(args.rl)
    scripted_games = load_traces(args.scripted)
    print(f"  RLQBot:   {len(rl_games)} winning games")
    print(f"  Scripted: {len(scripted_games)} winning games")

    print("\nQ1 — Hand sequence n-grams …")
    rl_ngrams = q1_ngrams(rl_games, n=2)
    sc_ngrams = q1_ngrams(scripted_games, n=2)

    print("Q2 — Hunt aggressiveness (RL) …")
    hunt = q2_hunt_aggressiveness(rl_games)

    print("Q3 — Bot comparison …")
    comparison = q3_bot_comparison(rl_games, scripted_games)

    print("Q4 — Conservative vs aggressive (RL) …")
    q4 = q4_conservative_vs_aggressive(rl_games)

    # ── Save insights JSON ────────────────────────────────────────────────────
    insights = {
        "q1_top_ngrams_rl": rl_ngrams,
        "q1_top_ngrams_scripted": sc_ngrams,
        "q2_hunt_aggressiveness": hunt,
        "q3_bot_comparison": comparison,
        "q4_conservative_vs_aggressive": q4,
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)
    print(f"\nSaved insights to {args.output_json}")

    # ── Print key numbers ─────────────────────────────────────────────────────
    print("\n── Key insights ─────────────────────────────────────────────────")
    print(f"  Top RL bigram:      {list(rl_ngrams.items())[0]}")
    print(f"  Top scripted bigram:{list(sc_ngrams.items())[0]}")
    print(f"  Multi-discard hunts:{hunt['multi_discard_hunts']} total")
    print(f"  Comfortable premium rate after hunt: {hunt['comfortable_premium_rate']:.0%}")
    print(f"  Desperate  premium rate after hunt:  {hunt['desperate_premium_rate']:.0%}")
    rl_s = comparison["RLQBot"]
    sc_s = comparison["PrunedSampledLookaheadBot"]
    print(f"  RL  avg win margin: {rl_s['avg_win_margin']}  discard rate: {rl_s['avg_discard_rate']:.0%}  premium rate: {rl_s['avg_premium_hand_rate']:.0%}")
    print(f"  Sc  avg win margin: {sc_s['avg_win_margin']}  discard rate: {sc_s['avg_discard_rate']:.0%}  premium rate: {sc_s['avg_premium_hand_rate']:.0%}")
    print(f"  Conservative wins avg margin: {q4['conservative_avg_margin']}")
    print(f"  Aggressive   wins avg margin: {q4['aggressive_avg_margin']}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    paths = [
        plot_q1_ngrams(rl_ngrams, sc_ngrams, args.figures_dir),
        plot_q2_hunt(hunt, args.figures_dir),
        plot_q3_comparison(comparison, args.figures_dir),
        plot_q4_conservative_vs_aggressive(q4, args.figures_dir),
    ]
    for p in paths:
        print(f"  Saved {p}")


if __name__ == "__main__":
    main()
