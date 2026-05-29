"""Analyze RLQBot win traces: score games for instructiveness and emit an HTML report.

Reads the JSON produced by generate_rl_win_traces.py and:
  1. Detects strategy patterns in each game (flush hunt, straight hunt, etc.)
  2. Scores each game for human instructiveness
  3. Prints a ranked summary table to stdout
  4. Saves a standalone HTML report with fully annotated trace views

Usage
-----
python scripts/analyze_rl_traces.py results/traces/rl_wins/easy_seed0_78wins.json
python scripts/analyze_rl_traces.py results/traces/rl_wins/easy_seed0_78wins.json --top 15
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

PREMIUM_HANDS = {"flush", "straight", "full_house", "four_of_a_kind", "straight_flush", "royal_flush"}
STRONG_HANDS   = {"three_of_a_kind", "two_pair"}

HAND_SCORES: dict[str, int] = {
    "high_card":      5,
    "pair":          20,
    "two_pair":      40,
    "three_of_a_kind": 90,
    "straight":     120,
    "flush":        140,
    "full_house":   160,
    "four_of_a_kind": 420,
    "straight_flush": 800,
    "royal_flush":   800,
}

SUIT_SYMBOL = {"club": "♣", "diamond": "♦", "heart": "♥", "spade": "♠"}

# ── Pattern detection ─────────────────────────────────────────────────────────

@dataclass
class Pattern:
    """A detected strategic pattern within one round."""
    kind: str          # "flush_hunt", "straight_hunt", "efficient_close", "premium_hand", etc.
    label: str         # Short human-readable label
    detail: str        # One-sentence explanation
    step_indices: list[int] = field(default_factory=list)  # Which steps are involved
    score: int = 0     # Points added to the game's instructiveness score


def detect_patterns(round_steps: list[dict[str, Any]]) -> list[Pattern]:
    """Detect strategy patterns within a single round's steps."""
    patterns: list[Pattern] = []

    # Walk the steps and look for discard sequences followed by a strong play.
    i = 0
    while i < len(round_steps):
        s = round_steps[i]

        if s["action"]["type"] == "discard":
            # Collect consecutive discards starting here.
            discard_run = [i]
            j = i + 1
            while j < len(round_steps) and round_steps[j]["action"]["type"] == "discard":
                discard_run.append(j)
                j += 1

            # Check what comes after the run.
            if j < len(round_steps):
                play_step = round_steps[j]
                hand_type = play_step.get("hand_type")
                n_discards = len(discard_run)

                if n_discards >= 2:
                    if hand_type in ("flush", "straight_flush", "royal_flush"):
                        patterns.append(Pattern(
                            kind="flush_hunt",
                            label="🎴 Flush hunt",
                            detail=f"Discarded {n_discards}× to build a {hand_type} "
                                   f"(+{play_step['chips_gained']} chips).",
                            step_indices=discard_run + [j],
                            score=4 + n_discards,
                        ))
                    elif hand_type == "straight":
                        patterns.append(Pattern(
                            kind="straight_hunt",
                            label="➡ Straight hunt",
                            detail=f"Discarded {n_discards}× to build a straight "
                                   f"(+{play_step['chips_gained']} chips).",
                            step_indices=discard_run + [j],
                            score=3 + n_discards,
                        ))
                    elif hand_type == "full_house":
                        patterns.append(Pattern(
                            kind="full_house_build",
                            label="🏠 Full house build",
                            detail=f"Discarded {n_discards}× then scored a full house "
                                   f"(+{play_step['chips_gained']} chips).",
                            step_indices=discard_run + [j],
                            score=3,
                        ))
                    elif hand_type == "four_of_a_kind":
                        patterns.append(Pattern(
                            kind="quads_hunt",
                            label="⚡ Quads!",
                            detail=f"Discarded {n_discards}× and landed four of a kind "
                                   f"(+{play_step['chips_gained']} chips).",
                            step_indices=discard_run + [j],
                            score=6,
                        ))

            # Advance past the entire discard run.
            i = j
        else:
            # A play step — note premium hands even without a setup.
            hand_type = s.get("hand_type")
            if hand_type in PREMIUM_HANDS:
                already_flagged = any(
                    s_idx in p.step_indices for p in patterns for s_idx in [i]
                )
                if not already_flagged:
                    patterns.append(Pattern(
                        kind="premium_hand",
                        label=f"✨ {hand_type.replace('_', ' ').title()}",
                        detail=f"Scored a {hand_type.replace('_', ' ')} without setup discards "
                               f"(+{s['chips_gained']} chips).",
                        step_indices=[i],
                        score=2,
                    ))
            i += 1

    return patterns


def score_game(game: dict[str, Any]) -> tuple[int, list[Pattern]]:
    """Compute total instructiveness score and all detected patterns for one game."""
    all_patterns: list[Pattern] = []
    score = 0

    for r in game["rounds"]:
        steps = r["steps"]
        patterns = detect_patterns(steps)
        all_patterns.extend(patterns)
        score += sum(p.score for p in patterns)

    # Efficiency bonuses.
    total_steps = game["total_steps"]
    if total_steps <= 8:
        score += 3
        all_patterns.append(Pattern(
            kind="speed_run",
            label="⚡ Speed run",
            detail=f"Won in just {total_steps} steps.",
            score=3,
        ))
    elif total_steps <= 10:
        score += 1

    # Variety bonus: used at least 3 different hand types.
    hand_types_used = {
        s["hand_type"]
        for r in game["rounds"]
        for s in r["steps"]
        if s["hand_type"]
    }
    if len(hand_types_used) >= 3:
        score += 1

    return score, all_patterns


# ── Formatting helpers ────────────────────────────────────────────────────────

def hand_tag(hand_type: str | None) -> str:
    """HTML badge for a hand type."""
    if not hand_type:
        return ""
    premium = hand_type in PREMIUM_HANDS
    cls = "tag-premium" if premium else ("tag-strong" if hand_type in STRONG_HANDS else "tag-plain")
    label = hand_type.replace("_", " ")
    return f'<span class="tag {cls}">{label}</span>'


def cards_html(cards: list[str]) -> str:
    """Render a list of card strings as coloured HTML spans."""
    parts = []
    for c in cards:
        suit_char = c[-1] if c else ""
        red = suit_char in ("♥", "♦")
        cls = "card-red" if red else "card-black"
        parts.append(f'<span class="{cls}">{c}</span>')
    return " ".join(parts)


def render_step(step: dict[str, Any], highlighted: bool, pattern_label: str) -> str:
    """Render one step as an HTML table row."""
    action = step["action"]
    action_type = action["type"]
    action_cls = "action-play" if action_type == "play" else "action-discard"

    chips_after = step["chips_scored"] + step["chips_gained"]
    chips_needed = step["chips_needed"]
    pct = min(chips_after / chips_needed, 1.0) if chips_needed > 0 else 0

    highlight_cls = " row-highlight" if highlighted else ""
    pattern_cell = f'<td class="pattern-cell">{pattern_label}</td>' if pattern_label else "<td></td>"

    drawn_html = ""
    if step.get("drawn"):
        drawn_html = f'<div class="drawn">drew: {cards_html(step["drawn"])}</div>'

    return f"""
    <tr class="step-row{highlight_cls}">
      <td class="turn-cell">{step['turn']}</td>
      <td class="action-cell"><span class="{action_cls}">{action_type}</span></td>
      <td class="cards-cell">
        {cards_html(action['cards'])}
        {drawn_html}
      </td>
      <td class="hand-type-cell">{hand_tag(step.get('hand_type'))}</td>
      <td class="chips-cell">
        <span class="chips-gained">+{step['chips_gained']}</span>
        <div class="progress-wrap">
          <div class="progress-bar" style="width:{pct*100:.0f}%"></div>
        </div>
        <span class="chips-total">{chips_after}/{chips_needed}</span>
      </td>
      {pattern_cell}
    </tr>"""


def render_game(rank: int, game: dict[str, Any], score: int, patterns: list[Pattern]) -> str:
    """Render one game as a full HTML section."""
    # Build a step-index → pattern-label map for highlighting.
    step_highlight: dict[tuple[int, int], str] = {}  # (round_idx, step_idx) → label
    for p in patterns:
        if p.kind in ("speed_run",):
            continue
        for si in p.step_indices:
            for ri, r in enumerate(game["rounds"]):
                if si < len(r["steps"]):
                    step_highlight[(ri, si)] = p.label

    pattern_pills = "".join(
        f'<span class="pattern-pill">{p.label} <em>{p.detail}</em></span>'
        for p in patterns
        if p.kind not in ("speed_run",)
    )
    if not pattern_pills:
        pattern_pills = '<span class="pattern-pill plain">No dominant pattern detected</span>'

    speed_note = ""
    for p in patterns:
        if p.kind == "speed_run":
            speed_note = f'<span class="speed-badge">⚡ {p.detail}</span>'

    rounds_html = ""
    for ri, r in enumerate(game["rounds"]):
        result_cls = "result-win" if r.get("result") == "round_win" else "result-loss"
        result_label = "WIN" if r.get("result") == "round_win" else "LOSS"
        steps_html = "".join(
            render_step(
                step=s,
                highlighted=(ri, si) in step_highlight,
                pattern_label=step_highlight.get((ri, si), ""),
            )
            for si, s in enumerate(r["steps"])
        )
        rounds_html += f"""
        <div class="round-block">
          <div class="round-header">
            Round {r['round_index']} &nbsp;·&nbsp; target: {r['chips_needed']} chips
            <span class="{result_cls}">{result_label}</span>
          </div>
          <table class="step-table">
            <thead>
              <tr>
                <th>Turn</th><th>Action</th><th>Cards</th>
                <th>Hand</th><th>Chips</th><th>Pattern</th>
              </tr>
            </thead>
            <tbody>{steps_html}</tbody>
          </table>
        </div>"""

    return f"""
    <div class="game-card" id="game-{game['game_index']}">
      <div class="game-header">
        <span class="game-rank">#{rank}</span>
        <span class="game-title">Game {game['game_index']} &nbsp;·&nbsp; seed {game['seed']}</span>
        <span class="game-meta">{game['total_steps']} steps · {game['final_chips']} final chips</span>
        <span class="game-score">score {score}</span>
        {speed_note}
      </div>
      <div class="pattern-row">{pattern_pills}</div>
      {rounds_html}
    </div>"""


# ── HTML template ─────────────────────────────────────────────────────────────

HTML_STYLE = """
<style>
  :root {
    --bg: #0f0f0f; --bg2: #161616; --bg3: #1e1e1e;
    --text: #e8e4df; --text2: #9a9590; --text3: #555;
    --gold: #d4a574; --pink: #e8b4b8; --green: #7ec8a0;
    --red: #e07070;
    --font: 'IBM Plex Sans', sans-serif;
    --serif: 'Cormorant', serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font);
         font-size: 14px; line-height: 1.5; }
  a { color: var(--gold); text-decoration: none; }

  /* Layout */
  .page-wrap { max-width: 1100px; margin: 0 auto; padding: 40px 24px 80px; }

  /* Header */
  .page-header { margin-bottom: 40px; border-bottom: 1px solid rgba(212,165,116,0.15); padding-bottom: 24px; }
  .page-header h1 { font-family: var(--serif); font-size: 2.4rem; font-weight: 400;
                    color: var(--text); letter-spacing: -0.01em; }
  .page-header h1 em { font-style: italic; color: var(--gold); }
  .page-header p { color: var(--text2); font-size: 0.85rem; margin-top: 6px; font-weight: 300; }

  /* Summary stats bar */
  .stats-bar { display: flex; gap: 32px; margin-bottom: 36px; flex-wrap: wrap; }
  .stat-box { background: var(--bg2); border: 1px solid rgba(232,228,223,0.07);
               border-radius: 4px; padding: 14px 20px; min-width: 120px; }
  .stat-val { font-family: var(--serif); font-size: 1.8rem; font-weight: 600; color: var(--text); line-height: 1; }
  .stat-val em { color: var(--gold); font-style: normal; font-size: 0.65em; vertical-align: super; }
  .stat-lbl { font-size: 0.65rem; font-weight: 300; letter-spacing: 0.15em;
               text-transform: uppercase; color: var(--text2); margin-top: 4px; }

  /* Pattern frequency table */
  .pattern-freq { margin-bottom: 40px; }
  .pattern-freq h2 { font-family: var(--serif); font-size: 1.3rem; font-weight: 400;
                      color: var(--text); margin-bottom: 12px; }
  .freq-table { border-collapse: collapse; width: 100%; }
  .freq-table th { text-align: left; font-size: 0.65rem; letter-spacing: 0.15em;
                    text-transform: uppercase; color: var(--text2); font-weight: 300;
                    padding: 6px 12px; border-bottom: 1px solid rgba(232,228,223,0.07); }
  .freq-table td { padding: 7px 12px; border-bottom: 1px solid rgba(232,228,223,0.04);
                    font-size: 0.82rem; color: var(--text2); }
  .freq-table td:first-child { color: var(--text); }
  .freq-bar-cell { width: 200px; }
  .freq-bar-wrap { background: rgba(232,228,223,0.05); border-radius: 2px; height: 4px; }
  .freq-bar-fill { background: var(--gold); border-radius: 2px; height: 4px; }

  /* Index */
  .index-section { margin-bottom: 40px; }
  .index-section h2 { font-family: var(--serif); font-size: 1.3rem; font-weight: 400;
                        color: var(--text); margin-bottom: 12px; }
  .index-grid { display: flex; flex-direction: column; gap: 4px; }
  .index-row { display: flex; gap: 12px; align-items: baseline; padding: 5px 0;
                border-bottom: 1px solid rgba(232,228,223,0.04); font-size: 0.82rem; }
  .index-rank { color: var(--gold); font-family: var(--serif); font-style: italic;
                 font-size: 1rem; min-width: 28px; }
  .index-link { color: var(--text); }
  .index-link:hover { color: var(--gold); }
  .index-pills { display: flex; gap: 6px; flex-wrap: wrap; }
  .index-score { margin-left: auto; color: var(--text3); font-size: 0.75rem; }

  /* Game cards */
  .game-card { background: var(--bg2); border: 1px solid rgba(232,228,223,0.07);
                border-radius: 6px; margin-bottom: 32px; overflow: hidden; }
  .game-header { display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
                  padding: 14px 20px; border-bottom: 1px solid rgba(232,228,223,0.07);
                  background: var(--bg3); }
  .game-rank { font-family: var(--serif); font-style: italic; font-size: 1.3rem;
                color: var(--gold); min-width: 28px; }
  .game-title { font-weight: 400; color: var(--text); }
  .game-meta { color: var(--text2); font-size: 0.8rem; }
  .game-score { margin-left: auto; font-size: 0.75rem; color: var(--text3);
                 border: 1px solid rgba(232,228,223,0.1); border-radius: 12px; padding: 2px 10px; }
  .speed-badge { font-size: 0.75rem; color: var(--gold); }

  /* Pattern pills */
  .pattern-row { padding: 10px 20px; display: flex; gap: 8px; flex-wrap: wrap;
                  border-bottom: 1px solid rgba(232,228,223,0.06); }
  .pattern-pill { font-size: 0.75rem; padding: 3px 10px; border-radius: 12px;
                   background: rgba(212,165,116,0.1); border: 1px solid rgba(212,165,116,0.2);
                   color: var(--text2); }
  .pattern-pill em { font-style: normal; color: var(--text3); margin-left: 4px; }
  .pattern-pill.plain { background: rgba(232,228,223,0.04); border-color: rgba(232,228,223,0.08); }

  /* Round blocks */
  .round-block { padding: 16px 20px 12px; }
  .round-block + .round-block { border-top: 1px solid rgba(232,228,223,0.06); }
  .round-header { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
                   color: var(--text2); margin-bottom: 10px; display: flex; align-items: center; gap: 10px; }
  .result-win  { color: var(--green); }
  .result-loss { color: var(--red); }

  /* Step table */
  .step-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .step-table th { text-align: left; font-size: 0.62rem; letter-spacing: 0.14em;
                    text-transform: uppercase; color: var(--text3); font-weight: 300;
                    padding: 4px 8px; border-bottom: 1px solid rgba(232,228,223,0.06); }
  .step-row td { padding: 7px 8px; border-bottom: 1px solid rgba(232,228,223,0.04);
                  vertical-align: top; }
  .step-row.row-highlight { background: rgba(212,165,116,0.05); }
  .step-row.row-highlight td { border-bottom-color: rgba(212,165,116,0.08); }

  .turn-cell    { color: var(--text3); width: 28px; }
  .action-cell  { width: 64px; }
  .cards-cell   { }
  .hand-type-cell { width: 120px; }
  .chips-cell   { width: 140px; }
  .pattern-cell { font-size: 0.72rem; color: var(--gold); white-space: nowrap; }

  .action-play    { color: var(--green); font-weight: 400; }
  .action-discard { color: var(--text2); }

  .card-red   { color: #e07070; font-weight: 400; }
  .card-black { color: var(--text); font-weight: 400; }

  .drawn { font-size: 0.72rem; color: var(--text3); margin-top: 3px; }

  /* Hand type tags */
  .tag { font-size: 0.68rem; padding: 2px 7px; border-radius: 10px; white-space: nowrap; }
  .tag-premium { background: rgba(212,165,116,0.15); border: 1px solid rgba(212,165,116,0.25); color: var(--gold); }
  .tag-strong  { background: rgba(126,200,160,0.1);  border: 1px solid rgba(126,200,160,0.2);  color: var(--green); }
  .tag-plain   { background: rgba(232,228,223,0.06); border: 1px solid rgba(232,228,223,0.1);  color: var(--text2); }

  /* Chip progress */
  .chips-gained  { color: var(--green); font-weight: 400; }
  .progress-wrap { background: rgba(232,228,223,0.06); border-radius: 2px; height: 3px;
                    margin: 3px 0; width: 100%; }
  .progress-bar  { background: var(--gold); border-radius: 2px; height: 3px; }
  .chips-total   { font-size: 0.7rem; color: var(--text3); }

  /* Hand before */
  .hand-before { font-size: 0.72rem; color: var(--text3); margin-bottom: 2px; }
</style>
<link href="https://fonts.googleapis.com/css2?family=Cormorant:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
"""


def build_html(
    d: dict[str, Any],
    ranked_games: list[tuple[int, dict[str, Any], list[Pattern]]],
    top_n: int,
    pattern_freq: dict[str, int],
) -> str:
    """Build the full HTML report."""

    total_patterns = sum(pattern_freq.values())
    max_freq = max(pattern_freq.values()) if pattern_freq else 1

    freq_rows = ""
    for kind, count in sorted(pattern_freq.items(), key=lambda x: -x[1]):
        pct = count / max_freq * 100
        freq_rows += f"""
        <tr>
          <td>{kind.replace('_', ' ')}</td>
          <td>{count}</td>
          <td>{count / d['num_wins']:.0%} of wins</td>
          <td class="freq-bar-cell">
            <div class="freq-bar-wrap"><div class="freq-bar-fill" style="width:{pct:.0f}%"></div></div>
          </td>
        </tr>"""

    # Index rows
    index_rows = ""
    for rank, (score, game, patterns) in enumerate(ranked_games[:top_n], 1):
        pills = " ".join(
            f'<span class="tag tag-premium">{p.label}</span>'
            for p in patterns if p.kind not in ("speed_run",)
        )
        index_rows += f"""
        <div class="index-row">
          <span class="index-rank">{rank}</span>
          <a href="#game-{game['game_index']}" class="index-link">
            Game {game['game_index']} · seed {game['seed']} · {game['total_steps']} steps
          </a>
          <span class="index-pills">{pills}</span>
          <span class="index-score">score {score}</span>
        </div>"""

    # Game sections
    game_sections = ""
    for rank, (score, game, patterns) in enumerate(ranked_games[:top_n], 1):
        game_sections += render_game(rank, game, score, patterns)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RLQBot Win Trace Analysis — {d['preset']} preset</title>
{HTML_STYLE}
</head>
<body>
<div class="page-wrap">

  <div class="page-header">
    <h1>RLQBot Win Trace Analysis — <em>{d['preset']}</em></h1>
    <p>
      {d['num_wins']} winning games · {d['num_games_run']} total games · win rate {d['win_rate']:.1%} ·
      checkpoint: {Path(d['checkpoint']).name} ·
      base seed {d['base_seed']}
    </p>
  </div>

  <div class="stats-bar">
    <div class="stat-box">
      <div class="stat-val">{d['num_wins']}</div>
      <div class="stat-lbl">Winning games</div>
    </div>
    <div class="stat-box">
      <div class="stat-val">{d['win_rate']:.0%}</div>
      <div class="stat-lbl">Win rate</div>
    </div>
    <div class="stat-box">
      <div class="stat-val">{total_patterns}</div>
      <div class="stat-lbl">Patterns detected</div>
    </div>
    <div class="stat-box">
      <div class="stat-val">{top_n}</div>
      <div class="stat-lbl">Games shown (top)</div>
    </div>
  </div>

  <div class="pattern-freq">
    <h2>Strategy pattern frequency</h2>
    <table class="freq-table">
      <thead><tr><th>Pattern</th><th>Count</th><th>% of wins</th><th>Frequency</th></tr></thead>
      <tbody>{freq_rows}</tbody>
    </table>
  </div>

  <div class="index-section">
    <h2>Top {top_n} most instructive games</h2>
    <div class="index-grid">{index_rows}</div>
  </div>

  {game_sections}

</div>
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, help="Path to the win traces JSON file.")
    parser.add_argument("--top", type=int, default=10, help="How many top games to show in the report.")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (auto-named if omitted).")
    args = parser.parse_args()

    with args.traces.open(encoding="utf-8") as f:
        d = json.load(f)

    games = d["games"]

    # Score and rank all games.
    scored: list[tuple[int, dict[str, Any], list[Pattern]]] = []
    pattern_freq: dict[str, int] = {}

    for game in games:
        score, patterns = score_game(game)
        scored.append((score, game, patterns))
        for p in patterns:
            if p.kind not in ("speed_run",):
                pattern_freq[p.kind] = pattern_freq.get(p.kind, 0) + 1

    ranked = sorted(scored, key=lambda x: -x[0])

    # ── Stdout summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RLQBot trace analysis — {d['preset']} preset")
    print(f"  {d['num_wins']} wins / {d['num_games_run']} games ({d['win_rate']:.1%})")
    print(f"{'='*60}\n")

    print("Strategy pattern frequency:")
    for kind, count in sorted(pattern_freq.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {kind:<22} {count:3d}  {bar}")

    print(f"\nTop {args.top} most instructive games:")
    print(f"  {'Rank':<5} {'Game':>5} {'Seed':>6} {'Steps':>6} {'Score':>6}  Patterns")
    print(f"  {'-'*60}")
    for rank, (score, game, patterns) in enumerate(ranked[:args.top], 1):
        labels = ", ".join(p.label for p in patterns if p.kind != "speed_run") or "—"
        print(f"  #{rank:<4} {game['game_index']:>5} {game['seed']:>6} {game['total_steps']:>6} {score:>6}  {labels}")

    # ── HTML report ─────────────────────────────────────────────────────────
    if args.output:
        out_path = args.output
    else:
        stem = args.traces.stem
        out_path = args.traces.parent / f"{stem}_report.html"

    html = build_html(d, ranked, top_n=args.top, pattern_freq=pattern_freq)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nHTML report saved to {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
