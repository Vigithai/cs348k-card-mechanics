# results_v1 — Archived Results (Pre-Ante/Blind Refactor)

All results in this folder were generated under the **old flat easy/hard preset system**
(easy: rounds 150/250, hard: rounds 300/500) and the **old scoring function** (base_chips × mult,
no card chip values added). They are superseded by results in `results/` which use the
real Balatro ante/blind progression and the corrected scoring formula.

---

## File Classification

### Root-level eval JSONs — Scripted bot evaluations, 200 games each

| File | Bots | Preset | Notes |
|---|---|---|---|
| `checkpoint2_eval_results.json` | RandomBot, StimBot, DiscardLowestChipBot | None (earliest run, no preset field) | Checkpoint 2 deliverable |
| `easy_eval_results.json` | All 5 scripted bots | easy (150/250) | Full ladder eval |
| `hard_eval_results.json` | All 5 scripted bots | hard (300/500) | Full ladder eval |

### `figures/` — Plots from scripted bot evals

| File | Source |
|---|---|
| `average_final_chips_by_bot.png` | checkpoint2 eval |
| `hand_type_distribution_by_bot.png` | checkpoint2 eval |
| `win_rate_by_bot.png` | checkpoint2 eval |
| `average_final_chips_by_bot_across_presets.png` | easy + hard comparison |
| `average_rounds_passed_by_bot_across_presets.png` | easy + hard comparison |
| `hand_type_distribution_by_bot_easy.png` | easy eval |
| `hand_type_distribution_by_bot_hard.png` | hard eval |
| `win_rate_by_bot_across_presets.png` | easy + hard comparison |

### `rl/` — RLQBot DQN training runs

| Path | Preset | Reward shaping | Notes |
|---|---|---|---|
| `rl/easy/seed_{0,1,2}/` | easy | None (bonus=0) | 500 episodes each, clean baseline |
| `rl/hard/seed_{0,1,2}/` | hard | Mixed / inconsistent | seed_0&2: bonus=150; seed_1: bonus=300. Iterative experiment, not a clean comparison |
| `rl/easy_comparison.json` | easy | — | RLQBot vs all scripted bots, final checkpoint |
| `rl/hard_comparison.json` | hard | — | RLQBot vs all scripted bots, final checkpoint |

### `rl_shaped/` — Reward-shaped hard-mode runs (bonus=300, penalty=-75)

| Path | Notes |
|---|---|
| `rl_shaped/hard/seed_0/` | bonus=300 — clean |
| `rl_shaped/hard/seed_2/` | bonus=300 — clean |
| `rl_shaped/seed_0/` | bonus=150 — stale copy, superseded by rl_shaped/hard/seed_0 |
| `rl_shaped/seed_1/` | bonus=300 — the clean seed_1 run |
| `rl_shaped/seed_2/` | bonus=150 — stale copy, superseded by rl_shaped/hard/seed_2 |

The canonical shaped result was: seed_0=0% win, seed_1=5% win, seed_2=5% win → avg 3.3%, beating scripted best of 3.0%.

### `rl_summary/` — Evaluation outputs and plots

| File(s) | Contents |
|---|---|
| `seed_summary.csv` | **Canonical results table** — mean win rate, avg chips, avg rounds per bot per preset across 3 seeds |
| `easy_comparison_seed_{0,1,2}.json` | Per-seed RLQBot vs scripted comparison (easy) |
| `hard_comparison_seed_{0,1,2}.json` | Per-seed RLQBot vs scripted comparison (hard) |
| `easy_seed_{0,1,2}.txt` | Console output from easy eval runs |
| `hard_seed_{0,1,2}.txt` | Console output from hard eval runs |
| `train_seed_{0,1,2}.txt` | Console output from training runs |
| `easy_seed_1_100games.txt` etc. | Larger eval runs (100 games vs default 20) |
| `*.png` | Win rate and chip summary plots |

### `traces/` — Per-game traces

| Path | Contents |
|---|---|
| `traces/easy/LookaheadDiscardBot_seed_{0,1,2}.json` | Single-game traces, scripted bot, easy |
| `traces/easy/PrunedSampledLookaheadBot_seed_{0,1}.json` | Single-game traces, scripted bot, easy |
| `traces/hard/LookaheadDiscardBot_seed_{0,1,2}.json` | Single-game traces, scripted bot, hard |
| `traces/hard/PrunedSampledLookaheadBot_seed_{0,1}.json` | Single-game traces, scripted bot, hard |
| `traces/rl_wins/easy_seed0_78wins.json` | 78 rich per-step win traces from RLQBot on easy (200 games run) |
| `traces/rl_wins/easy_seed0_78wins_report.html` | HTML analysis report — pattern frequency, hunt detection, Game 179 annotated trace |

---

## Headline Numbers (for reference)

**Easy preset** (round targets: 150 / 250):

| Bot | Win Rate | Avg Final Chips |
|---|---:|---:|
| RandomBot | 0% | 38 |
| StimBot | 5% | — |
| DiscardLowestChipBot | 8% | — |
| LookaheadDiscardBot | 23% | — |
| PrunedSampledLookaheadBot | 26% | 186 |
| RLQBot | 54% | 229 |

**Hard preset** (round targets: 300 / 500):

| Bot | Win Rate | Avg Final Chips |
|---|---:|---:|
| PrunedSampledLookaheadBot | 3.0% | 205 |
| RLQBot (unshaped) | 2.3% | 235 |
| RLQBot (reward-shaped, bonus=300) | 3.3% | 196 |
