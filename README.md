# Learning to Play Balatro

CS348K Final Project — Lily (lilyth720@gmail.com)

A simplified Balatro simulator with a six-agent ladder from random play to DQN-trained RL, plus a trace analysis pipeline that extracts strategy insights from winning games.

## Results

Evaluated on ante 1 (three blinds: 300 → 450 → 600 chips), 500 games, no jokers or shop.

| Agent | Win Rate |
|---|---:|
| RandomBot | 0% |
| StimBot | 5% |
| DiscardLowestChipBot | 8% |
| LookaheadDiscardBot | 14% |
| PrunedSampledLookaheadBot | 17% |
| **RLQBot (DQN, reward-shaped)** | **35%** |

RLQBot is trained with a DQN-style Q-network (59-dim state-action features → 128 → 64 → 1) over 500 episodes × 3 seeds. It wins at roughly 2× the rate of the best scripted bot without any hand-crafted strategy encoding.

A human baseline (the author) wins approximately 20% of games under the same rules.

## Strategy Analysis

After training, we ran 500 games each for RLQBot and the best scripted bot, saving full per-step traces for every winning game (177 RL wins, 84 scripted wins). The analysis pipeline detects "hunt sequences" — two or more consecutive discards followed by a play — and uses the hand type played immediately after to infer what the bot was building toward.

Four questions guided the analysis:

### Q1: What hand sequences actually win?

We computed bigrams of consecutive hand types played within winning games. RLQBot's top bigram is **two_pair → flush** (9.4% of all transitions), followed by two_pair → two_pair and pair → flush. The scripted bot's top bigram is **pair → pair** — it grinds the same safe hand repeatedly.

The RL agent learned to bank safe chips with two_pair, then spend discards hunting a flush or straight. The scripted bot, despite having an explicit flush-hunting heuristic, doesn't coordinate plays and discards in this way.

### Q2: What does the bot hunt for?

When the RL bot commits to a hunt (2+ consecutive discards), the outcome is a flush 39% of the time — by far the most common target. Straight is second at 12%. The bot overwhelmingly hunts toward premium hands (flush, straight, full house, or better).

### Q3: How does RL differ from the best scripted bot?

Comparing behavioral statistics across winning games:

| Metric | RLQBot | Scripted |
|---|---:|---:|
| Discard rate | 43% | 59% |
| Premium hand rate | 45% | 66% |
| Avg win margin | +120 chips | +107 chips |

The scripted bot discards more aggressively and plays more premium hands per game, yet wins with less margin. The RL agent learned restraint — it knows when a two_pair is good enough and doesn't over-hunt when the deck doesn't support it.

### Q4: The human takeaway

The bot's strategy suggests a heuristic for human play: **not flush-or-bust.** Patiently play two pair or better to bank safe points, then use remaining discards to redraw into a flush or other premium hand. Don't discard a two pair to chase a flush from scratch.

## Limitations and Next Steps

The current analysis conditions only on winning games, which introduces survivorship bias — we don't know whether the same patterns appear in losses or whether losses have distinct signatures worth avoiding.

Other next steps:

- Analyze losses to identify what to avoid and remove survivorship bias from the win-only analysis
- Train the bots further (500 episodes is insufficient for reliable convergence; cross-seed variance is high)
- Discard-aware sequence analysis — move beyond discard-agnostic bigrams to distinguish traces by what was discarded, not just what was played
- Add jokers and the shop layer, where Balatro's real complexity begins

## Environment

The simulator models a standard 52-card deck with no jokers or shop. Each blind begins with a full deck shuffle, a 7-card opening hand, 4 plays, and 4 discards. Scoring uses a fixed poker-hand lookup table:

```
high_card: 5    pair: 20       two_pair: 40       three_of_a_kind: 90
straight: 120   flush: 136     full_house: 150    four_of_a_kind: 728
```

Score = base_chips × mult (simplified from full Balatro scoring).

The environment exposes a gym-style interface: `get_observation()`, `get_legal_actions()`, `step(action)`. Legal action enumeration and all game logic live in the environment; agents only receive an observation dict and choose from the provided action list.

## DQN Details

The Q-network receives a concatenated state-action feature vector (59 dimensions) and outputs a scalar Q-value. At each step the agent scores all legal actions and picks the highest (or samples uniformly during ε-greedy exploration).

**State features (38):** chips_needed, chips_scored, hands_left, discards_left, rank histogram of hand (13), suit histogram of hand (4), rank histogram of unseen deck (13), suit histogram of unseen deck (4).

**Action features (21):** one-hot play/discard (2), number of selected cards (1), rank histogram of selected cards (13), suit histogram of selected cards (4), immediate score or 0 for discards (1).

**Training:** replay buffer (5k), batch size 64, ε decay 1.0 → 0.05 over 2k steps, target network synced every 100 steps, γ = 0.99, Adam lr = 0.001. Reward shaped with +300 round-win bonus and −75 round-loss penalty.

## Key Files

| Path | Purpose |
|---|---|
| `src/balatro_mvp/environment.py` | Core simulator |
| `src/balatro_mvp/agents.py` | All scripted bots |
| `src/balatro_mvp/rl_training.py` | DQN trainer + reward shaping |
| `src/balatro_mvp/rl_features.py` | State-action feature extraction |
| `scripts/train_rl_qbot.py` | Training entry point |
| `scripts/evaluate_rl_qbot.py` | Evaluation runner |
| `scripts/generate_rl_win_traces.py` | RL win trace generator |
| `scripts/generate_scripted_win_traces.py` | Scripted win trace generator |
| `scripts/analyze_win_traces.py` | Q1–Q4 analysis + figure generation |
| `results/traces/` | Winning game traces (JSON) |
| `results/figures/` | Analysis figures (PNG) |

## Reproducing Results

```bash
# Train
python scripts/train_rl_qbot.py

# Evaluate
python scripts/evaluate_rl_qbot.py

# Generate traces
python scripts/generate_rl_win_traces.py
python scripts/generate_scripted_win_traces.py

# Analyze and generate figures
python scripts/analyze_win_traces.py
```

## References

1. LocalThunk. *Balatro*. Playstack, 2024.
2. Mnih et al. "Human-level control through deep reinforcement learning." *Nature*, 2015.
