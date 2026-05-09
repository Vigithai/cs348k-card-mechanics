# cs348k-card-mechanics

## Overview

This project builds a simplified Balatro-like environment in Python for testing baseline agents and analyzing strategy traces. The immediate goal is to create a reproducible simulator where bots can play under a fixed ruleset, so I can compare policies, inspect their decisions, and learn useful strategies and synergies from successful runs.

The broader idea is to use the bot as a teacher. Rather than relying only on subjective playtesting, I want a framework where strategies can be evaluated quantitatively through repeated simulation.

## Problem

I want to get better at Balatro while staying within the game rules and without cheating. The challenge is that it is hard to reason systematically about what decisions are actually strong, what strategies are robust, and what synergies are worth pursuing across many possible runs.

## MVP Goal

Build a simplified Balatro-like environment that supports interchangeable agents. The first milestone is to run simple baseline bots in a reproducible simulator. A secondary goal is to analyze successful traces to learn useful strategies and patterns.

## MVP Rules and Modeling Choices

- Use a standard 52-card deck.
- The agent can always see the full multiset of undrawn cards.
- The agent cannot see the future draw order.
- The game is turn-based and represented as discrete states.
- Legal actions are non-empty plays or discards of 1 to 5 cards.
- A round can end only after a play action, never after a discard.
- Redraw happens only if the round continues.
- Scoring is based on a simplified poker-hand lookup table.

## Environment State

Each round maintains three mutually exclusive subsets whose union is the current playing deck:

- `hand`
- `unseen_deck`
- `discard_pile`

The environment also tracks:

- `chips_needed`
- `chips_scored`
- `hands_left`
- `discards_left`
- `round_index`

For the MVP:
- hand size starts at 7
- each round starts with `hands_left = 4`
- each round starts with `discards_left = 4`
- round 1 target is `300`
- round 2 target is `500`
- passing round 2 wins the run

Each new round starts fresh from the same fixed 52-card deck, with a fresh opening hand and reset counters.

## Core Datatypes

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Card:
    rank: str        # "2"-"10", "J", "Q", "K", "A"
    suit: str        # "club", "spade", "heart", "diamond"
    chip_value: int  # kept for MVP even though scoring uses lookup-table hand values

@dataclass(frozen=True)
class Action:
    type: Literal["play", "discard"]
    card_indices: tuple[int, ...]   # indices into current hand, length 1-5
````

## Bot API

The environment owns game logic. Agents only inspect the observation and choose one action from the legal action list.

```python
obs = env.get_observation()
legal_actions = env.get_legal_actions()
action = agent.act(obs, legal_actions)
next_obs, reward, done, info = env.step(action)
```

## Scoring

For the MVP, played cards are classified into poker-hand categories and scored with a fixed lookup table.

```python
HAND_SCORES = {
    "high_card": (5, 1),
    "pair": (10, 2),
    "two_pair": (20, 2),
    "three_of_a_kind": (30, 3),
    "straight": (30, 4),
    "flush": (35, 4),
    "full_house": (40, 4),
    "four_of_a_kind": (60, 7),
    "straight_flush": (100, 8),
    "royal_flush": (100, 8),
}
```

For the MVP, score is:

```python
score = base_chips * mult
```

This is intentionally simpler than full Balatro scoring.

## Transition Rules

### Play

A play action:

1. validates the selected indices
2. resolves the selected cards
3. scores the selected hand
4. moves played cards to the discard pile
5. decrements `hands_left`
6. adds the resulting score to `chips_scored`
7. checks round win/loss conditions
8. redraws only if the round continues

### Discard

A discard action:

1. validates the selected indices
2. moves selected cards to the discard pile
3. decrements `discards_left`
4. redraws from `unseen_deck`
5. never ends the round directly

## Reward Convention

For the naive MVP:

```python
reward = chips_gained_from_action
```

This aligns naturally with immediate-score baseline policies and makes debugging easier.

## Baseline Bots

### RandomBot

Chooses uniformly from legal actions. This is mainly for sanity-checking the environment.

### StimBot

A strict immediate-score heuristic bot.

* considers only legal play actions
* never discards
* scores each legal play using the current lookup table
* chooses the highest-scoring play
* if the best current play is a pair, it plays only those two cards
* tie-break 1: prefer fewer cards played
* tie-break 2: if still tied, break ties randomly

### ArchetypeBot

A possible later extension that chases a preferred hand family, such as flushes or pairs.

## Evaluation Plan

### Performance Metrics

* win rate
* average rounds passed
* average final chips scored
* variance across random seeds

### Strategy Metrics

* histogram of hand types scored
* frequency of different action sizes
* evidence of dominant or degenerate strategies

### First Outputs

1. table of win rate and average rounds passed by bot
2. histogram of hand types used by each bot
3. sample successful traces with short annotations

## Repo Structure

* `README.md` — project overview and current MVP spec summary
* `cs_348_k_balatro_mvp_spec.md` — detailed implementation spec for the environment and bots
* `src/` — simulator, environment, scoring, and agent code
* `tests/` — unit tests
* `experiments/` — experiment scripts and configs
* `results/` — saved outputs, plots, and tables

## Suggested Implementation Order

1. Implement `Card`, `Action`, and `GameState`
2. Implement deck creation and round initialization
3. Implement `get_observation()`
4. Implement legal-action enumeration
5. Implement poker-hand classification and scoring lookup
6. Implement `step(action)` for play and discard
7. Implement `RandomBot`
8. Implement `StimBot`
9. Run seeded simulations and log traces
10. Add analysis scripts for metrics and plots

## Current Status

The project is currently in MVP implementation planning. The main focus is now on:

* building the core environment
* defining legal actions cleanly
* implementing the scoring logic
* adding baseline bots
* producing trace-based evaluation outputs

## Notes

This repository is focused first on the simplified simulator and evaluation harness. More advanced mechanics, richer scoring, or LLM-based agents can be added later, but the current priority is a clean, reproducible MVP.
