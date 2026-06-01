# CS348K Balatro MVP Spec

## Goal

Build a simplified Balatro-like environment in Python that supports interchangeable agents. The first goal is to run baseline bots in a reproducible simulator. A secondary goal is to analyze successful traces to learn useful strategies and synergies.

## Project framing

### Problem

Get better at Balatro within the game rules, with no cheating.

### Solution

Build a bot that plays a simplified Balatro-like game, compare bot policies, and inspect decision traces to extract strategies and synergies. The bot acts as a teacher.

### Metric

Primary performance metric: passing blinds by reaching the target chip threshold. A run is won by defeating the Boss Blind at Ante 8.

---

## MVP modeling choices

- Use a standard 52-card deck.
- The agent can always see the full multiset of undrawn cards.
- The agent cannot see the future draw order.
- The game is turn-based and represented as discrete states.
- Legal actions are only non-empty plays or discards of 1 to 5 cards.
- A round can end only after a play action, never after a discard.
- Redraw happens only if the round continues.
- For MVP scoring, played cards are classified using a poker-hand lookup table.

---

## Core datatypes

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Card:
    rank: str        # one of: "2"-"10", "J", "Q", "K", "A"
    suit: str        # one of: "club", "spade", "heart", "diamond"
    chip_value: int  # 2-10 for number cards, 10 for face cards, 11 for ace

@dataclass(frozen=True)
class Action:
    type: Literal["play", "discard"]
    card_indices: tuple[int, ...]   # indices into current hand, length 1-5
```

### Example actions

```python
Action(type="play", card_indices=(1, 2, 4, 5, 6))
Action(type="discard", card_indices=(0, 3))
```

---

## Environment state

The full deck is fixed for the run, and each blind maintains three mutually exclusive subsets whose union is the current playing deck:

- `hand`
- `unseen_deck`
- `discard_pile`

### State invariant

At all times:

```python
set(hand) ∩ set(unseen_deck) = ∅
set(hand) ∩ set(discard_pile) = ∅
set(unseen_deck) ∩ set(discard_pile) = ∅
hand ∪ unseen_deck ∪ discard_pile = full_deck
```

### Suggested environment state

```python
@dataclass
class GameState:
    full_deck: tuple[Card, ...]
    hand: list[Card]
    unseen_deck: list[Card]
    discard_pile: list[Card]

    chips_needed: int
    chips_scored: int
    hands_left: int        # starts at 4 each blind
    discards_left: int     # starts at 4 each blind
    target_hand_size: int  # 7

    ante: int              # 1-8
    blind_type: str        # "small_blind", "big_blind", or "boss_blind"
    max_ante: int          # default 8
    is_terminal: bool
    result: str | None     # None, "round_win", "round_loss", "run_win", "run_loss"
```

---

## Observation exposed to agents

The environment state is fully observable except for the future draw order.

```python
{
    "chips_needed": 300,
    "chips_scored": 10,
    "hands_left": 1,
    "discards_left": 1,
    "ante": 1,
    "blind_type": "small_blind",
    "hand": [
        Card(rank="2", suit="heart", chip_value=2),
        Card(rank="K", suit="heart", chip_value=10),
        Card(rank="Q", suit="heart", chip_value=10),
        Card(rank="2", suit="club", chip_value=2),
        Card(rank="J", suit="heart", chip_value=10),
        Card(rank="10", suit="heart", chip_value=10),
        Card(rank="A", suit="heart", chip_value=11),
    ],
    "unseen_deck": [
        Card(rank="2", suit="spade", chip_value=2),
    ],
}
```

---

## Bot API

### Minimal interface

```python
obs = env.get_observation()
legal_actions = env.get_legal_actions()
action = agent.act(obs, legal_actions)
next_obs, reward, done, info = env.step(action)
```

For the naive MVP, `reward` should be the chips gained from the chosen action. Round or run win/loss should be exposed through `done` and `info`.

### Agent contract

```python
class Agent:
    def act(self, observation, legal_actions) -> Action:
        ...
```

The environment owns all game logic. Agents only inspect the observation and choose one action from the legal action list.

---

## Legal actions

A legal action is either:

- `play` on any subset of hand indices of size 1 to 5
- `discard` on any subset of hand indices of size 1 to 5

Additional legality constraints:

- no empty actions
- no duplicate indices
- all indices must refer to valid current-hand positions
- `hands_left > 0` is required for play actions
- `discards_left > 0` is required for discard actions

### Action enumeration rule

Given a hand of size `n`, legal actions are all subsets of indices of size `1..min(5, n)`, labeled as `play` and `discard` when the corresponding resource is available.

---

## Scoring

For the MVP, scoring uses a deterministic lookup table over poker hand categories.

### Scoring rule

- A play action selects a subset of cards.
- The environment evaluates the selected cards.
- The selected cards are mapped to the best valid poker-hand category for the MVP rule set.
- Each hand category has a fixed `(chips, mult)` pair.
- For the MVP, the played-hand score is computed as:

```python
score = base_chips * mult
```

- That value is added to `chips_scored`.

### MVP hand table

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

### Hand definitions for the MVP

- `high_card`: when no other hand is possible, score whichever played card has the highest rank
- `pair`: two cards with a matching rank
- `two_pair`: two cards with a matching rank and two cards with a different matching rank
- `three_of_a_kind`: three cards with a matching rank
- `straight`: five cards with consecutive ranks; aces can count as high (`A K Q J 10`) or low (`A 2 3 4 5`), but not both at the same time
- `flush`: five cards of any rank sharing the same suit
- `full_house`: three cards with one matching rank and two cards with a different matching rank
- `four_of_a_kind`: four cards with a matching rank
- `straight_flush`: five cards in consecutive order that all share the same suit
- `royal_flush`: a straight flush containing `A, K, Q, J, 10` of the same suit

This deliberately simplifies real Balatro scoring. For the MVP, the lookup table alone determines the immediate score contribution of a played hand.

## Transition rules

## Play transition

Given `Action(type="play", card_indices=...)`:

1. Validate that the action is legal.
2. Resolve the selected cards from the current hand.
3. Score the selected cards using the lookup table.
4. Move selected cards from `hand` to `discard_pile`.
5. Decrement `hands_left` by 1.
6. Add the scored chips to `chips_scored`.
7. Check round terminal conditions:
   - if `chips_scored >= chips_needed`, the round ends in a win
   - else if `hands_left == 0`, the round ends in a loss
8. If the round continues, redraw up to the number of played cards from `unseen_deck`.
9. Update terminal/run state if needed.

## Discard transition

Given `Action(type="discard", card_indices=...)`:

1. Validate that the action is legal.
2. Resolve the selected cards from the current hand.
3. Move selected cards from `hand` to `discard_pile`.
4. Decrement `discards_left` by 1.
5. Redraw up to the number of discarded cards from `unseen_deck`.
6. The round does not end here.
7. Continue to the next decision state.

### Redraw rule

Redraw is without replacement from `unseen_deck`. If fewer cards remain than requested, draw as many as are available.

For the MVP, the normal target hand size is 7. After a play or discard, redraw toward hand size 7 unless the unseen deck is exhausted.

---

## Blind and run termination

### Blind ends in win

After a play action, if:

```python
chips_scored >= chips_needed
```

### Blind ends in loss

After a play action, if:

```python
chips_scored < chips_needed and hands_left == 0
```

### Ante/Blind scoring system

An Ante consists of three Blinds played in sequence: Small Blind, Big Blind, Boss Blind. A run always begins at Ante 1. After defeating the Boss Blind at Ante 8, the run is won.

The chip requirement for each blind is:

```python
chips_needed = ANTE_BASE_CHIPS[ante] * BLIND_MULTIPLIERS[blind_type]
```

Where:

```python
BLIND_MULTIPLIERS = {"small_blind": 1.0, "big_blind": 1.5, "boss_blind": 2.0}

ANTE_BASE_CHIPS = {
    1: 300,
    2: 800,
    3: 2_000,
    4: 5_000,
    5: 11_000,
    6: 20_000,
    7: 35_000,
    8: 50_000,
}
```

### Blind advancement

- Beating the small blind advances to the big blind of the same ante.
- Beating the big blind advances to the boss blind of the same ante.
- Beating the boss blind advances to the small blind of the next ante.
- Beating the boss blind at Ante 8 (max_ante) wins the run.

### Run rules

- Each new blind starts fresh from the same fixed 52-card deck.
- On each new blind, reset `hand`, `unseen_deck`, and `discard_pile` from that deck.
- On each new blind, reset `hands_left = 4`, `discards_left = 4`, and `chips_scored = 0`.
- A run is lost immediately when any blind is lost.

---

## Worked example

### Starting observation

```python
{
    "chips_needed": 300,
    "chips_scored": 10,
    "hands_left": 1,
    "discards_left": 1,
    "ante": 1,
    "blind_type": "small_blind",
    "hand": [
        Card(rank="2", suit="heart", chip_value=2),
        Card(rank="K", suit="heart", chip_value=10),
        Card(rank="Q", suit="heart", chip_value=10),
        Card(rank="2", suit="club", chip_value=2),
        Card(rank="J", suit="heart", chip_value=10),
        Card(rank="10", suit="heart", chip_value=10),
        Card(rank="A", suit="heart", chip_value=11),
    ],
    "unseen_deck": [
        Card(rank="2", suit="spade", chip_value=2),
    ],
}
```

### Example 1: play action

```python
Action(type="play", card_indices=(1, 2, 4, 5, 6))
```

Resolved cards:

- K heart
- Q heart
- J heart
- 10 heart
- A heart

For MVP scoring, this is a royal flush. From the lookup table, royal flush = `(100 chips, x8 mult)`, so the score added is `100 * 8 = 800`.

State update:

- move those 5 cards to `discard_pile`
- decrement `hands_left` from 1 to 0
- update `chips_scored` from 10 to 810
- since `chips_scored >= chips_needed`, the blind ends in a win
- advance to the next blind (big blind of ante 1)
- no redraw is needed because the blind already ended

### Example 2: discard action

```python
Action(type="discard", card_indices=(0, 3))
```

Resolved cards:

- 2 heart
- 2 club

State update:

- move those 2 cards to `discard_pile`
- decrement `discards_left` from 1 to 0
- redraw up to 2 cards from `unseen_deck`
- only one card is available, so draw just that one
- `unseen_deck` becomes empty
- hand size becomes 6
- round continues
- because `discards_left == 0`, future legal actions can only be plays

---

## Initial bot plan

### Baseline bots

1. `RandomBot`

   - chooses uniformly from legal actions
   - purpose: sanity-check environment

2. `StimBot`

   - enumerates all legal play actions and scores each one using the current lookup table
   - never discards
   - picks the immediately highest-scoring play
   - if the best current scoring play is a pair, it plays only those two cards
   - tie-break 1: prefer the action that uses fewer cards
   - tie-break 2: if there is still a tie, break ties randomly
   - purpose: strong baseline and easy-to-interpret policy

3. `ArchetypeBot`

   - chases one preferred hand family, such as flushes or pairs
   - purpose: compare specialized strategy behavior

### Recommendation

Start with heuristic bots first. RL agents are possible later, but are not required for the MVP.

If adding an LLM-based agent later, Gemma 4 is a reasonable candidate because Google currently documents Gemma 4 as an open-weights model family intended for reasoning, coding, and agentic workflows, with instruction-tuned variants and long-context support. However, it should be treated as an extension after the heuristic baselines are working. ([ai.google.dev](https://ai.google.dev/gemma/docs/core/model_card_4?utm_source=chatgpt.com))

---

## Reward convention

For the naive MVP:

```python
reward = chips_gained_from_action
```

This makes the environment easy to debug and aligns with `StimBot`'s immediate-score objective.

Use `info` to expose higher-level outcomes such as:
- whether the action ended the round
- whether the round was won or lost
- whether the run was won or lost
- what hand category was scored

## Trace format

A trace is a sequence of decision-time records collected over a run.

```python
[
    {
        "observation": ...,
        "legal_actions": ...,
        "chosen_action": ...,
        "reward": ...,
        "info": ...,
    },
    ...
]
```

This will support later analysis of strategies and synergies.

---

## Initial evaluation plan

### Performance metrics

- win rate
- average rounds passed
- average final chips scored
- variance across random seeds

### Strategy metrics

- histogram of hand types scored
- frequency of different action sizes
- evidence of dominant or degenerate strategies

### First outputs to generate

1. table of win rate and average rounds passed by bot
2. histogram of hand types used by each bot
3. sample successful traces with short annotations

---

## Suggested implementation order

1. Implement `Card`, `Action`, and `GameState`.
2. Implement deck creation and round initialization.
   - use the same fixed 52-card deck each round
   - deal a fresh opening hand of 7 cards
   - reset hands/discards counters to 4 each round
3. Implement `get_observation()`.
4. Implement legal-action enumeration.
5. Implement scoring lookup for poker categories.
6. Implement `step(action)` for play and discard.
7. Implement `RandomBot`.
8. Implement `StimBot`.
9. Run seeded simulations and log traces.
10. Add analysis scripts for metrics and plots.

---

## Notes for Codex / Claude

Important constraints:

- Keep environment logic separate from agent logic.
- Enumerate legal actions inside the environment.
- Agents should only choose from provided legal actions.
- Use deterministic behavior when seed is fixed.
- Prefer clean, debuggable code over optimization.
- Do not implement full Balatro mechanics yet. Stay within the MVP defined above.
- Implement `RandomBot` and the strict immediate-score `StimBot` before attempting an LLM agent.
- If an LLM agent is added later, Gemma 4 is a plausible option, but keep it behind the same `agent.act(observation, legal_actions)` interface so it can be swapped cleanly with heuristic agents.
- `get_legal_actions()` should still include both play and discard actions whenever they are legal, even for bots like `StimBot` that will ignore discard actions.

