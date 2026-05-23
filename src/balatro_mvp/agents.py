"""Baseline agent implementations for the Balatro MVP."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from itertools import combinations
import math
import random
from typing import Any

from .environment import Action, Card, HAND_SCORES, score_cards


class Agent(ABC):
    """Minimal agent interface for the Balatro MVP."""

    @abstractmethod
    def act(self, observation: dict[str, Any], legal_actions: list[Action]) -> Action:
        """Choose one action from the provided legal action list."""


class RandomBot(Agent):
    """Pick uniformly from the legal action list using the provided RNG."""

    def __init__(self, *, rng: random.Random | None = None) -> None:
        self.rng = rng if rng is not None else random.Random()

    def act(self, observation: dict[str, Any], legal_actions: list[Action]) -> Action:
        if not legal_actions:
            raise ValueError("RandomBot requires at least one legal action.")
        return self.rng.choice(legal_actions)


class StimBot(Agent):
    """Choose the immediate highest-scoring legal play action."""

    def __init__(self, *, rng: random.Random | None = None) -> None:
        self.rng = rng if rng is not None else random.Random()

    def act(self, observation: dict[str, Any], legal_actions: list[Action]) -> Action:
        best_play_action = _choose_best_play_action(
            observation["hand"],
            legal_actions,
            rng=self.rng,
        )
        if best_play_action is None:
            raise ValueError("StimBot requires at least one legal play action.")
        return best_play_action


class DiscardLowestChipBot(Agent):
    """Discard the weakest single card unless a pair-or-better play is available."""

    def __init__(self, *, rng: random.Random | None = None) -> None:
        self.rng = rng if rng is not None else random.Random()

    def act(self, observation: dict[str, Any], legal_actions: list[Action]) -> Action:
        hand = observation["hand"]
        best_play_action = _choose_best_play_action(hand, legal_actions, rng=self.rng)
        pair_score = HAND_SCORES["pair"][0] * HAND_SCORES["pair"][1]

        if best_play_action is not None:
            _, best_play_score = score_cards(_resolve_cards(hand, best_play_action))
            if best_play_score >= pair_score:
                return best_play_action

        lowest_discard_action = _choose_lowest_single_card_discard(hand, legal_actions)
        if lowest_discard_action is not None:
            return lowest_discard_action

        if best_play_action is None:
            raise ValueError("DiscardLowestChipBot requires at least one legal action.")
        return best_play_action


class LookaheadDiscardBot(Agent):
    """Compare immediate plays against one-card discard lookahead values."""

    def __init__(self, *, rng: random.Random | None = None) -> None:
        self.rng = rng if rng is not None else random.Random()

    def act(self, observation: dict[str, Any], legal_actions: list[Action]) -> Action:
        if not legal_actions:
            raise ValueError("LookaheadDiscardBot requires at least one legal action.")

        hand = observation["hand"]
        unseen_deck = observation["unseen_deck"]
        can_play_next = observation["hands_left"] > 0
        candidate_values: list[dict[str, float | int | Action]] = []

        for action in legal_actions:
            if action.type == "play":
                _, score = score_cards(_resolve_cards(hand, action))
                candidate_values.append(
                    {
                        "action": action,
                        "value": float(score),
                        "card_count": len(action.card_indices),
                    }
                )
            elif len(action.card_indices) == 1:
                expected_value = _expected_best_next_play_score_after_discard(
                    hand,
                    action,
                    unseen_deck,
                    can_play_next=can_play_next,
                )
                candidate_values.append(
                    {
                        "action": action,
                        "value": expected_value,
                        "card_count": len(action.card_indices),
                    }
                )

        if not candidate_values:
            raise ValueError("LookaheadDiscardBot requires at least one supported action.")
        return _choose_highest_value_action(candidate_values, rng=self.rng)


def _choose_best_play_action(
    hand: tuple[Card, ...] | list[Card],
    legal_actions: list[Action],
    *,
    rng: random.Random,
) -> Action | None:
    play_actions = [action for action in legal_actions if action.type == "play"]
    if not play_actions:
        return None

    best_score: int | None = None
    best_card_count: int | None = None
    tied_actions: list[Action] = []

    for action in play_actions:
        _, score = score_cards(_resolve_cards(hand, action))
        card_count = len(action.card_indices)

        if best_score is None or score > best_score:
            best_score = score
            best_card_count = card_count
            tied_actions = [action]
            continue

        if score == best_score:
            if best_card_count is None or card_count < best_card_count:
                best_card_count = card_count
                tied_actions = [action]
            elif card_count == best_card_count:
                tied_actions.append(action)

    return rng.choice(tied_actions)


def _choose_highest_value_action(
    candidate_values: list[dict[str, float | int | Action]],
    *,
    rng: random.Random,
) -> Action:
    best_value = max(float(candidate["value"]) for candidate in candidate_values)
    best_candidates = [
        candidate
        for candidate in candidate_values
        if math.isclose(float(candidate["value"]), best_value, rel_tol=0.0, abs_tol=1e-9)
    ]

    if any(candidate["action"].type == "play" for candidate in best_candidates):
        best_candidates = [
            candidate for candidate in best_candidates if candidate["action"].type == "play"
        ]

    fewest_cards = min(int(candidate["card_count"]) for candidate in best_candidates)
    best_candidates = [
        candidate for candidate in best_candidates if int(candidate["card_count"]) == fewest_cards
    ]
    return rng.choice([candidate["action"] for candidate in best_candidates])


def _choose_lowest_single_card_discard(
    hand: tuple[Card, ...] | list[Card],
    legal_actions: list[Action],
) -> Action | None:
    discard_actions = [action for action in legal_actions if action.type == "discard"]
    if not discard_actions:
        return None

    lowest_index = min(range(len(hand)), key=lambda index: (hand[index].chip_value, index))
    target_indices = (lowest_index,)
    for action in discard_actions:
        if action.card_indices == target_indices:
            return action
    return None


def _expected_best_next_play_score_after_discard(
    hand: tuple[Card, ...] | list[Card],
    discard_action: Action,
    unseen_deck: tuple[Card, ...] | list[Card],
    *,
    can_play_next: bool,
) -> float:
    retained_cards = [card for index, card in enumerate(hand) if index not in discard_action.card_indices]
    if not can_play_next:
        return 0.0
    if not unseen_deck:
        return float(_best_play_score_for_hand(tuple(retained_cards), can_play=True))

    total_score = 0.0
    for candidate_card in unseen_deck:
        next_hand = tuple(retained_cards + [candidate_card])
        total_score += float(_best_play_score_for_hand(next_hand, can_play=True))
    return total_score / len(unseen_deck)


def _best_play_score_for_hand(hand: tuple[Card, ...] | list[Card], *, can_play: bool) -> int:
    if not can_play or not hand:
        return 0

    best_score = 0
    for index_subset in _play_index_subsets_for_hand_size(len(hand)):
        _, score = score_cards(tuple(hand[index] for index in index_subset))
        if score > best_score:
            best_score = score
    return best_score


@lru_cache(maxsize=None)
def _play_index_subsets_for_hand_size(hand_size: int) -> tuple[tuple[int, ...], ...]:
    subset_limit = min(5, hand_size)
    return tuple(
        index_subset
        for subset_size in range(1, subset_limit + 1)
        for index_subset in combinations(range(hand_size), subset_size)
    )


def _resolve_cards(hand: tuple[Card, ...] | list[Card], action: Action) -> tuple[Card, ...]:
    return tuple(hand[index] for index in action.card_indices)
