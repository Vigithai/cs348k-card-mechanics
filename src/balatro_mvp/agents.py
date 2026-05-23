"""Baseline agent implementations for the Balatro MVP."""

from __future__ import annotations

from abc import ABC, abstractmethod
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


def _resolve_cards(hand: tuple[Card, ...] | list[Card], action: Action) -> tuple[Card, ...]:
    return tuple(hand[index] for index in action.card_indices)
