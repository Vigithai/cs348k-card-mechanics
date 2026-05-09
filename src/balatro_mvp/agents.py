"""Baseline agent implementations for the Balatro MVP."""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Any

from .environment import Action, Card, score_cards


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
        play_actions = [action for action in legal_actions if action.type == "play"]
        if not play_actions:
            raise ValueError("StimBot requires at least one legal play action.")

        hand = observation["hand"]
        best_score: int | None = None
        best_card_count: int | None = None
        tied_actions: list[Action] = []

        for action in play_actions:
            selected_cards = self._resolve_cards(hand, action)
            _, score = score_cards(selected_cards)
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

        return self.rng.choice(tied_actions)

    @staticmethod
    def _resolve_cards(hand: tuple[Card, ...] | list[Card], action: Action) -> tuple[Card, ...]:
        return tuple(hand[index] for index in action.card_indices)
