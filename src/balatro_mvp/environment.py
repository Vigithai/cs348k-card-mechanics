"""Core environment primitives for the Balatro MVP."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import random
from typing import Any, Literal


RANKS: tuple[str, ...] = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS: tuple[str, ...] = ("club", "diamond", "heart", "spade")
RANK_TO_CHIP_VALUE: dict[str, int] = {
    **{str(rank): rank for rank in range(2, 11)},
    "J": 10,
    "Q": 10,
    "K": 10,
    "A": 11,
}
RANK_TO_SORT_INDEX: dict[str, int] = {rank: index for index, rank in enumerate(RANKS)}
SUIT_TO_SORT_INDEX: dict[str, int] = {suit: index for index, suit in enumerate(SUITS)}
ROUND_CHIP_TARGETS: dict[int, int] = {
    1: 300,
    2: 500,
}


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str
    chip_value: int

    def __post_init__(self) -> None:
        if self.rank not in RANKS:
            raise ValueError(f"Unsupported rank: {self.rank!r}")
        if self.suit not in SUITS:
            raise ValueError(f"Unsupported suit: {self.suit!r}")

        expected_chip_value = RANK_TO_CHIP_VALUE[self.rank]
        if self.chip_value != expected_chip_value:
            raise ValueError(
                f"chip_value for {self.rank!r} must be {expected_chip_value}, got {self.chip_value}"
            )


@dataclass(frozen=True)
class Action:
    type: Literal["play", "discard"]
    card_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        indices = tuple(self.card_indices)
        object.__setattr__(self, "card_indices", indices)

        if self.type not in {"play", "discard"}:
            raise ValueError(f"Unsupported action type: {self.type!r}")
        if not indices:
            raise ValueError("card_indices must be non-empty")
        if len(indices) > 5:
            raise ValueError("card_indices must contain at most 5 indices")
        if len(set(indices)) != len(indices):
            raise ValueError("card_indices cannot contain duplicates")
        if any(not isinstance(index, int) for index in indices):
            raise TypeError("card_indices must contain integers only")
        if any(index < 0 for index in indices):
            raise ValueError("card_indices cannot contain negative indices")


@dataclass
class GameState:
    full_deck: tuple[Card, ...]
    hand: list[Card]
    unseen_deck: list[Card]
    discard_pile: list[Card]
    chips_needed: int
    chips_scored: int
    hands_left: int
    discards_left: int
    target_hand_size: int
    round_index: int
    max_rounds_to_win: int
    is_terminal: bool
    result: str | None


def create_standard_deck() -> tuple[Card, ...]:
    """Return a standard 52-card deck in canonical rank/suit order."""
    return tuple(
        Card(rank=rank, suit=suit, chip_value=RANK_TO_CHIP_VALUE[rank])
        for suit in SUITS
        for rank in RANKS
    )


def _card_sort_key(card: Card) -> tuple[int, int]:
    return (RANK_TO_SORT_INDEX[card.rank], SUIT_TO_SORT_INDEX[card.suit])


class BalatroMVPEnvironment:
    """Environment slice requested for the Balatro MVP.

    This class currently implements:
    - standard deck creation
    - seeded round initialization
    - observation generation that hides unseen draw order
    - legal action enumeration
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        full_deck: tuple[Card, ...] | None = None,
        target_hand_size: int = 7,
        hands_per_round: int = 4,
        discards_per_round: int = 4,
        max_rounds_to_win: int = 2,
    ) -> None:
        self._rng = random.Random(seed)
        self.full_deck = tuple(full_deck) if full_deck is not None else create_standard_deck()
        self.target_hand_size = target_hand_size
        self.hands_per_round = hands_per_round
        self.discards_per_round = discards_per_round
        self.max_rounds_to_win = max_rounds_to_win
        self.state: GameState | None = None

        self._validate_configuration()
        self._validate_full_deck()
        self.reset_run()

    def reset_run(self) -> GameState:
        """Reset the environment to the start of round 1."""
        return self.start_round(1)

    def start_round(self, round_index: int) -> GameState:
        """Initialize a fresh round from the same fixed 52-card deck."""
        if round_index not in ROUND_CHIP_TARGETS:
            raise ValueError(
                f"Unsupported round_index {round_index}. Supported rounds: {sorted(ROUND_CHIP_TARGETS)}"
            )
        if round_index > self.max_rounds_to_win:
            raise ValueError(
                f"round_index {round_index} exceeds max_rounds_to_win={self.max_rounds_to_win}"
            )

        draw_pile = list(self.full_deck)
        self._rng.shuffle(draw_pile)

        hand = draw_pile[: self.target_hand_size]
        unseen_deck = draw_pile[self.target_hand_size :]
        state = GameState(
            full_deck=self.full_deck,
            hand=hand,
            unseen_deck=unseen_deck,
            discard_pile=[],
            chips_needed=ROUND_CHIP_TARGETS[round_index],
            chips_scored=0,
            hands_left=self.hands_per_round,
            discards_left=self.discards_per_round,
            target_hand_size=self.target_hand_size,
            round_index=round_index,
            max_rounds_to_win=self.max_rounds_to_win,
            is_terminal=False,
            result=None,
        )
        self._validate_state(state)
        self.state = state
        return state

    def get_observation(self) -> dict[str, Any]:
        """Return the public observation, hiding unseen draw order."""
        state = self._require_state()
        return {
            "chips_needed": state.chips_needed,
            "chips_scored": state.chips_scored,
            "hands_left": state.hands_left,
            "discards_left": state.discards_left,
            "round_index": state.round_index,
            "target_hand_size": state.target_hand_size,
            "hand": tuple(state.hand),
            "unseen_deck": tuple(sorted(state.unseen_deck, key=_card_sort_key)),
        }

    def get_legal_actions(self) -> list[Action]:
        """Enumerate all legal non-empty play/discard subsets of size 1-5."""
        state = self._require_state()
        if state.is_terminal:
            return []

        hand_size = len(state.hand)
        if hand_size == 0:
            return []

        subset_limit = min(5, hand_size)
        index_subsets = [
            indices
            for subset_size in range(1, subset_limit + 1)
            for indices in combinations(range(hand_size), subset_size)
        ]

        legal_actions: list[Action] = []
        if state.hands_left > 0:
            legal_actions.extend(Action(type="play", card_indices=indices) for indices in index_subsets)
        if state.discards_left > 0:
            legal_actions.extend(
                Action(type="discard", card_indices=indices) for indices in index_subsets
            )
        return legal_actions

    def step(self, action: Action) -> tuple[dict[str, Any], int, bool, dict[str, Any]]:
        """Placeholder for later scoring and transition work."""
        raise NotImplementedError("step() is not implemented yet for this MVP slice.")

    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("Environment state is not initialized.")
        return self.state

    def _validate_full_deck(self) -> None:
        if len(self.full_deck) != 52:
            raise ValueError(f"full_deck must contain 52 cards, got {len(self.full_deck)}")
        if len(set(self.full_deck)) != 52:
            raise ValueError("full_deck must contain 52 unique cards")

    def _validate_configuration(self) -> None:
        if not 1 <= self.target_hand_size <= 52:
            raise ValueError("target_hand_size must be between 1 and 52")
        if self.hands_per_round < 0:
            raise ValueError("hands_per_round cannot be negative")
        if self.discards_per_round < 0:
            raise ValueError("discards_per_round cannot be negative")
        if self.max_rounds_to_win < 1:
            raise ValueError("max_rounds_to_win must be at least 1")

    def _validate_state(self, state: GameState) -> None:
        hand_set = set(state.hand)
        unseen_set = set(state.unseen_deck)
        discard_set = set(state.discard_pile)

        if hand_set & unseen_set:
            raise ValueError("hand and unseen_deck must be disjoint")
        if hand_set & discard_set:
            raise ValueError("hand and discard_pile must be disjoint")
        if unseen_set & discard_set:
            raise ValueError("unseen_deck and discard_pile must be disjoint")

        combined_cards = Counter(state.hand) + Counter(state.unseen_deck) + Counter(state.discard_pile)
        if combined_cards != Counter(state.full_deck):
            raise ValueError("hand, unseen_deck, and discard_pile must partition full_deck")

        if len(state.hand) > state.target_hand_size:
            raise ValueError("hand cannot exceed target_hand_size")
        if state.hands_left < 0:
            raise ValueError("hands_left cannot be negative")
        if state.discards_left < 0:
            raise ValueError("discards_left cannot be negative")
