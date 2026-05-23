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
STRAIGHT_RANK_SETS: frozenset[frozenset[str]] = frozenset(
    frozenset(rank_sequence)
    for rank_sequence in (
        ("A", "2", "3", "4", "5"),
        ("2", "3", "4", "5", "6"),
        ("3", "4", "5", "6", "7"),
        ("4", "5", "6", "7", "8"),
        ("5", "6", "7", "8", "9"),
        ("6", "7", "8", "9", "10"),
        ("7", "8", "9", "10", "J"),
        ("8", "9", "10", "J", "Q"),
        ("9", "10", "J", "Q", "K"),
        ("10", "J", "Q", "K", "A"),
    )
)
DEFAULT_ROUND_TARGET_PRESET: str = "hard"
ROUND_TARGET_PRESETS: dict[str, dict[int, int]] = {
    "hard": {
        1: 300,
        2: 500,
    },
    "easy": {
        1: 150,
        2: 250,
    },
}
ROUND_CHIP_TARGETS: dict[int, int] = dict(ROUND_TARGET_PRESETS[DEFAULT_ROUND_TARGET_PRESET])
HAND_SCORES: dict[str, tuple[int, int]] = {
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


def classify_poker_hand(cards: list[Card] | tuple[Card, ...]) -> str:
    """Classify the selected cards using the MVP poker-hand rules."""
    if not cards:
        raise ValueError("At least one card is required to classify a hand.")
    if len(cards) > 5:
        raise ValueError("At most five cards can be classified in the MVP.")

    rank_counts = Counter(card.rank for card in cards)
    sorted_count_values = sorted(rank_counts.values(), reverse=True)
    is_flush = len(cards) == 5 and len({card.suit for card in cards}) == 1
    is_straight = _is_straight(cards)
    rank_set = {card.rank for card in cards}

    if len(cards) == 5 and is_flush and rank_set == {"10", "J", "Q", "K", "A"}:
        return "royal_flush"
    if len(cards) == 5 and is_flush and is_straight:
        return "straight_flush"
    if sorted_count_values == [4, 1] or sorted_count_values == [4]:
        return "four_of_a_kind"
    if len(cards) == 5 and sorted_count_values == [3, 2]:
        return "full_house"
    if len(cards) == 5 and is_flush:
        return "flush"
    if len(cards) == 5 and is_straight:
        return "straight"
    if 3 in rank_counts.values():
        return "three_of_a_kind"
    if sorted_count_values.count(2) == 2:
        return "two_pair"
    if 2 in rank_counts.values():
        return "pair"
    return "high_card"


def score_cards(cards: list[Card] | tuple[Card, ...]) -> tuple[str, int]:
    """Return the MVP hand category and chips gained for the selected cards."""
    hand_category = classify_poker_hand(cards)
    base_chips, multiplier = HAND_SCORES[hand_category]
    return hand_category, base_chips * multiplier


def _is_straight(cards: list[Card] | tuple[Card, ...]) -> bool:
    if len(cards) != 5:
        return False

    rank_set = frozenset(card.rank for card in cards)
    if len(rank_set) != 5:
        return False
    return rank_set in STRAIGHT_RANK_SETS


class BalatroMVPEnvironment:
    """Environment slice requested for the Balatro MVP.

    This class currently implements:
    - standard deck creation
    - seeded round initialization
    - observation generation that hides unseen draw order
    - legal action enumeration
    - poker-hand classification and lookup-table scoring
    - play/discard state transitions across rounds
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        full_deck: tuple[Card, ...] | None = None,
        round_chip_targets: dict[int, int] | None = None,
        target_hand_size: int = 7,
        hands_per_round: int = 4,
        discards_per_round: int = 4,
        max_rounds_to_win: int = 2,
    ) -> None:
        self._rng = random.Random(seed)
        self.full_deck = tuple(full_deck) if full_deck is not None else create_standard_deck()
        self.round_chip_targets = (
            dict(round_chip_targets) if round_chip_targets is not None else dict(ROUND_CHIP_TARGETS)
        )
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
        if round_index not in self.round_chip_targets:
            raise ValueError(
                f"Unsupported round_index {round_index}. Supported rounds: {sorted(self.round_chip_targets)}"
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
            chips_needed=self.round_chip_targets[round_index],
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
        """Apply a play or discard action and return the next environment interface tuple."""
        state = self._require_state()
        if state.is_terminal:
            raise RuntimeError("Cannot step a terminal environment.")
        if not isinstance(action, Action):
            raise TypeError("step() expects an Action instance.")

        selected_indices = self._validate_action(action, state)
        selected_cards = tuple(state.hand[index] for index in selected_indices)
        reward = 0
        hand_category: str | None = None
        round_ended = False
        round_result: str | None = None
        run_result: str | None = None
        next_round_started = False

        self._move_selected_cards_to_discard(state, selected_indices)

        if action.type == "play":
            hand_category, reward = score_cards(selected_cards)
            state.hands_left -= 1
            state.chips_scored += reward

            if state.chips_scored >= state.chips_needed:
                round_ended = True
                round_result = "round_win"
                if state.round_index >= state.max_rounds_to_win:
                    state.is_terminal = True
                    state.result = "run_win"
                    run_result = "run_win"
                else:
                    next_round_started = True
                    next_round_index = state.round_index + 1
                    self._validate_state(state)
                    self.start_round(next_round_index)
            elif state.hands_left == 0:
                round_ended = True
                round_result = "round_loss"
                state.is_terminal = True
                state.result = "run_loss"
                run_result = "run_loss"
            else:
                self._redraw_to_target(state, selected_count=len(selected_cards))
                state.result = None
        else:
            state.discards_left -= 1
            self._redraw_to_target(state, selected_count=len(selected_cards))
            state.result = None

        if not next_round_started:
            self._validate_state(state)

        current_state = self._require_state()
        info = {
            "action_type": action.type,
            "selected_cards": selected_cards,
            "chips_gained": reward,
            "hand_category": hand_category,
            "round_ended": round_ended,
            "round_result": round_result,
            "run_result": run_result,
            "next_round_started": next_round_started,
        }
        return self.get_observation(), reward, current_state.is_terminal, info

    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("Environment state is not initialized.")
        return self.state

    def _validate_action(self, action: Action, state: GameState) -> tuple[int, ...]:
        if action.type == "play" and state.hands_left <= 0:
            raise ValueError("Play actions require hands_left > 0.")
        if action.type == "discard" and state.discards_left <= 0:
            raise ValueError("Discard actions require discards_left > 0.")

        hand_size = len(state.hand)
        if any(index >= hand_size for index in action.card_indices):
            raise ValueError("Action indices must refer to the current hand.")

        return tuple(sorted(action.card_indices))

    def _move_selected_cards_to_discard(
        self, state: GameState, selected_indices: tuple[int, ...]
    ) -> None:
        moved_cards: list[Card] = []
        for index in selected_indices:
            moved_cards.append(state.hand[index])
        for index in sorted(selected_indices, reverse=True):
            del state.hand[index]
        state.discard_pile.extend(moved_cards)

    def _redraw_to_target(self, state: GameState, *, selected_count: int) -> None:
        cards_needed = max(0, state.target_hand_size - len(state.hand))
        draw_count = min(selected_count, cards_needed, len(state.unseen_deck))
        if draw_count <= 0:
            return

        drawn_cards = state.unseen_deck[:draw_count]
        del state.unseen_deck[:draw_count]
        state.hand.extend(drawn_cards)

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
        required_rounds = set(range(1, self.max_rounds_to_win + 1))
        available_rounds = set(self.round_chip_targets)
        if not required_rounds.issubset(available_rounds):
            raise ValueError(
                f"round_chip_targets must define rounds {sorted(required_rounds)}, got {sorted(available_rounds)}"
            )
        if any(target <= 0 for target in self.round_chip_targets.values()):
            raise ValueError("round chip targets must be positive")

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
