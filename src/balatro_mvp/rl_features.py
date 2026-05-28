"""Feature extraction utilities for state-action value learning."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .environment import Action, Card, RANKS, SUITS, score_cards


OBSERVATION_FEATURE_DIM = 4 + len(RANKS) + len(SUITS) + len(RANKS) + len(SUITS)
ACTION_FEATURE_DIM = 2 + 1 + len(RANKS) + len(SUITS) + 1
STATE_ACTION_FEATURE_DIM = OBSERVATION_FEATURE_DIM + ACTION_FEATURE_DIM


def extract_observation_features(observation: dict[str, Any]) -> list[float]:
    """Convert one environment observation into a flat numeric feature list.

    The feature layout is:
    - chips_needed
    - chips_scored
    - hands_left
    - discards_left
    - hand rank histogram
    - hand suit histogram
    - unseen deck rank histogram
    - unseen deck suit histogram
    """

    hand = tuple(observation["hand"])
    unseen_deck = tuple(observation["unseen_deck"])

    features: list[float] = [
        float(observation["chips_needed"]),
        float(observation["chips_scored"]),
        float(observation["hands_left"]),
        float(observation["discards_left"]),
    ]
    features.extend(_rank_histogram(hand))
    features.extend(_suit_histogram(hand))
    features.extend(_rank_histogram(unseen_deck))
    features.extend(_suit_histogram(unseen_deck))
    return features


def extract_action_features(observation: dict[str, Any], action: Action) -> list[float]:
    """Convert one legal action into a flat numeric feature list.

    The feature layout is:
    - action type one-hot (play, discard)
    - number of selected cards
    - selected-card rank histogram
    - selected-card suit histogram
    - immediate lookup-table score for plays, else 0
    """

    hand = tuple(observation["hand"])
    selected_cards = tuple(hand[index] for index in action.card_indices)
    immediate_score = 0.0
    if action.type == "play":
        _, chips_gained = score_cards(selected_cards)
        immediate_score = float(chips_gained)

    features: list[float] = [
        1.0 if action.type == "play" else 0.0,
        1.0 if action.type == "discard" else 0.0,
        float(len(action.card_indices)),
    ]
    features.extend(_rank_histogram(selected_cards))
    features.extend(_suit_histogram(selected_cards))
    features.append(immediate_score)
    return features


def extract_state_action_features(observation: dict[str, Any], action: Action) -> list[float]:
    """Concatenate observation and action features for Q(s, a) scoring."""

    return extract_observation_features(observation) + extract_action_features(observation, action)


def _rank_histogram(cards: Iterable[Card]) -> list[float]:
    """Count cards by rank in the canonical Balatro MVP rank order."""

    rank_counts = Counter(card.rank for card in cards)
    return [float(rank_counts[rank]) for rank in RANKS]


def _suit_histogram(cards: Iterable[Card]) -> list[float]:
    """Count cards by suit in the canonical Balatro MVP suit order."""

    suit_counts = Counter(card.suit for card in cards)
    return [float(suit_counts[suit]) for suit in SUITS]
