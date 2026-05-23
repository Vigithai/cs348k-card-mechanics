"""Flush-biased discard pruning helpers for future lookahead bots.

This module is intentionally heuristic and presentation-friendly.
It does not model draw probabilities directly. Instead, it scores how
valuable each card is to keep in a flush-heavy playstyle, then builds a
small discard search set from the cards that look least worth keeping.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Sequence

from .environment import Action, Card, STRAIGHT_RANK_SETS, score_cards


@dataclass(frozen=True)
class DiscardPruningAnalysis:
    """Summary of the flush-biased discard pruning process."""

    keep_scores_by_index: dict[int, float]
    ranked_discardable_indices: tuple[int, ...]
    pruned_discard_actions: tuple[Action, ...]
    four_of_a_kind_short_circuited: bool
    best_play_action: Action | None
    best_play_category: str | None
    best_play_score: int


def analyze_pruned_discard_candidates(
    hand: Sequence[Card],
    *,
    candidate_pool_size: int = 5,
) -> DiscardPruningAnalysis:
    """Score cards for discardability and generate a small discard search set.

    The heuristic is deliberately flush-biased:
    - suit concentration is protected most strongly
    - straight-building structure is protected next
    - trips/quads and full-house potential are protected after that
    - simple pairs matter, but less than flush structure
    - raw chip value only nudges decisions slightly
    """

    if not hand:
        raise ValueError("Discard pruning requires at least one card in hand.")
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive.")

    best_play_action, best_play_category, best_play_score = find_best_immediate_play(hand)
    keep_scores_by_index = compute_keep_scores_by_index(hand)
    ranked_discardable_indices = rank_discardable_indices(keep_scores_by_index)

    if best_play_category == "four_of_a_kind":
        return DiscardPruningAnalysis(
            keep_scores_by_index=keep_scores_by_index,
            ranked_discardable_indices=ranked_discardable_indices,
            pruned_discard_actions=(),
            four_of_a_kind_short_circuited=True,
            best_play_action=best_play_action,
            best_play_category=best_play_category,
            best_play_score=best_play_score,
        )

    candidate_pool = ranked_discardable_indices[: min(candidate_pool_size, len(hand))]
    allowed_discard_sizes = _allowed_discard_sizes(
        best_play_category=best_play_category,
        candidate_pool_size=len(candidate_pool),
    )

    pruned_discard_actions: list[Action] = []
    for discard_size in allowed_discard_sizes:
        for discard_indices in combinations(candidate_pool, discard_size):
            if _passes_guardrails(hand, discard_indices):
                pruned_discard_actions.append(Action(type="discard", card_indices=discard_indices))

    pruned_discard_actions.sort(
        key=lambda action: (
            len(action.card_indices),
            sum(keep_scores_by_index[index] for index in action.card_indices),
            action.card_indices,
        )
    )

    return DiscardPruningAnalysis(
        keep_scores_by_index=keep_scores_by_index,
        ranked_discardable_indices=ranked_discardable_indices,
        pruned_discard_actions=tuple(pruned_discard_actions),
        four_of_a_kind_short_circuited=False,
        best_play_action=best_play_action,
        best_play_category=best_play_category,
        best_play_score=best_play_score,
    )


def compute_keep_scores_by_index(hand: Sequence[Card]) -> dict[int, float]:
    """Return a flush-biased keep score for each hand index.

    Lower scores mean a card is more discardable. These scores are not
    probability-optimal; they are hand-crafted to preserve a flush-heavy,
    easy-to-explain playstyle.
    """

    suit_counts = Counter(card.suit for card in hand)
    rank_counts = Counter(card.rank for card in hand)
    hand_rank_set = {card.rank for card in hand}
    has_full_house_shape = 3 in rank_counts.values() and 2 in rank_counts.values()

    keep_scores: dict[int, float] = {}
    for index, card in enumerate(hand):
        keep_score = 0.0

        # Flush-heavy playstyle: strong suit concentration gets the biggest boosts.
        suit_count = suit_counts[card.suit]
        if suit_count >= 3:
            keep_score += 8.0
        if suit_count >= 4:
            keep_score += 6.0
        if suit_count >= 5:
            keep_score += 2.0

        # Straight support: preserve cards that sit inside strong local rank clusters.
        keep_score += _straight_support_score(card.rank, hand_rank_set)

        # Duplicate-rank structure: trips/quads matter more than simple pairs.
        rank_count = rank_counts[card.rank]
        if rank_count == 2:
            keep_score += 2.5
        elif rank_count == 3:
            keep_score += 6.0
        elif rank_count >= 4:
            keep_score += 9.0

        # Full-house potential is protected, but still below a strong flush build.
        if has_full_house_shape and rank_count >= 2:
            keep_score += 2.0

        # Raw chip value is only a small tie-breaker.
        keep_score += 0.2 * card.chip_value

        keep_scores[index] = keep_score

    return keep_scores


def rank_discardable_indices(keep_scores_by_index: dict[int, float]) -> tuple[int, ...]:
    """Rank hand indices from most discardable to least discardable."""
    return tuple(sorted(keep_scores_by_index, key=lambda index: (keep_scores_by_index[index], index)))


def find_best_immediate_play(hand: Sequence[Card]) -> tuple[Action | None, str | None, int]:
    """Return the best current play under immediate lookup-table scoring."""

    if not hand:
        return None, None, 0

    best_action: Action | None = None
    best_category: str | None = None
    best_score = -1
    best_card_count: int | None = None

    for play_indices in _play_index_subsets_for_hand_size(len(hand)):
        played_cards = tuple(hand[index] for index in play_indices)
        hand_category, hand_score = score_cards(played_cards)
        card_count = len(play_indices)

        if hand_score > best_score:
            best_action = Action(type="play", card_indices=play_indices)
            best_category = hand_category
            best_score = hand_score
            best_card_count = card_count
            continue

        if hand_score == best_score and (best_card_count is None or card_count < best_card_count):
            best_action = Action(type="play", card_indices=play_indices)
            best_category = hand_category
            best_card_count = card_count

    return best_action, best_category, max(0, best_score)


def _straight_support_score(rank: str, hand_rank_set: set[str]) -> float:
    """Reward cards that fit into strong straight windows or local clusters."""
    best_window_overlap = max(
        len(straight_rank_set & hand_rank_set)
        for straight_rank_set in STRAIGHT_RANK_SETS
        if rank in straight_rank_set
    )
    if best_window_overlap < 2:
        return 0.0
    return 2.0 * (best_window_overlap - 1)


def _allowed_discard_sizes(*, best_play_category: str | None, candidate_pool_size: int) -> range:
    maximum_discard_size = min(5, candidate_pool_size)
    if best_play_category == "high_card":
        return range(1, maximum_discard_size + 1)
    return range(1, min(3, maximum_discard_size) + 1)


def _passes_guardrails(hand: Sequence[Card], discard_indices: tuple[int, ...]) -> bool:
    """Reject discard subsets that destroy too much useful structure."""
    return not (
        _violates_flush_guardrail(hand, discard_indices)
        or _violates_straight_guardrail(hand, discard_indices)
        or _violates_duplicate_guardrail(hand, discard_indices)
    )


def _violates_flush_guardrail(hand: Sequence[Card], discard_indices: tuple[int, ...]) -> bool:
    """Keep flush-heavy suit clusters mostly intact."""
    suit_counts = Counter(card.suit for card in hand)
    remaining_suit_counts = Counter(card.suit for index, card in enumerate(hand) if index not in discard_indices)

    for suit, suit_count in suit_counts.items():
        remaining_count = remaining_suit_counts[suit]
        if suit_count >= 4 and remaining_count < 3:
            return True
        if suit_count == 3 and remaining_count < 2:
            return True
    return False


def _violates_straight_guardrail(hand: Sequence[Card], discard_indices: tuple[int, ...]) -> bool:
    """Avoid breaking obvious near-straights too aggressively."""
    current_overlap = _best_straight_window_overlap({card.rank for card in hand})
    remaining_overlap = _best_straight_window_overlap(
        {card.rank for index, card in enumerate(hand) if index not in discard_indices}
    )

    if current_overlap >= 4 and remaining_overlap < 3:
        return True
    if current_overlap == 3 and len(discard_indices) >= 3 and remaining_overlap < 2:
        return True
    return False


def _violates_duplicate_guardrail(hand: Sequence[Card], discard_indices: tuple[int, ...]) -> bool:
    """Protect trips, quads, and obvious full-house potential."""
    rank_counts = Counter(card.rank for card in hand)
    remaining_rank_counts = Counter(
        card.rank for index, card in enumerate(hand) if index not in discard_indices
    )

    if 3 in rank_counts.values() and 2 in rank_counts.values():
        for rank, rank_count in rank_counts.items():
            if rank_count >= 3 and remaining_rank_counts[rank] < 3:
                return True
            if rank_count == 2 and remaining_rank_counts[rank] < 2:
                return True

    for rank, rank_count in rank_counts.items():
        if rank_count >= 3 and remaining_rank_counts[rank] < 2:
            return True
    return False


def _best_straight_window_overlap(ranks: set[str]) -> int:
    if not ranks:
        return 0
    return max(len(straight_rank_set & ranks) for straight_rank_set in STRAIGHT_RANK_SETS)


@lru_cache(maxsize=None)
def _play_index_subsets_for_hand_size(hand_size: int) -> tuple[tuple[int, ...], ...]:
    subset_limit = min(5, hand_size)
    return tuple(
        index_subset
        for subset_size in range(1, subset_limit + 1)
        for index_subset in combinations(range(hand_size), subset_size)
    )
