from __future__ import annotations

from math import comb
from pathlib import Path
import sys
import unittest


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp import (
    analyze_pruned_discard_candidates,
    compute_keep_scores_by_index,
    create_standard_deck,
)


def build_hand(*card_codes: str):
    deck_lookup = {
        (card.rank, card.suit): card
        for card in create_standard_deck()
    }
    suit_map = {
        "C": "club",
        "D": "diamond",
        "H": "heart",
        "S": "spade",
    }
    hand = []
    for card_code in card_codes:
        normalized_code = card_code.upper()
        rank_code = normalized_code[:-1]
        suit_code = normalized_code[-1]
        rank = "10" if rank_code == "T" else rank_code
        hand.append(deck_lookup[(rank, suit_map[suit_code])])
    return tuple(hand)


class FlushBiasedPruningTests(unittest.TestCase):
    def test_flush_supporting_cards_are_protected_more_than_simple_pair_cards(self) -> None:
        hand = build_hand("AH", "KH", "7H", "3H", "9C", "9S", "2D")

        keep_scores = compute_keep_scores_by_index(hand)

        heart_scores = [keep_scores[index] for index in range(4)]
        pair_scores = [keep_scores[4], keep_scores[5]]
        self.assertGreater(min(heart_scores), max(pair_scores))

    def test_obvious_junk_singleton_is_most_discardable(self) -> None:
        hand = build_hand("AH", "KH", "7H", "3H", "9C", "9S", "2D")

        analysis = analyze_pruned_discard_candidates(hand)

        self.assertEqual(analysis.ranked_discardable_indices[0], 6)

    def test_trips_are_protected_more_than_junk_cards(self) -> None:
        hand = build_hand("8C", "8D", "8H", "KH", "4S", "2D", "AS")

        keep_scores = compute_keep_scores_by_index(hand)

        trip_scores = [keep_scores[index] for index in (0, 1, 2)]
        junk_score = keep_scores[5]
        self.assertGreater(min(trip_scores), junk_score)

    def test_four_of_a_kind_short_circuits_discard_generation(self) -> None:
        hand = build_hand("8C", "8D", "8H", "8S", "AH", "2D", "3C")

        analysis = analyze_pruned_discard_candidates(hand)

        self.assertTrue(analysis.four_of_a_kind_short_circuited)
        self.assertEqual(analysis.best_play_category, "four_of_a_kind")
        self.assertEqual(analysis.pruned_discard_actions, ())

    def test_pruned_discard_candidates_are_much_smaller_than_full_discard_set(self) -> None:
        hand = build_hand("AH", "KH", "7H", "3H", "9C", "9S", "2D")

        analysis = analyze_pruned_discard_candidates(hand)
        full_discard_count = sum(comb(len(hand), subset_size) for subset_size in range(1, 6))

        self.assertLess(len(analysis.pruned_discard_actions), full_discard_count / 2)
        self.assertLessEqual(len(analysis.pruned_discard_actions), 31)

    def test_discard_sizes_four_and_five_only_appear_when_best_play_is_high_card(self) -> None:
        high_card_hand = build_hand("2C", "5D", "8H", "JS", "KC", "3D", "7S")
        pair_hand = build_hand("2C", "2D", "5H", "8S", "JC", "KH", "7D")

        high_card_analysis = analyze_pruned_discard_candidates(high_card_hand)
        pair_analysis = analyze_pruned_discard_candidates(pair_hand)

        self.assertEqual(high_card_analysis.best_play_category, "high_card")
        self.assertTrue(
            any(len(action.card_indices) in {4, 5} for action in high_card_analysis.pruned_discard_actions)
        )
        self.assertNotEqual(pair_analysis.best_play_category, "high_card")
        self.assertTrue(
            all(len(action.card_indices) <= 3 for action in pair_analysis.pruned_discard_actions)
        )


if __name__ == "__main__":
    unittest.main()
