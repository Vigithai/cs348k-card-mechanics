from __future__ import annotations

from collections import Counter
from math import comb
from pathlib import Path
import sys
import unittest


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp import BalatroMVPEnvironment, Card, create_standard_deck


class StandardDeckTests(unittest.TestCase):
    def test_create_standard_deck_has_52_unique_cards_with_expected_distribution(self) -> None:
        deck = create_standard_deck()

        self.assertEqual(len(deck), 52)
        self.assertEqual(len(set(deck)), 52)

        rank_counts = Counter(card.rank for card in deck)
        suit_counts = Counter(card.suit for card in deck)

        self.assertTrue(all(count == 4 for count in rank_counts.values()))
        self.assertTrue(all(count == 13 for count in suit_counts.values()))
        self.assertIn(Card(rank="A", suit="spade", chip_value=11), deck)
        self.assertIn(Card(rank="K", suit="heart", chip_value=10), deck)
        self.assertIn(Card(rank="2", suit="club", chip_value=2), deck)


class RoundInitializationTests(unittest.TestCase):
    def test_round_initialization_resets_round_state_for_rounds_one_and_two(self) -> None:
        env = BalatroMVPEnvironment(seed=7)

        round_one_state = env.state
        self.assertIsNotNone(round_one_state)
        assert round_one_state is not None

        self.assertEqual(round_one_state.round_index, 1)
        self.assertEqual(round_one_state.chips_needed, 300)
        self.assertEqual(round_one_state.chips_scored, 0)
        self.assertEqual(round_one_state.hands_left, 4)
        self.assertEqual(round_one_state.discards_left, 4)
        self.assertEqual(len(round_one_state.hand), 7)
        self.assertEqual(len(round_one_state.unseen_deck), 45)
        self.assertEqual(round_one_state.discard_pile, [])
        self.assertEqual(round_one_state.full_deck, create_standard_deck())

        round_two_state = env.start_round(2)
        self.assertEqual(round_two_state.round_index, 2)
        self.assertEqual(round_two_state.chips_needed, 500)
        self.assertEqual(round_two_state.chips_scored, 0)
        self.assertEqual(round_two_state.hands_left, 4)
        self.assertEqual(round_two_state.discards_left, 4)
        self.assertEqual(len(round_two_state.hand), 7)
        self.assertEqual(len(round_two_state.unseen_deck), 45)
        self.assertEqual(round_two_state.discard_pile, [])
        self.assertEqual(
            Counter(round_two_state.hand + round_two_state.unseen_deck),
            Counter(round_two_state.full_deck),
        )

    def test_get_observation_exposes_round_state_without_leaking_draw_order(self) -> None:
        env = BalatroMVPEnvironment(seed=17)

        observation = env.get_observation()

        self.assertEqual(observation["chips_needed"], 300)
        self.assertEqual(observation["chips_scored"], 0)
        self.assertEqual(observation["hands_left"], 4)
        self.assertEqual(observation["discards_left"], 4)
        self.assertEqual(observation["round_index"], 1)
        self.assertEqual(observation["target_hand_size"], 7)
        self.assertEqual(len(observation["hand"]), 7)
        self.assertEqual(len(observation["unseen_deck"]), 45)
        self.assertIsInstance(observation["hand"], tuple)
        self.assertIsInstance(observation["unseen_deck"], tuple)


class LegalActionEnumerationTests(unittest.TestCase):
    def test_get_legal_actions_enumerates_all_play_and_discard_subsets_up_to_size_five(self) -> None:
        env = BalatroMVPEnvironment(seed=11)

        legal_actions = env.get_legal_actions()
        subset_count = sum(comb(7, subset_size) for subset_size in range(1, 6))

        self.assertEqual(len(legal_actions), subset_count * 2)
        self.assertTrue(
            any(
                action.type == "play" and action.card_indices == (0, 1, 2, 3, 4)
                for action in legal_actions
            )
        )
        self.assertTrue(
            any(action.type == "discard" and action.card_indices == (6,) for action in legal_actions)
        )

        for action in legal_actions:
            self.assertGreaterEqual(len(action.card_indices), 1)
            self.assertLessEqual(len(action.card_indices), 5)
            self.assertEqual(len(set(action.card_indices)), len(action.card_indices))
            self.assertTrue(all(0 <= index < 7 for index in action.card_indices))

    def test_get_legal_actions_respects_remaining_play_and_discard_resources(self) -> None:
        env = BalatroMVPEnvironment(seed=13)
        self.assertIsNotNone(env.state)
        assert env.state is not None

        env.state.discards_left = 0
        play_only_actions = env.get_legal_actions()
        self.assertTrue(play_only_actions)
        self.assertTrue(all(action.type == "play" for action in play_only_actions))

        env.state.hands_left = 0
        env.state.discards_left = 4
        discard_only_actions = env.get_legal_actions()
        self.assertTrue(discard_only_actions)
        self.assertTrue(all(action.type == "discard" for action in discard_only_actions))


if __name__ == "__main__":
    unittest.main()
