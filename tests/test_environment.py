from __future__ import annotations

from collections import Counter
from math import comb
from pathlib import Path
import sys
from typing import Iterable
import unittest


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp import (
    Action,
    BalatroMVPEnvironment,
    Card,
    GameState,
    HAND_SCORES,
    ROUND_TARGET_PRESETS,
    classify_poker_hand,
    create_standard_deck,
    score_cards,
)

CardSpec = tuple[str, str]


def get_card(deck: tuple[Card, ...], spec: CardSpec) -> Card:
    rank, suit = spec
    for card in deck:
        if card.rank == rank and card.suit == suit:
            return card
    raise AssertionError(f"Missing card for spec {spec!r}")


def build_cards(deck: tuple[Card, ...], specs: Iterable[CardSpec]) -> list[Card]:
    return [get_card(deck, spec) for spec in specs]


def set_state(
    env: BalatroMVPEnvironment,
    *,
    hand_specs: Iterable[CardSpec],
    unseen_prefix_specs: Iterable[CardSpec] = (),
    discard_specs: Iterable[CardSpec] = (),
    chips_needed: int = 300,
    chips_scored: int = 0,
    hands_left: int = 4,
    discards_left: int = 4,
    round_index: int = 1,
    is_terminal: bool = False,
    result: str | None = None,
) -> GameState:
    full_deck = env.full_deck
    hand = build_cards(full_deck, hand_specs)
    discard_pile = build_cards(full_deck, discard_specs)
    unseen_prefix = build_cards(full_deck, unseen_prefix_specs)
    used_cards = set(hand) | set(discard_pile) | set(unseen_prefix)
    unseen_remainder = [card for card in full_deck if card not in used_cards]
    state = GameState(
        full_deck=full_deck,
        hand=hand,
        unseen_deck=unseen_prefix + unseen_remainder,
        discard_pile=discard_pile,
        chips_needed=chips_needed,
        chips_scored=chips_scored,
        hands_left=hands_left,
        discards_left=discards_left,
        target_hand_size=env.target_hand_size,
        round_index=round_index,
        max_rounds_to_win=env.max_rounds_to_win,
        is_terminal=is_terminal,
        result=result,
    )
    env.state = state
    return state


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

    def test_round_initialization_accepts_easy_round_target_preset(self) -> None:
        env = BalatroMVPEnvironment(seed=19, round_chip_targets=ROUND_TARGET_PRESETS["easy"])

        round_one_state = env.state
        self.assertIsNotNone(round_one_state)
        assert round_one_state is not None
        self.assertEqual(round_one_state.chips_needed, 150)

        round_two_state = env.start_round(2)
        self.assertEqual(round_two_state.chips_needed, 250)


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


class PokerScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.deck = create_standard_deck()

    def test_pair_classification_and_score(self) -> None:
        cards = build_cards(self.deck, [("2", "club"), ("2", "heart")])

        self.assertEqual(classify_poker_hand(cards), "pair")
        self.assertEqual(score_cards(cards), ("pair", HAND_SCORES["pair"][0] * HAND_SCORES["pair"][1]))

    def test_straight_classification_and_score(self) -> None:
        cards = build_cards(
            self.deck,
            [("A", "spade"), ("2", "club"), ("3", "heart"), ("4", "diamond"), ("5", "spade")],
        )

        self.assertEqual(classify_poker_hand(cards), "straight")
        self.assertEqual(
            score_cards(cards),
            ("straight", HAND_SCORES["straight"][0] * HAND_SCORES["straight"][1]),
        )

    def test_broadway_straight_classification_and_score(self) -> None:
        cards = build_cards(
            self.deck,
            [("10", "club"), ("J", "heart"), ("Q", "spade"), ("K", "diamond"), ("A", "club")],
        )

        self.assertEqual(classify_poker_hand(cards), "straight")
        self.assertEqual(
            score_cards(cards),
            ("straight", HAND_SCORES["straight"][0] * HAND_SCORES["straight"][1]),
        )

    def test_flush_classification_and_score(self) -> None:
        cards = build_cards(
            self.deck,
            [("2", "heart"), ("5", "heart"), ("8", "heart"), ("J", "heart"), ("K", "heart")],
        )

        self.assertEqual(classify_poker_hand(cards), "flush")
        self.assertEqual(score_cards(cards), ("flush", HAND_SCORES["flush"][0] * HAND_SCORES["flush"][1]))

    def test_royal_flush_classification_and_score(self) -> None:
        cards = build_cards(
            self.deck,
            [("10", "heart"), ("J", "heart"), ("Q", "heart"), ("K", "heart"), ("A", "heart")],
        )

        self.assertEqual(classify_poker_hand(cards), "royal_flush")
        self.assertEqual(
            score_cards(cards),
            ("royal_flush", HAND_SCORES["royal_flush"][0] * HAND_SCORES["royal_flush"][1]),
        )


class StepTransitionTests(unittest.TestCase):
    def test_discard_transition_redraws_and_never_ends_round(self) -> None:
        env = BalatroMVPEnvironment(seed=23)
        set_state(
            env,
            hand_specs=[
                ("2", "club"),
                ("3", "club"),
                ("4", "club"),
                ("5", "diamond"),
                ("6", "diamond"),
                ("7", "diamond"),
                ("8", "diamond"),
            ],
            unseen_prefix_specs=[("A", "spade"), ("K", "heart")],
            discards_left=1,
        )

        next_obs, reward, done, info = env.step(Action(type="discard", card_indices=(0, 3)))

        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertFalse(info["round_ended"])
        self.assertIsNone(info["round_result"])
        self.assertIsNone(info["run_result"])
        self.assertFalse(info["next_round_started"])
        self.assertIsNone(info["hand_category"])
        self.assertEqual(env.state.discards_left, 0)
        self.assertEqual(env.state.hands_left, 4)
        self.assertEqual(len(env.state.hand), 7)
        self.assertEqual(
            env.state.discard_pile,
            build_cards(env.full_deck, [("2", "club"), ("5", "diamond")]),
        )
        self.assertEqual(env.state.hand[-2:], build_cards(env.full_deck, [("A", "spade"), ("K", "heart")]))
        self.assertEqual(next_obs["round_index"], 1)

    def test_round_one_win_advances_to_round_two_with_fresh_state(self) -> None:
        env = BalatroMVPEnvironment(seed=29)
        set_state(
            env,
            hand_specs=[
                ("2", "club"),
                ("2", "heart"),
                ("4", "club"),
                ("5", "club"),
                ("6", "club"),
                ("7", "club"),
                ("8", "club"),
            ],
            chips_scored=280,
            hands_left=1,
            discards_left=2,
            round_index=1,
        )

        next_obs, reward, done, info = env.step(Action(type="play", card_indices=(0, 1)))

        self.assertEqual(reward, 20)
        self.assertFalse(done)
        self.assertTrue(info["round_ended"])
        self.assertEqual(info["round_result"], "round_win")
        self.assertIsNone(info["run_result"])
        self.assertTrue(info["next_round_started"])
        self.assertEqual(info["hand_category"], "pair")
        self.assertEqual(env.state.round_index, 2)
        self.assertEqual(env.state.chips_needed, 500)
        self.assertEqual(env.state.chips_scored, 0)
        self.assertEqual(env.state.hands_left, 4)
        self.assertEqual(env.state.discards_left, 4)
        self.assertEqual(env.state.discard_pile, [])
        self.assertEqual(len(env.state.hand), 7)
        self.assertEqual(next_obs["round_index"], 2)
        self.assertEqual(next_obs["chips_needed"], 500)

    def test_round_loss_sets_terminal_run_loss_and_skips_redraw(self) -> None:
        env = BalatroMVPEnvironment(seed=31)
        set_state(
            env,
            hand_specs=[
                ("2", "club"),
                ("4", "club"),
                ("5", "club"),
                ("6", "club"),
                ("7", "club"),
                ("8", "club"),
                ("9", "club"),
            ],
            unseen_prefix_specs=[("A", "spade")],
            hands_left=1,
            round_index=1,
        )

        next_obs, reward, done, info = env.step(Action(type="play", card_indices=(0,)))

        self.assertEqual(reward, 5)
        self.assertTrue(done)
        self.assertTrue(info["round_ended"])
        self.assertEqual(info["round_result"], "round_loss")
        self.assertEqual(info["run_result"], "run_loss")
        self.assertFalse(info["next_round_started"])
        self.assertEqual(info["hand_category"], "high_card")
        self.assertTrue(env.state.is_terminal)
        self.assertEqual(env.state.result, "run_loss")
        self.assertEqual(env.state.hands_left, 0)
        self.assertEqual(len(env.state.hand), 6)
        self.assertEqual(env.state.discard_pile, build_cards(env.full_deck, [("2", "club")]))
        self.assertEqual(len(env.state.unseen_deck), 45)
        self.assertEqual(next_obs["hands_left"], 0)

    def test_passing_round_two_wins_the_run(self) -> None:
        env = BalatroMVPEnvironment(seed=37)
        set_state(
            env,
            hand_specs=[
                ("K", "heart"),
                ("Q", "heart"),
                ("J", "heart"),
                ("10", "heart"),
                ("A", "heart"),
                ("2", "club"),
                ("3", "club"),
            ],
            chips_scored=0,
            chips_needed=500,
            hands_left=1,
            round_index=2,
        )

        next_obs, reward, done, info = env.step(Action(type="play", card_indices=(0, 1, 2, 3, 4)))

        self.assertEqual(reward, 800)
        self.assertTrue(done)
        self.assertTrue(info["round_ended"])
        self.assertEqual(info["round_result"], "round_win")
        self.assertEqual(info["run_result"], "run_win")
        self.assertFalse(info["next_round_started"])
        self.assertEqual(info["hand_category"], "royal_flush")
        self.assertTrue(env.state.is_terminal)
        self.assertEqual(env.state.result, "run_win")
        self.assertEqual(env.state.round_index, 2)
        self.assertEqual(next_obs["chips_scored"], 800)


if __name__ == "__main__":
    unittest.main()
