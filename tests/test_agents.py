from __future__ import annotations

from itertools import combinations
from pathlib import Path
import random
import sys
import unittest


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp import (
    Action,
    Card,
    DiscardLowestChipBot,
    LookaheadDiscardBot,
    PrunedSampledLookaheadBot,
    RandomBot,
    StimBot,
)


class StubChoiceRng:
    """Small deterministic RNG stub for tie-breaking tests."""

    def __init__(self, choice_index: int) -> None:
        self.choice_index = choice_index

    def choice(self, items: list[object]) -> object:
        return items[self.choice_index]


def make_observation(hand: tuple[Card, ...]) -> dict[str, object]:
    return {
        "chips_needed": 300,
        "chips_scored": 0,
        "hands_left": 4,
        "discards_left": 4,
        "round_index": 1,
        "target_hand_size": 7,
        "hand": hand,
        "unseen_deck": (),
    }


def make_all_legal_actions(hand_size: int) -> list[Action]:
    actions: list[Action] = []
    subset_limit = min(5, hand_size)
    for subset_size in range(1, subset_limit + 1):
        for index_subset in combinations(range(hand_size), subset_size):
            actions.append(Action(type="play", card_indices=index_subset))
            actions.append(Action(type="discard", card_indices=index_subset))
    return actions


class RandomBotTests(unittest.TestCase):
    def test_random_bot_always_returns_a_legal_action(self) -> None:
        observation = make_observation(
            (
                Card(rank="2", suit="club", chip_value=2),
                Card(rank="2", suit="heart", chip_value=2),
                Card(rank="5", suit="spade", chip_value=5),
            )
        )
        legal_actions = [
            Action(type="play", card_indices=(0,)),
            Action(type="play", card_indices=(1,)),
            Action(type="discard", card_indices=(0, 1)),
        ]
        bot = RandomBot(rng=random.Random(7))

        for _ in range(100):
            self.assertIn(bot.act(observation, legal_actions), legal_actions)


class StimBotTests(unittest.TestCase):
    def test_stim_bot_never_discards(self) -> None:
        observation = make_observation(
            (
                Card(rank="2", suit="club", chip_value=2),
                Card(rank="2", suit="heart", chip_value=2),
                Card(rank="5", suit="spade", chip_value=5),
            )
        )
        legal_actions = [
            Action(type="discard", card_indices=(2,)),
            Action(type="play", card_indices=(0, 1)),
        ]
        bot = StimBot(rng=random.Random(3))

        chosen_action = bot.act(observation, legal_actions)

        self.assertEqual(chosen_action.type, "play")

    def test_stim_bot_prefers_fewer_cards_when_scores_tie(self) -> None:
        observation = make_observation(
            (
                Card(rank="2", suit="club", chip_value=2),
                Card(rank="2", suit="heart", chip_value=2),
                Card(rank="5", suit="spade", chip_value=5),
            )
        )
        legal_actions = [
            Action(type="play", card_indices=(0, 1, 2)),
            Action(type="play", card_indices=(0, 1)),
        ]
        bot = StimBot(rng=random.Random(5))

        chosen_action = bot.act(observation, legal_actions)

        self.assertEqual(chosen_action, Action(type="play", card_indices=(0, 1)))

    def test_stim_bot_uses_rng_for_equal_score_equal_size_ties(self) -> None:
        observation = make_observation(
            (
                Card(rank="2", suit="club", chip_value=2),
                Card(rank="2", suit="heart", chip_value=2),
                Card(rank="3", suit="club", chip_value=3),
                Card(rank="3", suit="heart", chip_value=3),
            )
        )
        legal_actions = [
            Action(type="play", card_indices=(0, 1)),
            Action(type="play", card_indices=(2, 3)),
        ]

        first_bot = StimBot(rng=StubChoiceRng(choice_index=0))
        second_bot = StimBot(rng=StubChoiceRng(choice_index=1))

        self.assertEqual(first_bot.act(observation, legal_actions), legal_actions[0])
        self.assertEqual(second_bot.act(observation, legal_actions), legal_actions[1])


class DiscardLowestChipBotTests(unittest.TestCase):
    def test_discards_lowest_chip_card_when_no_play_reaches_pair_score(self) -> None:
        observation = make_observation(
            (
                Card(rank="A", suit="heart", chip_value=11),
                Card(rank="5", suit="spade", chip_value=5),
                Card(rank="2", suit="club", chip_value=2),
            )
        )
        legal_actions = [
            Action(type="play", card_indices=(0,)),
            Action(type="play", card_indices=(1,)),
            Action(type="discard", card_indices=(0,)),
            Action(type="discard", card_indices=(1,)),
            Action(type="discard", card_indices=(2,)),
        ]
        bot = DiscardLowestChipBot(rng=random.Random(11))

        chosen_action = bot.act(observation, legal_actions)

        self.assertEqual(chosen_action, Action(type="discard", card_indices=(2,)))

    def test_plays_when_pair_or_better_exists(self) -> None:
        observation = make_observation(
            (
                Card(rank="2", suit="club", chip_value=2),
                Card(rank="2", suit="heart", chip_value=2),
                Card(rank="A", suit="spade", chip_value=11),
            )
        )
        legal_actions = [
            Action(type="play", card_indices=(0,)),
            Action(type="play", card_indices=(0, 1)),
            Action(type="discard", card_indices=(2,)),
        ]
        bot = DiscardLowestChipBot(rng=random.Random(13))

        chosen_action = bot.act(observation, legal_actions)

        self.assertEqual(chosen_action, Action(type="play", card_indices=(0, 1)))

    def test_lowest_chip_tie_break_uses_earliest_hand_index(self) -> None:
        observation = make_observation(
            (
                Card(rank="2", suit="club", chip_value=2),
                Card(rank="2", suit="heart", chip_value=2),
                Card(rank="6", suit="spade", chip_value=6),
            )
        )
        legal_actions = [
            Action(type="play", card_indices=(2,)),
            Action(type="discard", card_indices=(0,)),
            Action(type="discard", card_indices=(1,)),
        ]
        bot = DiscardLowestChipBot(rng=random.Random(17))

        chosen_action = bot.act(observation, legal_actions)

        self.assertEqual(chosen_action, Action(type="discard", card_indices=(0,)))


class LookaheadDiscardBotTests(unittest.TestCase):
    def test_chooses_discard_one_when_expected_next_play_value_is_better(self) -> None:
        observation = {
            **make_observation(
                (
                    Card(rank="9", suit="club", chip_value=9),
                    Card(rank="K", suit="heart", chip_value=10),
                    Card(rank="A", suit="spade", chip_value=11),
                )
            ),
            "unseen_deck": (
                Card(rank="9", suit="diamond", chip_value=9),
                Card(rank="9", suit="heart", chip_value=9),
            ),
        }
        legal_actions = [
            Action(type="play", card_indices=(0,)),
            Action(type="play", card_indices=(1,)),
            Action(type="play", card_indices=(2,)),
            Action(type="discard", card_indices=(1,)),
        ]
        bot = LookaheadDiscardBot(rng=random.Random(19))

        chosen_action = bot.act(observation, legal_actions)

        self.assertEqual(chosen_action, Action(type="discard", card_indices=(1,)))

    def test_chooses_play_when_immediate_play_is_better(self) -> None:
        observation = {
            **make_observation(
                (
                    Card(rank="2", suit="club", chip_value=2),
                    Card(rank="2", suit="heart", chip_value=2),
                    Card(rank="A", suit="spade", chip_value=11),
                )
            ),
            "unseen_deck": (
                Card(rank="4", suit="club", chip_value=4),
                Card(rank="5", suit="diamond", chip_value=5),
            ),
        }
        legal_actions = [
            Action(type="play", card_indices=(0,)),
            Action(type="play", card_indices=(0, 1)),
            Action(type="discard", card_indices=(0,)),
        ]
        bot = LookaheadDiscardBot(rng=random.Random(23))

        chosen_action = bot.act(observation, legal_actions)

        self.assertEqual(chosen_action, Action(type="play", card_indices=(0, 1)))

    def test_prefers_play_over_discard_when_values_tie(self) -> None:
        observation = {
            **make_observation(
                (
                    Card(rank="2", suit="club", chip_value=2),
                    Card(rank="7", suit="heart", chip_value=7),
                    Card(rank="A", suit="spade", chip_value=11),
                )
            ),
            "unseen_deck": (Card(rank="3", suit="diamond", chip_value=3),),
        }
        legal_actions = [
            Action(type="play", card_indices=(0,)),
            Action(type="discard", card_indices=(2,)),
        ]
        bot = LookaheadDiscardBot(rng=random.Random(29))

        chosen_action = bot.act(observation, legal_actions)

        self.assertEqual(chosen_action, Action(type="play", card_indices=(0,)))

    def test_prefers_fewer_cards_when_equal_value_play_options_tie(self) -> None:
        observation = make_observation(
            (
                Card(rank="2", suit="club", chip_value=2),
                Card(rank="2", suit="heart", chip_value=2),
                Card(rank="5", suit="spade", chip_value=5),
            )
        )
        legal_actions = [
            Action(type="play", card_indices=(0, 1, 2)),
            Action(type="play", card_indices=(0, 1)),
        ]
        bot = LookaheadDiscardBot(rng=random.Random(31))

        chosen_action = bot.act(observation, legal_actions)

        self.assertEqual(chosen_action, Action(type="play", card_indices=(0, 1)))

    def test_uses_rng_for_equal_value_equal_type_equal_size_ties(self) -> None:
        observation = make_observation(
            (
                Card(rank="2", suit="club", chip_value=2),
                Card(rank="2", suit="heart", chip_value=2),
                Card(rank="3", suit="club", chip_value=3),
                Card(rank="3", suit="heart", chip_value=3),
            )
        )
        legal_actions = [
            Action(type="play", card_indices=(0, 1)),
            Action(type="play", card_indices=(2, 3)),
        ]

        first_bot = LookaheadDiscardBot(rng=StubChoiceRng(choice_index=0))
        second_bot = LookaheadDiscardBot(rng=StubChoiceRng(choice_index=1))

        self.assertEqual(first_bot.act(observation, legal_actions), legal_actions[0])
        self.assertEqual(second_bot.act(observation, legal_actions), legal_actions[1])


class PrunedSampledLookaheadBotTests(unittest.TestCase):
    def test_can_choose_discard_action_larger_than_size_one(self) -> None:
        hand = (
            Card(rank="A", suit="heart", chip_value=11),
            Card(rank="K", suit="heart", chip_value=10),
            Card(rank="Q", suit="heart", chip_value=10),
            Card(rank="2", suit="heart", chip_value=2),
            Card(rank="3", suit="club", chip_value=3),
            Card(rank="4", suit="diamond", chip_value=4),
            Card(rank="5", suit="spade", chip_value=5),
        )
        observation = {
            **make_observation(hand),
            "unseen_deck": (
                Card(rank="J", suit="heart", chip_value=10),
                Card(rank="10", suit="heart", chip_value=10),
            ),
        }
        legal_actions = make_all_legal_actions(len(hand))
        bot = PrunedSampledLookaheadBot(rng=random.Random(41), sample_count=8, discard_margin=10)

        chosen_action = bot.act(observation, legal_actions)

        self.assertEqual(chosen_action.type, "discard")
        self.assertEqual(len(chosen_action.card_indices), 2)

    def test_pruning_is_used_before_sampling(self) -> None:
        hand = (
            Card(rank="A", suit="heart", chip_value=11),
            Card(rank="K", suit="heart", chip_value=10),
            Card(rank="7", suit="heart", chip_value=7),
            Card(rank="3", suit="heart", chip_value=3),
            Card(rank="9", suit="club", chip_value=9),
            Card(rank="9", suit="spade", chip_value=9),
            Card(rank="2", suit="diamond", chip_value=2),
        )
        observation = {
            **make_observation(hand),
            "unseen_deck": (
                Card(rank="J", suit="heart", chip_value=10),
                Card(rank="10", suit="heart", chip_value=10),
                Card(rank="5", suit="club", chip_value=5),
            ),
        }
        legal_actions = make_all_legal_actions(len(hand))
        bot = PrunedSampledLookaheadBot(rng=random.Random(43), sample_count=6, pruning_candidate_pool_size=5)

        bot.act(observation, legal_actions)
        decision_info = bot.get_last_decision_info()

        self.assertIsNotNone(decision_info)
        assert decision_info is not None
        self.assertGreater(decision_info["raw_legal_discard_count"], decision_info["pruned_discard_candidate_count"])
        self.assertEqual(
            decision_info["pruned_discard_candidate_count"],
            decision_info["sampled_discard_candidate_count"],
        )

    def test_sampling_is_deterministic_when_seed_is_fixed(self) -> None:
        hand = (
            Card(rank="A", suit="heart", chip_value=11),
            Card(rank="K", suit="heart", chip_value=10),
            Card(rank="Q", suit="heart", chip_value=10),
            Card(rank="2", suit="heart", chip_value=2),
            Card(rank="3", suit="club", chip_value=3),
            Card(rank="4", suit="diamond", chip_value=4),
            Card(rank="5", suit="spade", chip_value=5),
        )
        observation = {
            **make_observation(hand),
            "unseen_deck": (
                Card(rank="J", suit="heart", chip_value=10),
                Card(rank="10", suit="heart", chip_value=10),
                Card(rank="6", suit="club", chip_value=6),
                Card(rank="8", suit="diamond", chip_value=8),
                Card(rank="9", suit="spade", chip_value=9),
            ),
        }
        legal_actions = make_all_legal_actions(len(hand))

        first_bot = PrunedSampledLookaheadBot(rng=random.Random(47), sample_count=7)
        second_bot = PrunedSampledLookaheadBot(rng=random.Random(47), sample_count=7)

        first_action = first_bot.act(observation, legal_actions)
        second_action = second_bot.act(observation, legal_actions)
        first_info = first_bot.get_last_decision_info()
        second_info = second_bot.get_last_decision_info()

        self.assertEqual(first_action, second_action)
        self.assertEqual(first_info, second_info)

    def test_margin_rule_prefers_play_when_discard_gain_is_too_small(self) -> None:
        hand = (
            Card(rank="9", suit="club", chip_value=9),
            Card(rank="9", suit="diamond", chip_value=9),
            Card(rank="A", suit="heart", chip_value=11),
            Card(rank="K", suit="spade", chip_value=10),
            Card(rank="2", suit="club", chip_value=2),
        )
        observation = {
            **make_observation(hand),
            "unseen_deck": (Card(rank="3", suit="heart", chip_value=3),),
        }
        legal_actions = make_all_legal_actions(len(hand))
        bot = PrunedSampledLookaheadBot(rng=random.Random(53), sample_count=5, discard_margin=10)

        chosen_action = bot.act(observation, legal_actions)

        self.assertEqual(chosen_action.type, "play")
        self.assertEqual(chosen_action.card_indices, (0, 1))

    def test_four_of_a_kind_short_circuit_skips_discard_search(self) -> None:
        hand = (
            Card(rank="8", suit="club", chip_value=8),
            Card(rank="8", suit="diamond", chip_value=8),
            Card(rank="8", suit="heart", chip_value=8),
            Card(rank="8", suit="spade", chip_value=8),
            Card(rank="A", suit="heart", chip_value=11),
            Card(rank="2", suit="diamond", chip_value=2),
            Card(rank="3", suit="club", chip_value=3),
        )
        observation = {
            **make_observation(hand),
            "unseen_deck": (
                Card(rank="J", suit="heart", chip_value=10),
                Card(rank="10", suit="heart", chip_value=10),
            ),
        }
        legal_actions = make_all_legal_actions(len(hand))
        bot = PrunedSampledLookaheadBot(rng=random.Random(59), sample_count=6)

        chosen_action = bot.act(observation, legal_actions)
        decision_info = bot.get_last_decision_info()

        self.assertEqual(chosen_action, Action(type="play", card_indices=(0, 1, 2, 3)))
        self.assertIsNotNone(decision_info)
        assert decision_info is not None
        self.assertTrue(decision_info["four_of_a_kind_short_circuited"])
        self.assertEqual(decision_info["pruned_discard_candidate_count"], 0)
        self.assertEqual(decision_info["sampled_discard_candidate_count"], 0)
        self.assertEqual(decision_info["redraw_sample_count_used"], 0)


if __name__ == "__main__":
    unittest.main()
