from __future__ import annotations

from pathlib import Path
import random
import sys
import unittest


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp import Action, Card, RandomBot, StimBot


class StubChoiceRng:
    """Small deterministic RNG stub for tie-breaking tests."""

    def __init__(self, choice_index: int) -> None:
        self.choice_index = choice_index

    def choice(self, items: list[Action]) -> Action:
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


if __name__ == "__main__":
    unittest.main()
