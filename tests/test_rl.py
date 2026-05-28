from __future__ import annotations

from pathlib import Path
import random
import sys
import unittest

import torch


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp import (
    ACTION_FEATURE_DIM,
    OBSERVATION_FEATURE_DIM,
    STATE_ACTION_FEATURE_DIM,
    Action,
    Card,
    QNetwork,
    RLQBot,
    ReplayBuffer,
    ReplayTransition,
    extract_action_features,
    extract_observation_features,
    extract_state_action_features,
)


def make_observation() -> dict[str, object]:
    return {
        "chips_needed": 150,
        "chips_scored": 40,
        "hands_left": 3,
        "discards_left": 2,
        "round_index": 1,
        "target_hand_size": 7,
        "hand": (
            Card(rank="2", suit="club", chip_value=2),
            Card(rank="2", suit="heart", chip_value=2),
            Card(rank="A", suit="spade", chip_value=11),
        ),
        "unseen_deck": (
            Card(rank="5", suit="diamond", chip_value=5),
            Card(rank="K", suit="spade", chip_value=10),
        ),
    }


class FeatureExtractionTests(unittest.TestCase):
    def test_feature_extraction_shapes_are_stable(self) -> None:
        observation = make_observation()
        play_action = Action(type="play", card_indices=(0, 1))
        discard_action = Action(type="discard", card_indices=(2,))

        observation_features = extract_observation_features(observation)
        play_action_features = extract_action_features(observation, play_action)
        discard_action_features = extract_action_features(observation, discard_action)
        state_action_features = extract_state_action_features(observation, play_action)

        self.assertEqual(len(observation_features), OBSERVATION_FEATURE_DIM)
        self.assertEqual(len(play_action_features), ACTION_FEATURE_DIM)
        self.assertEqual(len(discard_action_features), ACTION_FEATURE_DIM)
        self.assertEqual(len(state_action_features), STATE_ACTION_FEATURE_DIM)
        self.assertEqual(sum(observation_features[4:17]), 3.0)
        self.assertEqual(sum(observation_features[17:21]), 3.0)
        self.assertEqual(play_action_features[-1], 20.0)
        self.assertEqual(discard_action_features[-1], 0.0)


class RLQBotTests(unittest.TestCase):
    def test_rlqbot_always_returns_a_legal_action(self) -> None:
        observation = make_observation()
        legal_actions = [
            Action(type="play", card_indices=(0,)),
            Action(type="play", card_indices=(0, 1)),
            Action(type="discard", card_indices=(2,)),
        ]
        q_network = QNetwork(input_dim=STATE_ACTION_FEATURE_DIM, hidden_dims=(16, 8))
        bot = RLQBot(q_network, rng=random.Random(7), epsilon=1.0, training=True)

        for _ in range(50):
            self.assertIn(bot.act(observation, legal_actions), legal_actions)

    def test_deterministic_argmax_choice_works_with_fixed_weights_and_zero_epsilon(self) -> None:
        observation = make_observation()
        legal_actions = [
            Action(type="play", card_indices=(0,)),
            Action(type="play", card_indices=(0, 1)),
            Action(type="discard", card_indices=(2,)),
        ]
        q_network = torch.nn.Linear(STATE_ACTION_FEATURE_DIM, 1, bias=False)
        with torch.no_grad():
            q_network.weight.zero_()
            q_network.weight[0, STATE_ACTION_FEATURE_DIM - 1] = 1.0

        bot = RLQBot(q_network, rng=random.Random(11), epsilon=0.0, training=False)
        chosen_action = bot.act(observation, legal_actions)

        self.assertEqual(chosen_action, Action(type="play", card_indices=(0, 1)))


class ReplayBufferTests(unittest.TestCase):
    def test_replay_buffer_stores_and_evicts_transitions_correctly(self) -> None:
        observation = make_observation()
        transition_one = ReplayTransition(
            observation=observation,
            action=Action(type="play", card_indices=(0,)),
            reward=5.0,
            next_observation=observation,
            next_legal_actions=(Action(type="play", card_indices=(1,)),),
            done=False,
        )
        transition_two = ReplayTransition(
            observation=observation,
            action=Action(type="discard", card_indices=(2,)),
            reward=0.0,
            next_observation=observation,
            next_legal_actions=(Action(type="play", card_indices=(0, 1)),),
            done=False,
        )
        transition_three = ReplayTransition(
            observation=observation,
            action=Action(type="play", card_indices=(0, 1)),
            reward=20.0,
            next_observation=observation,
            next_legal_actions=(),
            done=True,
        )
        replay_buffer = ReplayBuffer(2, rng=random.Random(13))

        replay_buffer.add(transition_one)
        replay_buffer.add(transition_two)
        replay_buffer.add(transition_three)

        self.assertEqual(len(replay_buffer), 2)
        sampled_batch = replay_buffer.sample(2)
        sampled_actions = {transition.action for transition in sampled_batch}
        self.assertNotIn(transition_one.action, sampled_actions)
        self.assertEqual(
            sampled_actions,
            {transition_two.action, transition_three.action},
        )


if __name__ == "__main__":
    unittest.main()
