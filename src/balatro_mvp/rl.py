"""RL agent and utility primitives for the Balatro MVP."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from .agents import Agent
from .environment import Action
from .rl_features import STATE_ACTION_FEATURE_DIM, extract_state_action_features


@dataclass(frozen=True)
class ReplayTransition:
    """One replay-buffer transition for DQN-style training."""

    observation: dict[str, Any]
    action: Action
    reward: float
    next_observation: dict[str, Any]
    next_legal_actions: tuple[Action, ...]
    done: bool


class ReplayBuffer:
    """Small replay buffer with deterministic sampling via a provided RNG."""

    def __init__(self, capacity: int, *, rng: random.Random | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self.rng = rng if rng is not None else random.Random()
        self._items: deque[ReplayTransition] = deque(maxlen=capacity)

    def add(self, transition: ReplayTransition) -> None:
        """Append one transition, dropping the oldest item when full."""

        self._items.append(transition)

    def sample(self, batch_size: int) -> list[ReplayTransition]:
        """Return a random batch sampled without replacement."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > len(self._items):
            raise ValueError("Cannot sample more items than currently stored.")
        return self.rng.sample(list(self._items), k=batch_size)

    def __len__(self) -> int:
        return len(self._items)


class QNetwork(nn.Module):
    """Simple MLP that scores concatenated state-action feature vectors."""

    def __init__(
        self,
        input_dim: int = STATE_ACTION_FEATURE_DIM,
        hidden_dims: Sequence[int] = (128, 64),
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size.")
        if any(hidden_dim <= 0 for hidden_dim in hidden_dims):
            raise ValueError("hidden_dims must contain positive sizes only.")

        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, state_action_features: torch.Tensor) -> torch.Tensor:
        """Return scalar Q-values for one batch of state-action feature rows."""

        return self.model(state_action_features).squeeze(-1)


class RLQBot(Agent):
    """Choose among legal actions by scoring each (state, action) pair with Q(s, a)."""

    def __init__(
        self,
        q_network: nn.Module,
        *,
        rng: random.Random | None = None,
        epsilon: float = 0.0,
        training: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must be between 0.0 and 1.0.")
        self.q_network = q_network
        self.rng = rng if rng is not None else random.Random()
        self.epsilon = epsilon
        self.training = training
        self.device = torch.device(device)
        self.q_network.to(self.device)
        self._last_decision_info: dict[str, Any] | None = None

    def act(self, observation: dict[str, Any], legal_actions: list[Action]) -> Action:
        """Choose epsilon-greedily in training mode or greedily in evaluation mode."""

        if not legal_actions:
            raise ValueError("RLQBot requires at least one legal action.")

        if self.training and self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            chosen_action = self.rng.choice(legal_actions)
            self._last_decision_info = {
                "epsilon": self.epsilon,
                "exploratory": True,
                "legal_action_count": len(legal_actions),
            }
            return chosen_action

        q_values = self.score_legal_actions(observation, legal_actions)
        best_index = max(range(len(legal_actions)), key=lambda index: q_values[index])
        chosen_action = legal_actions[best_index]
        self._last_decision_info = {
            "epsilon": self.epsilon,
            "exploratory": False,
            "legal_action_count": len(legal_actions),
            "chosen_q_value": q_values[best_index],
        }
        return chosen_action

    def score_legal_actions(
        self,
        observation: dict[str, Any],
        legal_actions: Sequence[Action],
    ) -> list[float]:
        """Return one Q-value per legal action in the given order."""

        feature_tensor = build_state_action_feature_tensor(
            observation,
            legal_actions,
            device=self.device,
        )
        current_training_mode = self.q_network.training
        self.q_network.eval()
        with torch.no_grad():
            q_tensor = torch.as_tensor(self.q_network(feature_tensor)).squeeze(-1)
            q_values = q_tensor.detach().cpu().tolist()
        if current_training_mode:
            self.q_network.train()
        if isinstance(q_values, float):
            return [q_values]
        return [float(value) for value in q_values]

    def get_last_decision_info(self) -> dict[str, Any] | None:
        if self._last_decision_info is None:
            return None
        return dict(self._last_decision_info)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        rng: random.Random | None = None,
        epsilon: float = 0.0,
        training: bool = False,
        device: str | torch.device = "cpu",
    ) -> RLQBot:
        """Load a saved policy checkpoint and wrap it in an RLQBot."""

        q_network, _ = load_q_network_from_checkpoint(checkpoint_path, device=device)
        return cls(
            q_network,
            rng=rng,
            epsilon=epsilon,
            training=training,
            device=device,
        )


def build_state_action_feature_tensor(
    observation: dict[str, Any],
    legal_actions: Iterable[Action],
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a tensor of concatenated state-action features for one legal-action list."""

    feature_rows = [extract_state_action_features(observation, action) for action in legal_actions]
    if not feature_rows:
        raise ValueError("At least one legal action is required to build a feature tensor.")
    return torch.tensor(feature_rows, dtype=torch.float32, device=device)


def load_q_network_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[QNetwork, dict[str, Any]]:
    """Load a saved policy network and its raw checkpoint payload."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=torch.device(device))
    config = checkpoint.get("config", {})
    hidden_dims = tuple(config.get("hidden_dims", (128, 64)))
    input_dim = int(checkpoint.get("input_dim", STATE_ACTION_FEATURE_DIM))
    q_network = QNetwork(input_dim=input_dim, hidden_dims=hidden_dims)
    q_network.load_state_dict(checkpoint["policy_state_dict"])
    q_network.to(torch.device(device))
    q_network.eval()
    return q_network, checkpoint
