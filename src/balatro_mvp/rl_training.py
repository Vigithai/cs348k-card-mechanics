"""Readable DQN-style training utilities for the Balatro MVP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import random
from typing import Any, Sequence

import torch
from torch import nn

from .environment import BalatroMVPEnvironment, ROUND_TARGET_PRESETS
from .rl import QNetwork, RLQBot, ReplayBuffer, ReplayTransition, build_state_action_feature_tensor
from .rl_features import STATE_ACTION_FEATURE_DIM


@dataclass(frozen=True)
class RLTrainingConfig:
    """Configuration for one RLQBot training run."""

    preset_name: str = "easy"
    num_episodes: int = 200
    eval_interval: int = 25
    eval_games: int = 20
    checkpoint_interval: int = 50
    seed: int = 0
    replay_capacity: int = 5000
    batch_size: int = 64
    warmup_transitions: int = 100
    gamma: float = 0.99
    learning_rate: float = 1e-3
    target_sync_interval: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2000
    train_every: int = 1
    hidden_dims: tuple[int, ...] = (128, 64)
    device: str = "cpu"
    output_dir: str = "results/rl"


@dataclass
class RLTrainingHistory:
    """Collected machine-readable metrics from one training run."""

    config: dict[str, Any]
    episode_metrics: list[dict[str, Any]] = field(default_factory=list)
    evaluation_metrics: list[dict[str, Any]] = field(default_factory=list)
    optimization_metrics: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_paths: list[str] = field(default_factory=list)


class RLQTrainer:
    """Self-contained DQN-style trainer for the Balatro MVP environment."""

    def __init__(self, config: RLTrainingConfig) -> None:
        if config.preset_name not in ROUND_TARGET_PRESETS:
            raise ValueError(
                f"Unsupported preset {config.preset_name!r}. Expected one of {sorted(ROUND_TARGET_PRESETS)}."
            )
        if config.num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if config.replay_capacity < config.batch_size:
            raise ValueError("replay_capacity must be at least batch_size.")
        if config.eval_interval <= 0:
            raise ValueError("eval_interval must be positive.")
        if config.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive.")
        if config.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive.")
        if config.train_every <= 0:
            raise ValueError("train_every must be positive.")

        self.config = config
        self.device = torch.device(config.device)
        self.round_chip_targets = dict(ROUND_TARGET_PRESETS[config.preset_name])
        self.output_dir = Path(config.output_dir) / config.preset_name / f"seed_{config.seed}"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(config.seed)

        self.policy_network = QNetwork(
            input_dim=STATE_ACTION_FEATURE_DIM,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        self.target_network = QNetwork(
            input_dim=STATE_ACTION_FEATURE_DIM,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(config.replay_capacity, rng=random.Random(config.seed + 1))
        self.training_bot = RLQBot(
            self.policy_network,
            rng=random.Random(config.seed + 2),
            epsilon=config.epsilon_start,
            training=True,
            device=self.device,
        )

        self.global_step = 0
        self.history = RLTrainingHistory(config=asdict(config))

    def train(self) -> RLTrainingHistory:
        """Run the configured training loop and return saved metrics."""

        for episode_index in range(1, self.config.num_episodes + 1):
            episode_metrics = self._run_training_episode(episode_index)
            self.history.episode_metrics.append(episode_metrics)

            if episode_index % self.config.eval_interval == 0 or episode_index == self.config.num_episodes:
                evaluation_summary = self.evaluate_policy(
                    num_games=self.config.eval_games,
                    base_seed=100_000 + self.config.seed + episode_index,
                )
                evaluation_summary["episode_index"] = episode_index
                self.history.evaluation_metrics.append(evaluation_summary)

            if episode_index % self.config.checkpoint_interval == 0 or episode_index == self.config.num_episodes:
                checkpoint_path = self.save_checkpoint(episode_index)
                self.history.checkpoint_paths.append(str(checkpoint_path))

        self.save_metrics()
        return self.history

    def evaluate_policy(self, *, num_games: int, base_seed: int) -> dict[str, Any]:
        """Evaluate the current greedy policy over a seeded batch of games."""

        wins = 0
        total_rounds_passed = 0
        final_chip_scores: list[int] = []

        for game_index in range(num_games):
            seed = base_seed + game_index
            env = BalatroMVPEnvironment(seed=seed, round_chip_targets=self.round_chip_targets)
            evaluation_bot = RLQBot(
                self.policy_network,
                rng=random.Random(seed),
                epsilon=0.0,
                training=False,
                device=self.device,
            )

            done = False
            rounds_passed = 0
            while not done:
                observation = env.get_observation()
                legal_actions = env.get_legal_actions()
                action = evaluation_bot.act(observation, legal_actions)
                _, _, done, info = env.step(action)
                if info.get("round_result") == "round_win":
                    rounds_passed += 1

            if env.state is None:
                raise RuntimeError("Environment state missing after evaluation.")
            if env.state.result == "run_win":
                wins += 1
            total_rounds_passed += rounds_passed
            final_chip_scores.append(env.state.chips_scored)

        average_final_chips = sum(final_chip_scores) / num_games
        return {
            "num_games": num_games,
            "base_seed": base_seed,
            "win_rate": wins / num_games,
            "average_rounds_passed": total_rounds_passed / num_games,
            "average_final_chips_scored": average_final_chips,
            "average_episode_reward": average_final_chips,
        }

    def save_checkpoint(self, episode_index: int) -> Path:
        """Write one model checkpoint and return its path."""

        checkpoint_path = self.checkpoint_dir / f"episode_{episode_index:04d}.pt"
        checkpoint_payload = {
            "episode_index": episode_index,
            "global_step": self.global_step,
            "input_dim": STATE_ACTION_FEATURE_DIM,
            "config": {
                **asdict(self.config),
                "hidden_dims": list(self.config.hidden_dims),
            },
            "round_chip_targets": self.round_chip_targets,
            "policy_state_dict": self.policy_network.state_dict(),
            "target_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint_payload, checkpoint_path)
        return checkpoint_path

    def save_metrics(self) -> Path:
        """Persist the collected training metrics to JSON."""

        import json

        metrics_path = self.output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(
                {
                    "config": self.history.config,
                    "episode_metrics": self.history.episode_metrics,
                    "evaluation_metrics": self.history.evaluation_metrics,
                    "optimization_metrics": self.history.optimization_metrics,
                    "checkpoint_paths": self.history.checkpoint_paths,
                },
                metrics_file,
                indent=2,
                sort_keys=True,
            )
        return metrics_path

    def _run_training_episode(self, episode_index: int) -> dict[str, Any]:
        """Collect one training episode and add its transitions to replay."""

        episode_seed = self.config.seed + episode_index - 1
        env = BalatroMVPEnvironment(seed=episode_seed, round_chip_targets=self.round_chip_targets)
        done = False
        episode_reward = 0.0
        rounds_passed = 0
        turn_count = 0

        while not done:
            observation = env.get_observation()
            legal_actions = env.get_legal_actions()
            epsilon = self._epsilon_for_step(self.global_step)
            self.training_bot.epsilon = epsilon
            action = self.training_bot.act(observation, legal_actions)
            next_observation, reward, done, info = env.step(action)
            next_legal_actions = tuple(env.get_legal_actions())
            self.replay_buffer.add(
                ReplayTransition(
                    observation=observation,
                    action=action,
                    reward=float(reward),
                    next_observation=next_observation,
                    next_legal_actions=next_legal_actions,
                    done=done,
                )
            )

            self.global_step += 1
            turn_count += 1
            episode_reward += reward
            if info.get("round_result") == "round_win":
                rounds_passed += 1

            if len(self.replay_buffer) >= max(self.config.batch_size, self.config.warmup_transitions):
                optimization_summary = self._maybe_optimize()
                if optimization_summary is not None:
                    self.history.optimization_metrics.append(optimization_summary)

            if self.global_step > 0 and self.global_step % self.config.target_sync_interval == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

        if env.state is None:
            raise RuntimeError("Environment state missing after training episode.")

        return {
            "episode_index": episode_index,
            "seed": episode_seed,
            "episode_reward": episode_reward,
            "final_chips_scored": env.state.chips_scored,
            "rounds_passed": rounds_passed,
            "run_result": env.state.result,
            "turn_count": turn_count,
            "epsilon_end_of_episode": self._epsilon_for_step(self.global_step),
        }

    def _maybe_optimize(self) -> dict[str, Any] | None:
        """Run one optimizer step when the train cadence permits it."""

        if self.global_step % self.config.train_every != 0:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        current_feature_rows = [
            build_state_action_feature_tensor(
                transition.observation,
                [transition.action],
                device=self.device,
            ).squeeze(0)
            for transition in batch
        ]
        current_features = torch.stack(current_feature_rows, dim=0)
        current_q_values = self.policy_network(current_features)

        target_values: list[float] = []
        for transition in batch:
            next_state_value = 0.0
            if not transition.done and transition.next_legal_actions:
                next_feature_tensor = build_state_action_feature_tensor(
                    transition.next_observation,
                    transition.next_legal_actions,
                    device=self.device,
                )
                with torch.no_grad():
                    next_state_value = float(torch.max(self.target_network(next_feature_tensor)).item())
            target_values.append(transition.reward + self.config.gamma * next_state_value)

        target_tensor = torch.tensor(target_values, dtype=torch.float32, device=self.device)
        loss = self.loss_fn(current_q_values, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "global_step": self.global_step,
            "loss": float(loss.item()),
        }

    def _epsilon_for_step(self, global_step: int) -> float:
        """Linearly decay epsilon from the configured start to end value."""

        progress = min(1.0, global_step / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)


def default_training_config(
    *,
    preset_name: str = "easy",
    output_dir: str = "results/rl",
    seed: int = 0,
    hidden_dims: Sequence[int] = (128, 64),
) -> RLTrainingConfig:
    """Build the default RL training configuration used by the CLI script."""

    return RLTrainingConfig(
        preset_name=preset_name,
        output_dir=output_dir,
        seed=seed,
        hidden_dims=tuple(hidden_dims),
    )


__all__ = [
    "RLQTrainer",
    "RLTrainingConfig",
    "RLTrainingHistory",
    "default_training_config",
]
