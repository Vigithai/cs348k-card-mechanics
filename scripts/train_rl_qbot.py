"""Train RLQBot with a small DQN-style state-action scorer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from balatro_mvp import DEFAULT_MAX_ANTE
from balatro_mvp.rl_training import RLQTrainer, RLTrainingConfig


def main() -> None:
    """Parse CLI arguments, train RLQBot, and save checkpoints plus metrics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-ante",
        type=int,
        default=DEFAULT_MAX_ANTE,
        help="Maximum ante to win the run (default 8).",
    )
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for training.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "rl_training",
        help="Root directory for RL checkpoints and metrics.",
    )
    parser.add_argument("--eval-interval", type=int, default=25, help="Evaluate every N episodes.")
    parser.add_argument("--eval-games", type=int, default=20, help="Games per periodic evaluation.")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save a checkpoint every N episodes.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Replay batch size.")
    parser.add_argument("--replay-capacity", type=int, default=5000, help="Replay buffer capacity.")
    parser.add_argument(
        "--warmup-transitions",
        type=int,
        default=100,
        help="Replay items required before optimization starts.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument(
        "--target-sync-interval",
        type=int,
        default=100,
        help="How many environment steps between target-network syncs.",
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon.")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon.")
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=2000,
        help="Linear epsilon decay horizon in environment steps.",
    )
    parser.add_argument(
        "--train-every",
        type=int,
        default=1,
        help="Run one optimizer step every N environment steps after warmup.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Hidden layer sizes for the Q-network MLP.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use for training.",
    )
    parser.add_argument(
        "--round-win-bonus",
        type=float,
        default=0.0,
        help=(
            "Extra reward added when the agent clears a round target. "
            "Set to a positive value (e.g. 150) to shape towards round completion."
        ),
    )
    parser.add_argument(
        "--round-loss-penalty",
        type=float,
        default=0.0,
        help=(
            "Reward delta (should be negative, e.g. -75) applied when a round is lost. "
            "Magnitude should be smaller than round-win-bonus to keep the agent exploring."
        ),
    )
    args = parser.parse_args()

    config = RLTrainingConfig(
        max_ante=args.max_ante,
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        warmup_transitions=args.warmup_transitions,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        target_sync_interval=args.target_sync_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        train_every=args.train_every,
        hidden_dims=tuple(args.hidden_dims),
        device=args.device,
        output_dir=str(args.output_dir),
        round_win_bonus=args.round_win_bonus,
        round_loss_penalty=args.round_loss_penalty,
    )
    trainer = RLQTrainer(config)
    history = trainer.train()

    last_evaluation = history.evaluation_metrics[-1] if history.evaluation_metrics else None
    final_checkpoint = history.checkpoint_paths[-1] if history.checkpoint_paths else "none"
    print(f"Max ante: {args.max_ante}")
    print(f"Episodes: {args.episodes}")
    print(f"Output directory: {trainer.output_dir}")
    print(f"Final checkpoint: {final_checkpoint}")
    if last_evaluation is not None:
        print(f"Last eval win rate: {last_evaluation['win_rate']:.2%}")
        print(f"Last eval avg rounds passed: {last_evaluation['average_rounds_passed']:.2f}")
        print(f"Last eval avg final chips: {last_evaluation['average_final_chips_scored']:.2f}")


if __name__ == "__main__":
    main()
