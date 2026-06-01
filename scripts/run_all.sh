#!/usr/bin/env bash
# Run the full evaluation pipeline for a given max-ante setting.
#
# Usage:
#   bash scripts/run_all.sh [options]
#
# Options:
#   --max-ante N          Ante ceiling for the run (default: 2)
#   --episodes N          Training episodes per seed (default: 500)
#   --games N             Games per bot for scripted + RL eval (default: 200)
#   --seeds "0 1 2"       Space-separated seed list (default: "0 1 2")
#   --round-win-bonus F   Reward shaping bonus for clearing a blind (default: 300)
#   --round-loss-penalty F  Reward penalty for failing a blind (default: -75)
#   --skip-scripted       Skip step 1 (scripted eval) — useful if already done
#   --skip-train          Skip step 2 (RL training) — evaluate existing checkpoints
#
# Example — full run on ante 2 with defaults:
#   bash scripts/run_all.sh --max-ante 2
#
# Example — re-evaluate existing checkpoints without retraining:
#   bash scripts/run_all.sh --max-ante 2 --skip-train

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# ── Defaults ──────────────────────────────────────────────────────────────────
MAX_ANTE=2
EPISODES=500
GAMES=200
SEEDS="0 1 2"
ROUND_WIN_BONUS=300
ROUND_LOSS_PENALTY=-75
SKIP_SCRIPTED=0
SKIP_TRAIN=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-ante)           MAX_ANTE="$2";           shift 2 ;;
    --episodes)           EPISODES="$2";           shift 2 ;;
    --games)              GAMES="$2";              shift 2 ;;
    --seeds)              SEEDS="$2";              shift 2 ;;
    --round-win-bonus)    ROUND_WIN_BONUS="$2";    shift 2 ;;
    --round-loss-penalty) ROUND_LOSS_PENALTY="$2"; shift 2 ;;
    --skip-scripted)      SKIP_SCRIPTED=1;         shift ;;
    --skip-train)         SKIP_TRAIN=1;            shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

echo "========================================================"
echo "  Balatro MVP — Full Evaluation Pipeline"
echo "========================================================"
echo "  max_ante:           $MAX_ANTE"
echo "  episodes:           $EPISODES"
echo "  games:              $GAMES"
echo "  seeds:              $SEEDS"
echo "  round_win_bonus:    $ROUND_WIN_BONUS"
echo "  round_loss_penalty: $ROUND_LOSS_PENALTY"
echo "========================================================"
echo ""

# ── Step 1: Scripted bot eval ─────────────────────────────────────────────────
if [[ $SKIP_SCRIPTED -eq 0 ]]; then
  echo "── Step 1/4: Scripted bot evaluation (ante $MAX_ANTE, $GAMES games) ──"
  python3 scripts/run_seeded_simulations.py \
    --max-ante "$MAX_ANTE" \
    --games "$GAMES"
  echo ""
else
  echo "── Step 1/4: Skipped (--skip-scripted) ──"
  echo ""
fi

# ── Step 2: Train RLQBot (3 seeds) ───────────────────────────────────────────
if [[ $SKIP_TRAIN -eq 0 ]]; then
  echo "── Step 2/4: Training RLQBot ──"
  for SEED in $SEEDS; do
    echo "  Training seed $SEED ..."
    python3 scripts/train_rl_qbot.py \
      --max-ante "$MAX_ANTE" \
      --episodes "$EPISODES" \
      --seed "$SEED" \
      --round-win-bonus "$ROUND_WIN_BONUS" \
      --round-loss-penalty "$ROUND_LOSS_PENALTY"
  done
  echo ""
else
  echo "── Step 2/4: Skipped (--skip-train) ──"
  echo ""
fi

# ── Step 3: Evaluate each seed's final checkpoint ────────────────────────────
echo "── Step 3/4: Evaluating RLQBot checkpoints ──"
for SEED in $SEEDS; do
  CHECKPOINT="results/rl_training/ante_${MAX_ANTE}/seed_${SEED}/checkpoints/episode_$(printf '%04d' $EPISODES).pt"
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "  WARNING: checkpoint not found: $CHECKPOINT — skipping seed $SEED"
    continue
  fi
  echo "  Evaluating seed $SEED ($CHECKPOINT) ..."
  python3 scripts/evaluate_rl_qbot.py \
    --checkpoint "$CHECKPOINT" \
    --max-ante "$MAX_ANTE" \
    --seed "$SEED" \
    --games "$GAMES"
done
echo ""

# ── Step 4: Aggregate and plot ────────────────────────────────────────────────
echo "── Step 4/4: Summarising and plotting ──"
python3 scripts/summarize_rl_seed_results.py
python3 scripts/plot_rl_seed_summary.py
echo ""

echo "========================================================"
echo "  Done."
echo "  results/scripted_eval/ante_${MAX_ANTE}_eval_results.json"
echo "  results/rl_training/ante_${MAX_ANTE}/seed_*/metrics.json"
echo "  results/rl_eval/ante_${MAX_ANTE}_comparison_seed_*.json"
echo "  results/rl_eval/seed_summary.csv"
echo "  results/figures/ante_${MAX_ANTE}_*.png"
echo "========================================================"
