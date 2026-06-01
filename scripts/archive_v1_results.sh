#!/bin/bash
# Run from the repo root: bash scripts/archive_v1_results.sh
# Removes stale v1 results from results/ (already copied to results_v1/).
# Safe to re-run — only removes files that exist.

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "Cleaning stale results from results/ ..."

rm -rf \
  results/checkpoint2_eval_results.json \
  results/easy_eval_results.json \
  results/hard_eval_results.json \
  results/figures \
  results/rl \
  results/rl_shaped \
  results/rl_summary \
  results/traces

# Create clean directory structure for v2 results
mkdir -p \
  results/scripted_eval \
  results/rl_training \
  results/rl_eval \
  results/traces \
  results/figures

echo "Done. New results/ structure:"
find results/ -type d | sort
