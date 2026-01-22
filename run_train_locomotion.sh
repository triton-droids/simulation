#!/usr/bin/env bash
set -euo pipefail

python scripts/rl_games/train.py \
  --task=Isaac-Humanoid-Locomotion-Flat-Direct-v0 \
  --num_envs=8192 \
  --headless \
  "$@"
