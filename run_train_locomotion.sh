#!/usr/bin/env bash
set -euo pipefail

PYTHON_CMD=()
if [[ -n "${ISAAC_PYTHON:-}" ]]; then
  PYTHON_CMD=("${ISAAC_PYTHON}")
elif [[ -n "${ISAACSIM_ROOT_PATH:-}" && -x "${ISAACSIM_ROOT_PATH}/python.sh" ]]; then
  PYTHON_CMD=(env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL "${ISAACSIM_ROOT_PATH}/python.sh")
elif [[ -x "/isaac-sim/python.sh" ]]; then
  PYTHON_CMD=(env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL "/isaac-sim/python.sh")
elif [[ -n "${ISAACLAB_PATH:-}" && -x "${ISAACLAB_PATH}/isaaclab.sh" && -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_CMD=("${ISAACLAB_PATH}/isaaclab.sh" "-p")
else
  PYTHON_CMD=("${PYTHON:-python}")
fi

"${PYTHON_CMD[@]}" scripts/rl_games/train.py \
  --task=Isaac-Humanoid-Locomotion-Flat-Direct-v0 \
  --num_envs=8192 \
  --headless \
  "$@"
