#!/usr/bin/env bash
set -euo pipefail

NUM_ENVS="${NUM_ENVS:-16384}"
FAST_TRAIN="${FAST_TRAIN:-0}"
DOMAIN_RANDOMIZATION_MODE="${DOMAIN_RANDOMIZATION_MODE:-}"
SOLVER_POSITION_ITERATIONS="${SOLVER_POSITION_ITERATIONS:-}"
SOLVER_VELOCITY_ITERATIONS="${SOLVER_VELOCITY_ITERATIONS:-}"
if [[ "${FAST_TRAIN}" != "0" && ( "${DOMAIN_RANDOMIZATION_MODE}" == "adaptive" || "${DOMAIN_RANDOMIZATION_MODE}" == "fixed" ) ]]; then
  echo "FAST_TRAIN disables domain randomization; use DOMAIN_RANDOMIZATION_MODE=off or leave FAST_TRAIN=0." >&2
  exit 2
fi

PYTHON_CMD=()
if [[ -n "${ISAAC_PYTHON:-}" ]]; then
  PYTHON_CMD=("${ISAAC_PYTHON}")
elif [[ -n "${ISAACSIM_ROOT_PATH:-}" && -x "${ISAACSIM_ROOT_PATH}/python.sh" ]]; then
  PYTHON_CMD=(env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL "${ISAACSIM_ROOT_PATH}/python.sh")
elif [[ -x "/workspace/isaaclab/_isaac_sim/python.sh" ]]; then
  PYTHON_CMD=(env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL "/workspace/isaaclab/_isaac_sim/python.sh")
elif [[ -x "/isaac-sim/python.sh" ]]; then
  PYTHON_CMD=(env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL "/isaac-sim/python.sh")
else
  PYTHON_CMD=("${PYTHON:-python}")
fi

EXTRA_OVERRIDES=()
if [[ "${FAST_TRAIN}" != "0" ]]; then
  EXTRA_OVERRIDES+=(
    env.events=null
    env.enable_adr=false
    env.enable_contact_sensor=false
    env.enable_reward_logging=false
    env.push_force_range=[0.0,0.0]
    env.reset_joint_pos_noise=0.0
    env.reset_joint_vel_noise=0.0
    env.action_max_latency=0
    env.obs_max_latency=0
    env.robot.spawn.activate_contact_sensors=false
    env.robot.spawn.articulation_props.enabled_self_collisions=false
  )
fi

if [[ -n "${SOLVER_POSITION_ITERATIONS}" ]]; then
  EXTRA_OVERRIDES+=(env.robot.spawn.articulation_props.solver_position_iteration_count="${SOLVER_POSITION_ITERATIONS}")
fi

if [[ -n "${SOLVER_VELOCITY_ITERATIONS}" ]]; then
  EXTRA_OVERRIDES+=(env.robot.spawn.articulation_props.solver_velocity_iteration_count="${SOLVER_VELOCITY_ITERATIONS}")
fi

if [[ -n "${DOMAIN_RANDOMIZATION_MODE}" ]]; then
  case "${DOMAIN_RANDOMIZATION_MODE}" in
    adaptive|fixed)
      EXTRA_OVERRIDES+=(
        env.enable_adr=true
        env.domain_randomization_mode="${DOMAIN_RANDOMIZATION_MODE}"
      )
      ;;
    off|none|disabled)
      EXTRA_OVERRIDES+=(env.enable_adr=false)
      ;;
    *)
      echo "DOMAIN_RANDOMIZATION_MODE must be adaptive, fixed, or off; got '${DOMAIN_RANDOMIZATION_MODE}'." >&2
      exit 2
      ;;
  esac
fi

"${PYTHON_CMD[@]}" scripts/rl_games/train.py \
  --task=Isaac-Humanoid-Locomotion-Flat-Direct-v0 \
  --num_envs="${NUM_ENVS}" \
  --headless \
  "${EXTRA_OVERRIDES[@]}" \
  "$@"
