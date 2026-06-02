#!/usr/bin/env bash
set -euo pipefail

MOTION_DIR="${MOTION_DIR:-/cephfs/holosoma/data/lafan/retargeted/ch_robot_stance_flatfoot_locomotion_full_floor_norm_with_vel}"
NUM_ENVS="${NUM_ENVS:-4096}"
LOG_ROOT="${LOG_ROOT:-logs/rl_games/humanoid_flat_direct}"
CHECKPOINT="${CHECKPOINT:-}"
RESUME_LAST="${RESUME_LAST:-0}"

WALK_CLIPS=(
  walk1_subject1_original_floor_norm_with_vel.npz
  walk1_subject2_original_floor_norm_with_vel.npz
  walk1_subject5_original_floor_norm_with_vel.npz
  walk2_subject1_original_floor_norm_with_vel.npz
  walk2_subject3_original_floor_norm_with_vel.npz
  walk2_subject4_original_floor_norm_with_vel.npz
  walk3_subject1_original_floor_norm_with_vel.npz
  walk3_subject2_original_floor_norm_with_vel.npz
  walk3_subject3_original_floor_norm_with_vel.npz
  walk3_subject4_original_floor_norm_with_vel.npz
  walk3_subject5_original_floor_norm_with_vel.npz
  walk4_subject1_original_floor_norm_with_vel.npz
)

if [[ ! -d "${MOTION_DIR}" ]]; then
  echo "[ERROR] Motion directory does not exist: ${MOTION_DIR}" >&2
  exit 1
fi

MOTION_MANIFEST="$(mktemp "${TMPDIR:-/tmp}/lafan_walk_manifest.XXXXXX")"
trap 'rm -f "${MOTION_MANIFEST}"' EXIT

missing=0
for clip in "${WALK_CLIPS[@]}"; do
  path="${MOTION_DIR}/${clip}"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing walk clip: ${path}" >&2
    missing=1
  else
    printf '%s\n' "${path}" >> "${MOTION_MANIFEST}"
  fi
done

if [[ "${missing}" != "0" ]]; then
  exit 1
fi

echo "[INFO] Training LAFAN walk tracking with ${#WALK_CLIPS[@]} clips"
echo "[INFO] Motion dir: ${MOTION_DIR}"
echo "[INFO] Manifest: ${MOTION_MANIFEST}"
echo "[INFO] Num envs: ${NUM_ENVS}"

if [[ "${CHECKPOINT}" == "latest" ]]; then
  RESUME_LAST=1
  CHECKPOINT=""
fi

if [[ -n "${CHECKPOINT}" && "${RESUME_LAST}" != "0" ]]; then
  echo "[ERROR] Set either CHECKPOINT or RESUME_LAST=1, not both." >&2
  exit 1
fi

if [[ "${RESUME_LAST}" != "0" ]]; then
  if [[ ! -d "${LOG_ROOT}" ]]; then
    echo "[ERROR] Cannot resume: log root does not exist: ${LOG_ROOT}" >&2
    exit 1
  fi
  CHECKPOINT="$(
    find "${LOG_ROOT}" -path '*/nn/*.pth' -type f -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | awk 'NR==1 {sub(/^[^ ]+ /, ""); print}'
  )"
  if [[ -z "${CHECKPOINT}" ]]; then
    echo "[ERROR] Cannot resume: no checkpoints found under ${LOG_ROOT}" >&2
    exit 1
  fi
fi

TRAIN_ARGS=(
  --task=Isaac-Humanoid-Locomotion-Flat-Direct-v0
  --num_envs="${NUM_ENVS}"
  --headless
)

if [[ -n "${MAX_ITERATIONS:-}" ]]; then
  TRAIN_ARGS+=(--max_iterations "${MAX_ITERATIONS}")
fi

if [[ -n "${CHECKPOINT}" ]]; then
  echo "[INFO] Resuming from checkpoint: ${CHECKPOINT}"
  TRAIN_ARGS+=(--checkpoint "${CHECKPOINT}")
fi

python scripts/rl_games/train.py \
  "${TRAIN_ARGS[@]}" \
  env.motion_reference_dir="${MOTION_DIR}" \
  env.motion_manifest_file="${MOTION_MANIFEST}" \
  env.motion_reference_playback=false \
  env.motion_random_start=true \
  env.motion_min_length_s=1.0 \
  env.motion_cache_on_gpu=true \
  "$@"
