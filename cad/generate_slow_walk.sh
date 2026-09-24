#!/usr/bin/env bash
set -euo pipefail
cad_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
kimodo_dir="$(cd -- "$cad_dir/../../kimodo" && pwd)"
mkdir -p "$cad_dir/motions"
export TEXT_ENCODER_DEVICE=cpu
export TEXT_ENCODER_MODE=local
"$kimodo_dir/.venv/bin/kimodo_gen" \
  'A person walks slowly forward on flat ground with short natural steps and an upright posture, keeping a steady direction.' \
  --model Kimodo-SOMA-RP-v1.1 --duration 6.0 --seed 42 \
  --num_samples 1 --bvh --bvh_standard_tpose \
  --output "$cad_dir/motions/slow_walk_seed42"
