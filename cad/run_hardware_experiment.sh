#!/usr/bin/env bash
set -euo pipefail

cad_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$cad_dir/.." && pwd)"
baseline="$repo_dir/logs/legs_tracking/20260914_170827"
run_dir="${1:-$repo_dir/logs/legs_tracking_hardware/$(date +%Y%m%d_%H%M%S)}"
iterations="${2:-100000}"
num_envs="${3:-256}"

cd "$repo_dir/mjlab"
export PYTHONPATH="$cad_dir${PYTHONPATH:+:$PYTHONPATH}"

uv run --frozen --python 3.12 python "$cad_dir/train_hardware_tracking.py" \
  --motion "$baseline/reference.npz" --run-dir "$run_dir" \
  --iterations "$iterations" --num-envs "$num_envs" --save-interval 500

checkpoint="$(find "$run_dir" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)"
test -n "$checkpoint"
uv run --frozen --python 3.12 python "$cad_dir/verify_tracking_onnx.py" \
  "$run_dir/$(basename "$run_dir").onnx" "$run_dir/reference.npz" \
  > "$run_dir/onnx_verification.txt"

for condition in nominal latency extreme; do
  uv run --frozen --python 3.12 python "$cad_dir/evaluate_hardware_tracking.py" \
    "$checkpoint" --condition "$condition" --num-envs 64 --steps 500 \
    --output "$run_dir/evaluation_${condition}.json" \
    > "$run_dir/evaluation_${condition}.log"
  uv run --frozen --python 3.12 python "$cad_dir/evaluate_hardware_tracking.py" \
    "$baseline/model_100997.pt" --reference "$baseline/reference.npz" \
    --condition "$condition" --num-envs 64 --steps 500 \
    --output "$run_dir/baseline_${condition}.json" \
    > "$run_dir/baseline_${condition}.log"
done

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_dir/COMPLETED"
