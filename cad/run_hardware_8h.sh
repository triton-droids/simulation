#!/usr/bin/env bash
set -euo pipefail
cad_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$cad_dir/.." && pwd)"
run_dir="${1:-$repo_dir/logs/legs_tracking_hardware/$(date +%Y%m%d_%H%M%S)_8h}"
cd "$repo_dir/mjlab"
uv_bin="${UV_BIN:-/home/thewheelneedsme/.local/bin/uv}"
export PYTHONPATH="$cad_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
"$uv_bin" run --no-sync python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
# Budget checked at checkpoint saves; final evaluations add a few minutes.
"$uv_bin" run --no-sync python "$cad_dir/train_hardware_tracking.py" --run-dir "$run_dir" --num-envs 256 --iterations 1000000 --hours 8 --save-interval 500 --evaluation-interval 5000
"$uv_bin" run --no-sync python "$cad_dir/verify_tracking_onnx.py" "$run_dir/best.onnx" "$run_dir/reference.npz" > "$run_dir/onnx_verification.txt"
baseline="$repo_dir/logs/legs_tracking/20260914_170827"
for condition in nominal latency extreme; do
  "$uv_bin" run --no-sync python "$cad_dir/evaluate_hardware_tracking.py" "$run_dir/best.onnx" --condition "$condition" --num-envs 64 --steps 500 --output "$run_dir/evaluation_${condition}.json" > "$run_dir/evaluation_${condition}.log" 2>&1
  "$uv_bin" run --no-sync python "$cad_dir/evaluate_hardware_tracking.py" "$baseline/20260914_170827.onnx" --condition "$condition" --num-envs 64 --steps 500 --output "$run_dir/baseline_${condition}.json" > "$run_dir/baseline_${condition}.log" 2>&1
done
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_dir/COMPLETED"
echo "Finished. Selected policy: $run_dir/best.onnx"
