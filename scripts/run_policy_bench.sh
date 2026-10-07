#!/usr/bin/env bash
set -euo pipefail
policy_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
policy_program="$policy_root/scripts/policy_bench/policy_runner.py"
policy_output_dir="${POLICY_OUTPUT_DIR:-$policy_root/policy_bench_runs/manual_$(date +%Y%m%d_%H%M%S)_$$}"
policy_python="${POLICY_PYTHON:-$policy_root/.venv-policy/bin/python}"
exec "$policy_python" "$policy_program" \
  --model "$policy_root/logs/legs_tracking/20260914_170827/20260914_170827.onnx" \
  --source serial --port /dev/ttyACM0 --baud 460800 --hz 50 --duration 30 \
  --out "$policy_output_dir" "$@"
