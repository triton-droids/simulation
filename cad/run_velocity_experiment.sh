#!/usr/bin/env bash
set -euo pipefail
# Train velocity-command locomotion for the Triton lower body in mjlab.
# Usage: cad/run_velocity_experiment.sh [extra train_velocity.py args...]
cad_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$cad_dir/.." && pwd)"
uv_bin="${UV_BIN:-/home/thewheelneedsme/.local/bin/uv}"
export PYTHONPATH="$cad_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$repo_dir/mjlab"
"$uv_bin" run --no-sync python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
exec "$uv_bin" run --no-sync python "$cad_dir/train_velocity.py" "$@"
