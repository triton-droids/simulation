#!/bin/bash
set -euo pipefail
# Run from repo root inside a pinned Isaac Lab3.0 Python environment.
python robocub/tools/benchmark_field01.py --robot g1 --counts 1 16 256 --output "${1:?fresh output path required}"
# 1024/4096 only after memory headroom and smaller cases pass.
