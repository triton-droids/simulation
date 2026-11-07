#!/bin/bash
#SBATCH --job-name=field01-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=field01-%j.log
set -euo pipefail
: "${FIELD01_REPO:?Set repository checkout path}"
: "${FIELD01_PYTHON:?Set pinned environment Python path}"
cd "$FIELD01_REPO"
export WARP_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/field01-warp-${SLURM_JOB_ID}"
"$FIELD01_PYTHON" robocub/tools/field01 doctor
"$FIELD01_PYTHON" robocub/tools/benchmark_field01.py --robot g1 --counts 1 16 256 --output "Artifacts/Results/field01/slurm-${SLURM_JOB_ID}"
