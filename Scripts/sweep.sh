#!/usr/bin/env bash
"true" ':' <<'DOC'
Send the current RoboCup branch to Atlantis and start a Slurm sweep.

Run this file from Git Bash. It validates the two JSON sweep inputs locally,
commits and pushes the selected branch, prepares a clean remote checkout, and
submits GPU work through Slurm. Extra arguments are forwarded unchanged to
App/Scripts/sweep.py, so `--smoke --no-video` creates a small verification run.
DOC

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
TRAINING_JSON="$REPO_ROOT/App/Configs/sweeps/training_hyperparameters.json"
ENVIRONMENT_JSON="$REPO_ROOT/App/Configs/sweeps/environment_hyperparameters.json"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

[ -f "$ENV_FILE" ] || die "Missing $ENV_FILE. Copy the documented Atlantis values into it first."
set -a
# Remove Windows carriage returns before Bash reads the local configuration.
source <(tr -d '\r' < "$ENV_FILE")
set +a

required_values=(BRANCH_NAME SSH_KEY_PATH GITHUB_KEY_PATH SSH_PORT ATLANTIS_HOST SSH_DIRECTORY GITHUB_REPO_SSH SLURM_ACCOUNT SLURM_PARTITION SLURM_TIME SLURM_GRES_TYPE SLURM_CPUS_PER_TASK SLURM_MEMORY)
for value_name in "${required_values[@]}"; do
  [ -n "${!value_name:-}" ] || die "$value_name is missing from .env"
done
[ -f "$SSH_KEY_PATH" ] || die "Atlantis SSH key not found: $SSH_KEY_PATH"
[ -f "$GITHUB_KEY_PATH" ] || die "GitHub SSH key not found: $GITHUB_KEY_PATH"

cd "$REPO_ROOT"
[ "$(git branch --show-current)" = "$BRANCH_NAME" ] || die "Current branch is not $BRANCH_NAME"
git check-ignore -q .env || die ".env must be ignored by Git before deployment"

LOCAL_PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"
[ -x "$LOCAL_PYTHON" ] || LOCAL_PYTHON="$(command -v python || true)"
[ -n "$LOCAL_PYTHON" ] || die "Python is required to validate sweep configuration"

# Validate every user-edited training input before changing Git or connecting.
"$LOCAL_PYTHON" - "$TRAINING_JSON" "$ENVIRONMENT_JSON" <<'PY'
import json
import sys
from pathlib import Path

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise SystemExit(f"{path} must contain a non-empty JSON list")
    if not all(isinstance(row, dict) for row in value):
        raise SystemExit(f"Every row in {path} must be a JSON object")
    print(f"Validated {path}: {len(value)} row(s)")
PY
"$LOCAL_PYTHON" "$REPO_ROOT/App/Scripts/sweep.py" --help >/dev/null

echo "Committing and pushing $BRANCH_NAME ..."
git add -A
git commit -m "Deploy RoboCup sweep to Atlantis" || echo "No local changes to commit."
git push -u origin "$BRANCH_NAME"

# Agent forwarding lets Atlantis authenticate to GitHub without copying a key.
if ! ssh-add -l >/dev/null 2>&1; then
  eval "$(ssh-agent -s)" >/dev/null
fi
ssh-add "$GITHUB_KEY_PATH" >/dev/null

SSH_ARGS=(-A -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=20 -i "$SSH_KEY_PATH" -p "$SSH_PORT")
MAX_ALLOCATION_RETRIES="${MAX_ALLOCATION_RETRIES:-4}"

echo "Connecting to $ATLANTIS_HOST ..."
ssh "${SSH_ARGS[@]}" "$ATLANTIS_HOST" bash -s -- \
  "$BRANCH_NAME" "$SSH_DIRECTORY" "$GITHUB_REPO_SSH" \
  "$SLURM_ACCOUNT" "$SLURM_PARTITION" "$SLURM_TIME" "$SLURM_GRES_TYPE" \
  "$SLURM_CPUS_PER_TASK" "$SLURM_MEMORY" "$MAX_ALLOCATION_RETRIES" "$@" <<'REMOTE'
set -euo pipefail

BRANCH_NAME="$1"; shift
SSH_DIRECTORY="$1"; shift
GITHUB_REPO_SSH="$1"; shift
SLURM_ACCOUNT="$1"; shift
SLURM_PARTITION="$1"; shift
SLURM_TIME="$1"; shift
SLURM_GRES_TYPE="$1"; shift
SLURM_CPUS_PER_TASK="$1"; shift
SLURM_MEMORY="$1"; shift
MAX_ALLOCATION_RETRIES="$1"; shift
SWEEP_ARGS=("$@")
SLURM_GRES="${SLURM_GRES_TYPE}:1"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable on the Atlantis login node." >&2; exit 2; }

# Test the account and requested hardware without printing Slurm's fake job ID.
if ! PREFLIGHT_ERROR="$(sbatch --test-only --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --time="$SLURM_TIME" --gres="$SLURM_GRES" --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="$SLURM_MEMORY" --wrap='true' 2>&1)"; then
  printf '%s\n' "$PREFLIGHT_ERROR" >&2
  echo "ERROR: Atlantis rejected the requested Slurm account or resources." >&2
  exit 2
fi
echo "Slurm account/resource preflight passed."

# Trust only GitHub's officially published ED25519 host key.
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
touch "$HOME/.ssh/known_hosts"
chmod 600 "$HOME/.ssh/known_hosts"
GITHUB_HOST_KEY='github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl'
grep -Fqx "$GITHUB_HOST_KEY" "$HOME/.ssh/known_hosts" || printf '%s\n' "$GITHUB_HOST_KEY" >> "$HOME/.ssh/known_hosts"

# Clone once, then force the reusable checkout to the pushed branch exactly.
if [ ! -d "$SSH_DIRECTORY/.git" ]; then
  mkdir -p "$(dirname "$SSH_DIRECTORY")"
  git clone "$GITHUB_REPO_SSH" "$SSH_DIRECTORY"
fi
cd "$SSH_DIRECTORY"
git fetch origin "$BRANCH_NAME"
git checkout -B "$BRANCH_NAME" "origin/$BRANCH_NAME"
git reset --hard "origin/$BRANCH_NAME"
git clean -fd

REMOTE_OUTPUT_ROOT="$SSH_DIRECTORY/Scripts/Outputs"
mkdir -p "$REMOTE_OUTPUT_ROOT"
NEXT_NUMBER=0
while [ -e "$REMOTE_OUTPUT_ROOT/Sweep_${NEXT_NUMBER}" ]; do NEXT_NUMBER=$((NEXT_NUMBER + 1)); done
SWEEP_LABEL="Sweep_${NEXT_NUMBER}"
SWEEP_DIR="$REMOTE_OUTPUT_ROOT/$SWEEP_LABEL"
mkdir -p "$SWEEP_DIR"

STATUS_FILE="$SWEEP_DIR/.remote_status"
EXIT_FILE="$SWEEP_DIR/.remote_exit_code"
PID_FILE="$SWEEP_DIR/.remote_pid"
JOB_INFO="$SWEEP_DIR/job_info.txt"
RUNNER_LOG="$SWEEP_DIR/remote_runner.log"
RUNNER="$SWEEP_DIR/run_remote_sweep.sh"
printf 'launching\n' > "$STATUS_FILE"
printf '%s\n' \
  "sweep_label=$SWEEP_LABEL" \
  "task_name=Unitree G1 PPO hyperparameter sweep" \
  "policy_name=ppo_g1" \
  "branch_name=$BRANCH_NAME" \
  "training_hyperparameters=App/Configs/sweeps/training_hyperparameters.json" \
  "environment_hyperparameters=App/Configs/sweeps/environment_hyperparameters.json" \
  "submitted_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  "slurm_account=$SLURM_ACCOUNT" \
  "slurm_partition=$SLURM_PARTITION" \
  "slurm_gres=$SLURM_GRES" > "$JOB_INFO"

cat > "$RUNNER" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$1"; shift
SWEEP_DIR="$1"; shift
STATUS_FILE="$1"; shift
EXIT_FILE="$1"; shift
SWEEP_ARGS=("$@")

finish() {
  exit_code="$?"
  printf '%s\n' "$exit_code" > "$EXIT_FILE"
  if [ "$exit_code" -eq 0 ]; then printf 'succeeded\n' > "$STATUS_FILE"; else printf 'failed\n' > "$STATUS_FILE"; fi
}
trap finish EXIT
printf 'starting\n' > "$STATUS_FILE"

echo "Compute node: $(hostname)"
echo "Slurm job ID: ${SLURM_JOB_ID:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# A newly allocated Atlantis GPU can briefly fail health checks. Retry before
# declaring this allocation unhealthy so the login-side submitter can replace it.
GPU_OK=false
for attempt in 1 2 3 4 5 6; do
  if nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader; then GPU_OK=true; break; fi
  echo "GPU_HEALTH_TRANSIENT: nvidia-smi failed ($attempt/6); retrying in 10 seconds." >&2
  sleep 10
done
if [ "$GPU_OK" != true ]; then
  echo "UNHEALTHY_GPU_ALLOCATION: nvidia-smi never succeeded; request a fresh allocation." >&2
  exit 75
fi

cd "$REPO_DIR"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv under $HOME/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
fi

# Keep one reusable Linux GPU environment outside generated sweep folders.
if [ ! -x ".venv-atlantis/bin/python" ]; then uv venv --python 3.12 .venv-atlantis; fi
uv pip install --python .venv-atlantis/bin/python -r requirements.txt
uv pip install --python .venv-atlantis/bin/python --upgrade 'jax[cuda12]==0.11.0'
PYTHON="$REPO_DIR/.venv-atlantis/bin/python"

# Imports, CUDA discovery, and environment construction must all pass before
# the status changes to running or the expensive sweep starts.
"$PYTHON" - <<'PY'
import jax
from omegaconf import OmegaConf
from robocub.config.g1 import G1MJXConfig
from robocub.locomotion.unitree_g1.joystick import Joystick
from robocub.robots.unitree_g1 import UnitreeG1Model

devices = jax.devices("gpu")
if not devices:
    raise SystemExit("CUDA_PREFLIGHT_FAILED: JAX detected no GPU")
print(f"CUDA preflight passed: {devices}")
config = OmegaConf.structured(G1MJXConfig())
config.noise.add_noise = False
config.push.add_push = False
config.domain_rand.add_domain_rand = False
Joystick("unitree_g1", UnitreeG1Model(fetch=True), "flat", config)
print("Policy environment initialized: Unitree G1 MJX joystick")
PY

printf 'running\n' > "$STATUS_FILE"
echo "Training preflights passed. Starting first RoboCup sweep trial."
"$PYTHON" App/Scripts/sweep.py \
  --output-dir "$SWEEP_DIR/training" \
  "${SWEEP_ARGS[@]}"
RUNNER
chmod +x "$RUNNER"

submit_job() {
  rm -f "$EXIT_FILE" "$RUNNER_LOG"
  printf 'launching\n' > "$STATUS_FILE"
  sbatch --parsable \
    --job-name="robocup-$SWEEP_LABEL" \
    --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --time="$SLURM_TIME" --gres="$SLURM_GRES" \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="$SLURM_MEMORY" \
    --output="$RUNNER_LOG" \
    "$RUNNER" "$SSH_DIRECTORY" "$SWEEP_DIR" "$STATUS_FILE" "$EXIT_FILE" "${SWEEP_ARGS[@]}"
}

STARTUP_VERIFIED=false
for retry in $(seq 1 "$MAX_ALLOCATION_RETRIES"); do
  JOB_ID="$(submit_job)"
  printf '%s\n' "$JOB_ID" > "$PID_FILE"
  echo "Submitted $SWEEP_LABEL as Slurm job $JOB_ID (allocation $retry/$MAX_ALLOCATION_RETRIES)."

  for _ in $(seq 1 24); do
    status="$(cat "$STATUS_FILE" 2>/dev/null || printf launching)"
    if [ "$status" = running ] || [ "$status" = succeeded ]; then STARTUP_VERIFIED=true; break 2; fi
    if [ "$status" = failed ]; then
      if grep -Fq 'UNHEALTHY_GPU_ALLOCATION' "$RUNNER_LOG" && [ "$retry" -lt "$MAX_ALLOCATION_RETRIES" ]; then
        echo "Allocation $JOB_ID had an unhealthy GPU; requesting a fresh allocation." >&2
        break
      fi
      echo "ERROR: RoboCup sweep failed during startup." >&2
      tail -n 120 "$RUNNER_LOG" >&2 || true
      exit 2
    fi
    sleep 5
  done

  status="$(cat "$STATUS_FILE" 2>/dev/null || printf launching)"
  if [ "$status" != failed ]; then break; fi
done

if [ "$STARTUP_VERIFIED" = true ]; then
  echo "Training startup verified successfully"
else
  PENDING_REASON="$(squeue -h -j "$JOB_ID" -o '%R' 2>/dev/null | head -n 1 || true)"
  case "$PENDING_REASON" in
    Resources|AssocGrpCpuLimit)
      echo "Sweep is queued because of $PENDING_REASON and will start automatically."
      ;;
    *)
      echo "Sweep remains queued or in startup (reason: ${PENDING_REASON:-unknown}) and will continue automatically."
      ;;
  esac
fi
echo "Remote sweep: $SWEEP_DIR"
echo "Remote log: $RUNNER_LOG"
echo "Slurm job ID: $JOB_ID"
REMOTE

echo "Deployment finished. Run Scripts/return.sh to inspect or download snapshots."
