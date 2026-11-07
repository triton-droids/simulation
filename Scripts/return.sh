#!/usr/bin/env bash
"true" ':' <<'DOC'
List Atlantis RoboCup sweeps and download a safe point-in-time snapshot.

Pass an optional label, for example `Scripts/return.sh Sweep_0`. Active jobs
are never deleted. Finished jobs are deleted only when the ignored `.env`
explicitly sets DELETE_REMOTE_AFTER_DOWNLOAD=true.
DOC

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
[ -f "$ENV_FILE" ] || { echo "ERROR: Missing $ENV_FILE" >&2; exit 2; }
set -a
source <(tr -d '\r' < "$ENV_FILE")
set +a

SSH_ARGS=(-o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=20 -i "$SSH_KEY_PATH" -p "$SSH_PORT")
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-$SSH_DIRECTORY/Scripts/Outputs}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$REPO_ROOT/Scripts/Outputs}"
DELETE_REMOTE_AFTER_DOWNLOAD="${DELETE_REMOTE_AFTER_DOWNLOAD:-false}"
mkdir -p "$LOCAL_OUTPUT_DIR"

LISTING_FILE="$(mktemp)"
trap 'rm -f "$LISTING_FILE"' EXIT

# ASCII unit separator preserves empty metadata fields when Bash reads rows.
ssh "${SSH_ARGS[@]}" "$ATLANTIS_HOST" bash -s -- "$REMOTE_OUTPUT_DIR" > "$LISTING_FILE" <<'REMOTE'
set -euo pipefail
root="$1"
[ -d "$root" ] || exit 0
find "$root" -mindepth 1 -maxdepth 1 -type d -name 'Sweep_[0-9]*' -print0 | sort -zV | while IFS= read -r -d '' dir; do
  label="$(basename "$dir")"
  status="$(cat "$dir/.remote_status" 2>/dev/null | tr -d '\r\n' || printf unknown)"
  exit_code="$(cat "$dir/.remote_exit_code" 2>/dev/null | tr -d '\r\n' || true)"
  job_id="$(cat "$dir/.remote_pid" 2>/dev/null | tr -d '\r\n' || true)"

  # A queued job can be cancelled before its runner starts, leaving the last
  # file status at launching. Reconcile that stale state with Slurm accounting.
  case "$status" in
    launching|starting|running)
      if [[ "$job_id" =~ ^[0-9]+$ ]] && ! squeue -h -j "$job_id" | grep -q .; then
        slurm_state="$(sacct -n -X -j "$job_id" --format=State 2>/dev/null | awk 'NF {print $1; exit}' | cut -d+ -f1)"
        case "$slurm_state" in
          COMPLETED)
            status="succeeded"
            exit_code="0"
            ;;
          CANCELLED|FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
            status="failed"
            exit_code="1"
            ;;
        esac
        if [ "$status" = succeeded ] || [ "$status" = failed ]; then
          printf '%s\n' "$status" > "$dir/.remote_status"
          printf '%s\n' "$exit_code" > "$dir/.remote_exit_code"
        fi
      fi
      ;;
  esac
  task="$(sed -n 's/^task_name=//p' "$dir/job_info.txt" 2>/dev/null | head -n 1 || true)"
  policy="$(sed -n 's/^policy_name=//p' "$dir/job_info.txt" 2>/dev/null | head -n 1 || true)"
  submitted="$(sed -n 's/^submitted_at_utc=//p' "$dir/job_info.txt" 2>/dev/null | head -n 1 || true)"
  completed=0
  if [ -f "$dir/training/results.json" ]; then
    completed="$(python3 - "$dir/training/results.json" <<'PY'
import json, sys
try:
    print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
except Exception:
    print(0)
PY
)"
  fi
  printf '%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\n' "$label" "$status" "$task" "$policy" "$submitted" "$completed" "$exit_code"
done
REMOTE

show_group() {
  heading="$1"; mode="$2"; found=false
  echo "$heading"
  while IFS=$'\x1f' read -r label status task policy submitted completed exit_code; do
    [ -n "${label:-}" ] || continue
    case "$mode:$status" in
      running:launching|running:starting|running:running) ;;
      finished:succeeded|finished:failed) ;;
      other:launching|other:starting|other:running|other:succeeded|other:failed) continue ;;
      running:*|finished:*) continue ;;
    esac
    found=true
    printf '  %s | status=%s | task=%s | policy=%s | submitted=%s | completed_trials=%s | exit_code=%s\n' \
      "$label" "${status:-unknown}" "${task:-unknown}" "${policy:-unknown}" "${submitted:-unknown}" "${completed:-0}" "${exit_code:-unknown}"
  done < "$LISTING_FILE"
  $found || echo "  (none)"
  echo
}

echo "Atlantis RoboCup sweeps under $REMOTE_OUTPUT_DIR"
show_group "Running:" running
show_group "Finished:" finished
show_group "Other:" other

SELECTED_LABEL="${1:-}"
if [ -z "$SELECTED_LABEL" ]; then
  if [ ! -t 0 ]; then exit 0; fi
  read -r -p "Sweep to download (for example Sweep_0; Enter cancels): " SELECTED_LABEL
fi
[ -n "$SELECTED_LABEL" ] || exit 0
[[ "$SELECTED_LABEL" =~ ^Sweep_[0-9]+$ ]] || { echo "ERROR: invalid sweep label: $SELECTED_LABEL" >&2; exit 2; }

SELECTED_ROW="$(grep -a -m1 "^${SELECTED_LABEL}"$'\x1f' "$LISTING_FILE" || true)"
[ -n "$SELECTED_ROW" ] || { echo "ERROR: $SELECTED_LABEL was not found remotely" >&2; exit 2; }
IFS=$'\x1f' read -r _ SELECTED_STATUS _ _ _ _ _ <<< "$SELECTED_ROW"

REMOTE_SWEEP="$REMOTE_OUTPUT_DIR/$SELECTED_LABEL"
LOCAL_SWEEP="$LOCAL_OUTPUT_DIR/$SELECTED_LABEL"
mkdir -p "$LOCAL_SWEEP"

# Tar streams one coherent snapshot and includes dotfiles such as status/PID.
echo "Downloading snapshot for $SELECTED_LABEL (status=$SELECTED_STATUS) ..."
ssh "${SSH_ARGS[@]}" "$ATLANTIS_HOST" tar -C "$REMOTE_OUTPUT_DIR" -cf - "$SELECTED_LABEL" | tar -C "$LOCAL_OUTPUT_DIR" -xf -
date -u +"Last snapshot: %Y-%m-%dT%H:%M:%SZ" > "$LOCAL_SWEEP/.last_sync.txt"
echo "Snapshot saved to $LOCAL_SWEEP"

case "$SELECTED_STATUS" in
  launching|starting|running)
    echo "Remote sweep kept because the Slurm job is active."
    ;;
  succeeded|failed)
    if [ "$DELETE_REMOTE_AFTER_DOWNLOAD" = true ]; then
      ssh "${SSH_ARGS[@]}" "$ATLANTIS_HOST" bash -s -- "$REMOTE_OUTPUT_DIR" "$SELECTED_LABEL" <<'REMOTE'
set -euo pipefail
root="$1"; label="$2"; target="$root/$label"
[[ "$label" =~ ^Sweep_[0-9]+$ ]] || { echo "Refusing unsafe label" >&2; exit 2; }
case "$target" in "$root"/Sweep_[0-9]*) ;; *) echo "Refusing unsafe path" >&2; exit 2;; esac
[ -d "$target" ] || { echo "Remote sweep no longer exists" >&2; exit 2; }
status="$(cat "$target/.remote_status" 2>/dev/null | tr -d '\r\n' || true)"
case "$status" in succeeded|failed) rm -rf -- "$target" ;; *) echo "Refusing to delete active sweep" >&2; exit 2;; esac
REMOTE
      echo "Completed remote sweep deleted because DELETE_REMOTE_AFTER_DOWNLOAD=true."
    else
      echo "Remote sweep kept because DELETE_REMOTE_AFTER_DOWNLOAD=$DELETE_REMOTE_AFTER_DOWNLOAD."
    fi
    ;;
  *) echo "Remote sweep kept because its status is not finalized." ;;
esac
