"""Run a frozen local recovery batch serially, preserving failures and budgets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def stop_queue(child):
    """Let the queue clean up; forcibly stop its owned stage groups if stuck."""
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGINT)
    try:
        child.wait(timeout=45)
        return
    except subprocess.TimeoutExpired:
        pass
    # The queue launches stages in separate sessions. Kill only live children
    # whose kernel-recorded parent is this queue, never an unverified saved PID.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            pid = int(entry.name)
            if int(fields[1]) == child.pid and os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError):
            continue
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=10)


def validate_manifest(manifest, root=ROOT):
    from robocub.scripts.run_g1_queue import local_path, validate_plan
    output = local_path(root, manifest['output_dir'])
    if output.exists():
        raise ValueError('Existing batch output; reconcile instead of replaying')
    if not manifest['runs'] or len({r['id'] for r in manifest['runs']}) != len(manifest['runs']):
        raise ValueError('Empty or duplicate batch IDs')
    if not 0 < manifest['max_wall_seconds'] <= 63 * 3600:
        raise ValueError('Batch exceeds frozen recovery cap')
    total = 0
    outputs = []
    for run in manifest['runs']:
        path = local_path(root, run['plan'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != run['sha256']:
            raise ValueError('Frozen plan hash mismatch')
        plan = json.loads(path.read_text())
        validate_plan(plan, root)
        if not 0 < plan['max_wall_seconds'] <= 21 * 3600:
            raise ValueError('Seed budget exceeds cap')
        total += plan['max_wall_seconds']
        output = local_path(root, plan['output_dir'])
        if output.exists() or output in outputs:
            raise ValueError('Existing or duplicate output; reconcile instead of replaying')
        outputs.append(output)
    if total > manifest['max_wall_seconds']:
        raise ValueError('Seed budgets exceed aggregate cap')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    if args.validate_only:
        print('Batch valid; no training launched.'); return
    if sys.platform != 'linux':
        raise RuntimeError('Use the existing WSL GPU runtime')
    from robocub.scripts.run_g1_queue import local_path, source_is_clean, write_json
    if not source_is_clean(ROOT):
        raise RuntimeError('Commit frozen source before launch')
    out = local_path(ROOT, manifest['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                 runner_pid=os.getpid(), runs=[], note='Parameter continuation; not exact resume or final validation')
    def publish():
        state['heartbeat_at'] = datetime.now(timezone.utc).isoformat()
        write_json(out/'status.json', state)
    publish()
    child = None
    deadline = time.monotonic() + manifest['max_wall_seconds']
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt('Batch interrupted')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for run in manifest['runs']:
            if (out/'STOP').exists():
                raise RuntimeError('Batch STOP requested')
            plan = json.loads((ROOT/run['plan']).read_text())
            row = dict(id=run['id'], status='running', plan=run['plan'])
            state['runs'].append(row); state['active_run'] = run['id']; publish()
            env = dict(os.environ, PYTHONUNBUFFERED='1', MUJOCO_GL='egl',
                       XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_FLAGS='--xla_gpu_autotune_level=0',
                       JAX_COMPILATION_CACHE_DIR=tempfile.mkdtemp(prefix='g1-local-recovery-'))
            env.pop('JAX_PLATFORMS', None)
            started = time.monotonic()
            with (out/(run['id']+'.log')).open('xb') as log:
                child = subprocess.Popen([sys.executable,'-u','robocub/scripts/run_g1_queue.py','--plan',run['plan']],
                                         cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                         start_new_session=True)
                row['pid'] = child.pid; publish()
                while child.poll() is None:
                    if (out/'STOP').exists() or time.monotonic() >= deadline or time.monotonic()-started >= plan['max_wall_seconds']:
                        raise RuntimeError('Stop requested or frozen wall-time budget exhausted')
                    publish(); time.sleep(10)
                row['exit_code'] = child.returncode
                row['wall_seconds'] = time.monotonic()-started
                row['status'] = 'needs_review' if child.returncode == 0 else 'error'
                publish()
                if child.returncode:
                    raise RuntimeError('Queue failed; preserve completed steps and review '+run['id'])
        state['status'] = 'needs_review'
    except BaseException as exc:
        state['status'] = 'error'; state['error'] = str(exc)
        raise
    finally:
        if child is not None and child.poll() is None:
            stop_queue(child)
        state['ended_at'] = datetime.now(timezone.utc).isoformat(); publish()


if __name__ == '__main__':
    main()
