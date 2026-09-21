"""Run frozen, bounded G1 experiment queues without an LLM or cloud service.

Linux/WSL execution: python source/scripts/run_g1_queue.py --plan research/queues/...
Each job runs serially, then applies explicit numeric gates. Visual review remains
required. Errors stop the queue; scientific rejection proceeds to the next
independently predeclared job. Never retries or overwrites experiment evidence.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def git_output(root, *args):
    # Match train.py/evaluate_g1.py's semantic Windows/WSL provenance checks.
    return subprocess.check_output(['git', '--no-optional-locks', '-c', 'core.autocrlf=true', *args],
                                   cwd=root, text=True).strip()


def source_is_clean(root):
    # Content comparison avoids status's CRLF-only stat-cache false positives.
    diff = subprocess.run(['git', '--no-optional-locks', '-c', 'core.autocrlf=true', 'diff', '--quiet', 'HEAD'], cwd=root)
    if diff.returncode not in (0, 1):
        raise RuntimeError('Git content comparison failed')
    return diff.returncode == 0 and not git_output(root, 'ls-files', '--others', '--exclude-standard')


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def local_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f'Path must stay inside project: {value}')
    return path


def metrics_health(path, monitor):
    """Ignore only an incomplete final JSONL record; all complete records count."""
    if not path.exists():
        return None, None
    data = path.read_text(encoding='utf-8')
    events = [json.loads(line) for line in data.splitlines(keepends=True)
              if line.endswith('\n') and line.strip()]
    rows = [row for row in events if row.get('event') == 'metrics']
    later_kls = []
    for row in rows:
        for key, value in row['metrics'].items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                return f'nonfinite metric {key} at step {row["step"]}', row['step']
        if row['step'] > monitor['ignore_kl_through_step']:
            if 'training/kl_mean' not in row['metrics']:
                return 'missing monitored training/kl_mean', row['step']
            later_kls.append(row['metrics']['training/kl_mean'])
    count = monitor['consecutive_kl_limit']
    if any(all(v >= monitor['kl_limit'] for v in later_kls[i:i+count])
           for i in range(len(later_kls) - count + 1)):
        return f'{count} consecutive post-transient KL values >= {monitor["kl_limit"]}', rows[-1]['step']
    return None, rows[-1]['step'] if rows else None


def assess_gate(csv_path, gate):
    """Validate complete matched evidence before evaluating declarative rules."""
    with csv_path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    expected = {(c, cmd, str(seed)) for c in gate['controllers']
                for cmd in gate['commands'] for seed in gate['seeds']}
    actual = [(r['controller'], r['command_name'], r['reset_seed']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Evaluation rows are missing, duplicated, or unexpected')
    for row in rows:
        if int(row['requested_steps']) != gate['horizon']:
            raise ValueError('Wrong evaluation horizon')
        if row['all_finite'] != 'True':
            raise ValueError('Evaluation reports nonfinite state')
        # Metric columns, including ones not used for selection, must be finite.
        for key, value in row.items():
            if key in ('controller', 'command_name') or value in ('True', 'False'):
                continue
            if not math.isfinite(float(value)):
                raise ValueError(f'Nonfinite evaluation column: {key}')
    trained = [r for r in rows if r['controller'] == 'trained']
    outcomes = {}
    for name, rules in gate['outcomes'].items():
        checks = []
        for rule in rules:
            selected = [r for r in trained if all(r[k] == str(v) for k, v in rule.get('where', {}).items())]
            if not selected:
                raise ValueError('Gate rule matched no rows')
            if rule['aggregate'] == 'count':
                value = sum(all(float(r[k]) >= bound for k, bound in rule['at_least'].items())
                            and all(float(r[k]) > bound for k, bound in rule.get('above', {}).items())
                            for r in selected)
            else:
                values = [float(r[rule['metric']]) for r in selected]
                value = {'min': min, 'max': max, 'mean': lambda x: sum(x) / len(x)}[rule['aggregate']](values)
            passed = (('min' not in rule or value >= rule['min'])
                      and ('max' not in rule or value <= rule['max'])
                      and ('above' not in rule or isinstance(rule['above'], dict) or value > rule['above']))
            checks.append({'name': rule['name'], 'value': value, 'passed': passed})
        outcomes[name] = {'passed': all(c['passed'] for c in checks), 'checks': checks}
    return {'outcomes': outcomes, 'visual_review_required': True,
            'trained_episodes': len(trained),
            'mean_steps': sum(float(r['episode_steps']) for r in trained) / len(trained),
            'mean_linear_rmse': sum(float(r['linear_velocity_vector_rmse']) for r in trained) / len(trained),
            'mean_yaw_rmse': sum(float(r['yaw_rate_rmse']) for r in trained) / len(trained)}


def stop_process(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def execute_stage(root, stage, output, state, publish, deadline, poll_seconds=10):
    for value in stage['fresh_paths']:
        if local_path(root, value).exists():
            raise FileExistsError(f'Refusing existing evidence: {value}')
    argv = [sys.executable] + [arg.replace('{root}', str(root)) for arg in stage['argv']]
    state.update(stage=stage['id'], stage_started_at=utc(), last_training_step=None)
    publish()
    started = time.monotonic()
    env = dict(os.environ, JAX_DEFAULT_MATMUL_PRECISION='highest', PYTHONUNBUFFERED='1', GIT_OPTIONAL_LOCKS='0')
    with (output / f'{stage["id"]}.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        state['child_pid'] = process.pid
        try:
            publish()
            while True:
                if (output / 'STOP').exists() or (output.parent / 'STOP').exists():
                    raise RuntimeError('STOP file requested termination')
                if time.monotonic() >= deadline or time.monotonic() - started >= stage['timeout_seconds']:
                    raise TimeoutError('Frozen wall-time budget exceeded')
                monitor = stage.get('monitor')
                if monitor:
                    reason, step = metrics_health(local_path(root, monitor['path']), monitor)
                    state['last_training_step'] = step
                    if reason:
                        raise RuntimeError(reason)
                state['heartbeat_at'] = utc()
                publish()
                code = process.poll()
                if code is not None:
                    if code != 0:
                        raise RuntimeError(f'{stage["id"]} exited {code}; inspect stage log')
                    break
                time.sleep(poll_seconds)
        finally:
            stop_process(process)
            state['child_pid'] = None
    for value in stage['required_paths']:
        if not local_path(root, value).exists():
            raise RuntimeError(f'Command exited without required artifact: {value}')
    if stage.get('monitor') and state['last_training_step'] != stage['monitor']['final_step']:
        raise RuntimeError('Training did not finish exactly the declared step budget')


def validate_plan(plan, root):
    if plan['version'] != 1 or not plan['jobs'] or plan['max_wall_seconds'] <= 0:
        raise ValueError('Invalid queue version or budget')
    paths, ids = [], []
    local_path(root, plan['output_dir'])
    for job in plan['jobs']:
        if not job['hypothesis'] or not job['evidence']:
            raise ValueError('Each job needs a hypothesis and prior diagnostic evidence')
        if ('review_artifact' in job) == ('episodes_csv' in job):
            raise ValueError('Job needs exactly one assessment source')
        if 'review_artifact' in job:
            local_path(root, job['review_artifact'])
        ids.append(job['id'])
        stage_ids = []
        for stage in job['stages']:
            stage_ids.append(stage['id'])
            if stage['timeout_seconds'] <= 0 or not stage['argv']:
                raise ValueError('Each stage needs a positive timeout and argv')
            for path in stage['fresh_paths'] + stage['required_paths']:
                local_path(root, path)
            paths.extend(stage['fresh_paths'])
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError('Duplicate stage ID')
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError('Duplicate job ID or output path')
    for path in paths:
        if local_path(root, path).exists():
            raise FileExistsError(f'Refusing existing evidence: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    validate_plan(plan, ROOT)
    if args.validate_only:
        print('Queue valid; no jobs launched.')
        return
    if sys.platform != 'linux':
        raise SystemExit('Run the queue inside WSL/Linux using the training virtualenv.')
    import fcntl
    lock_path = ROOT / 'results' / 'g1_queue.lock'
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        revision = git_output(ROOT, 'rev-parse', 'HEAD')
        if not source_is_clean(ROOT):
            raise RuntimeError('Commit source/plan before queue launch for reproducible provenance')
        output = local_path(ROOT, plan['output_dir'])
        output.mkdir(parents=True, exist_ok=False)
        (output / 'plan.json').write_bytes(plan_bytes)
        state = {'status': 'running', 'started_at': utc(), 'runner_pid': os.getpid(),
                 'git_commit': revision, 'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
                 'jobs': [], 'note': 'Numeric gates do not establish gait or Gate-4 success.'}
        publish = lambda: write_json(output / 'status.json', state)
        publish()
        deadline = time.monotonic() + plan['max_wall_seconds']
        try:
            for job in plan['jobs']:
                job_output = output / job['id']
                job_output.mkdir()
                current = {'id': job['id'], 'status': 'running'}
                state['jobs'].append(current)
                for stage in job['stages']:
                    if git_output(ROOT, 'rev-parse', 'HEAD') != revision or not source_is_clean(ROOT):
                        raise RuntimeError('Source changed during frozen queue; stopping before next stage')
                    execute_stage(ROOT, stage, job_output, current, publish, deadline)
                if 'review_artifact' in job:
                    artifact = local_path(ROOT, job['review_artifact'])
                    json.loads(artifact.read_text(encoding='utf-8'))
                    current['assessment'] = {'review_artifact': job['review_artifact'], 'numeric_gate_evaluated': False}
                    current['status'] = 'needs_review'
                else:
                    current['assessment'] = assess_gate(local_path(ROOT, job['episodes_csv']), job['gate'])
                    current['status'] = 'needs_visual_review' if current['assessment']['outcomes']['pass']['passed'] else 'numeric_gate_failed'
                publish()
            state['status'] = 'needs_review'
        except BaseException as error:
            state['status'] = 'error'
            state['error'] = f'{type(error).__name__}: {error}'
            if state['jobs']:
                state['jobs'][-1]['status'] = 'error'
            raise
        finally:
            state['ended_at'] = utc()
            publish()


if __name__ == '__main__':
    main()
