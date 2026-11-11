"""Isolated, bounded Atlantis research jobs. No GitHub push or shared reset.

Submit a committed queue; status fetches only compact runner/Slurm state.
Credentials are consumed by SSH from ignored local configuration, never copied.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile
import io

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from robocub.user_cli.cluster_dashboard import parse_env, windows_path

# uv-created environments need not contain pip. Query installed metadata directly.
PACKAGE_VERSIONS_CODE = "import importlib.metadata as m; print('\\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in m.distributions() if d.metadata['Name'])))"

RUNTIME_PROFILES = {'legacy': None, 'reproducible': '--xla_gpu_autotune_level=0'}


def runtime_setup(profile):
    """Set a recorded job-local profile before any JAX import; reject conflicts."""
    flags = RUNTIME_PROFILES[profile]
    if flags is None:
        return ''
    quoted = shlex.quote(flags)
    return (f'if [[ -n "${{XLA_FLAGS:-}}" && "${{XLA_FLAGS}}" != {quoted} ]]; then\n'
            '  echo "Conflicting inherited XLA_FLAGS" >&2; exit 2\nfi\n'
            f'export XLA_FLAGS={quoted}\n')


def ssh(settings, command, payload=None, timeout=60):
    args = ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
            '-o', 'ConnectTimeout=20', '-i', windows_path(settings['SSH_KEY_PATH']),
            '-p', settings['SSH_PORT'], settings['ATLANTIS_HOST'], command]
    result = subprocess.run(args, input=payload, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace') + result.stdout.decode(errors='replace'))
    return result.stdout.decode(errors='replace')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def validate_archive(data):
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for item in archive.getmembers():
            p = PurePosixPath(item.name)
            if p.is_absolute() or '..' in p.parts or '\\' in item.name or ':' in item.name or item.issym() or item.islnk():
                raise ValueError('Unsafe archive member')
            if any(x in p.parts for x in ['.env', '.ssh', '.aws', '.venv', '.cache']):
                raise ValueError('Private/local material must not be uploaded')


def compact_status(raw):
    text = raw.lstrip()
    try:
        data, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return {'status': 'unknown', 'scheduler': text[-1600:]}
    keys = ('id', 'status', 'stage', 'heartbeat_at', 'last_training_step')
    return {'status': data.get('status', 'unknown'), 'error': data.get('error'),
            'jobs': [{k: j[k] for k in keys if k in j} for j in data.get('jobs', [])],
            'scheduler': text[end:].strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['submit', 'status'])
    parser.add_argument('--run', required=True)
    parser.add_argument('--plan')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('--memory-gb', type=int, default=32)
    parser.add_argument('--runtime-profile', choices=RUNTIME_PROFILES, default='legacy')
    args = parser.parse_args()
    if not 1 <= args.cpus <= 8 or not 8 <= args.memory_gb <= 32:
        parser.error('Resources must be 1-8 CPUs and 8-32 GB')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.run):
        raise ValueError('Invalid run identifier')
    local = ROOT/'Artifacts/Results/cluster_research'/args.run
    manifest = local/'submission.json'
    settings = parse_env()
    if args.action == 'status':
        m = json.loads(manifest.read_text())
        remote = shlex.quote(m['remote_dir'])
        job = m.get('job_id', '')
        if job and not job.isdigit(): raise ValueError('Invalid Slurm ID')
        cmd = f"cd {remote}; cat source/{shlex.quote(m['output_dir'])}/status.json 2>/dev/null || true; "
        cmd += "printf '\nSLURM\n'; "
        if job:
            cmd += f"squeue -h -j {job} -o '%i %T %M %R'; sacct -n -X -j {job} --format=JobID,State,Elapsed,ExitCode; "
        cmd += "printf '\nLAUNCH_EXIT\n'; cat exit_code 2>/dev/null || true"
        raw = ssh(settings, cmd)
        print(json.dumps(compact_status(raw)) if args.compact else raw); return
    if manifest.exists():
        raise FileExistsError('Submission already attempted; reconcile saved remote directory/job before retrying')
    plan_path = (ROOT/args.plan).resolve()
    if not plan_path.is_relative_to(ROOT): raise ValueError('Plan must be inside repository')
    plan = json.loads(plan_path.read_text())
    subprocess.run([sys.executable, str(ROOT/'robocub/scripts/run_g1_queue.py'),
                    '--plan', str(plan_path), '--validate-only'], check=True, cwd=ROOT)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip():
        raise RuntimeError('Commit reviewed changes before snapshotting')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    archive = subprocess.check_output(['git','archive','--format=tar','HEAD'],cwd=ROOT)
    validate_archive(archive)
    remote = settings['SSH_DIRECTORY'].rstrip('/')+'/research_runs/'+args.run
    meta = dict(run=args.run, remote_dir=remote, local_commit=revision,
                archive_sha256=hashlib.sha256(archive).hexdigest(), output_dir=plan['output_dir'],
                plan=args.plan, status='submission_preparing', job_id=None,
                runtime_profile=args.runtime_profile, xla_flags=RUNTIME_PROFILES[args.runtime_profile],
                resources={'gpus':1,'cpus':args.cpus,'memory':str(args.memory_gb)+'G','partition':'long','time':'08:00:00'})
    save(manifest,meta)
    q = shlex.quote
    # mkdir is exclusive: an ambiguous previous submission can never be duplicated.
    ssh(settings, f'mkdir -p {q(settings["SSH_DIRECTORY"]+"/research_runs")}; mkdir {q(remote)} && mkdir {q(remote+"/source")} && tar -xf - -C {q(remote+"/source")}', archive, timeout=180)
    shared = settings['SSH_DIRECTORY']
    setup = f'''set -eu
cd {q(remote+'/source')}
git init -q
git add -A
git -c user.name='Research snapshot' -c user.email='snapshot@localhost' commit -qm {q('Frozen local source '+revision)}
mkdir -p .cache
ln -s {q(shared+'/.cache/mujoco_menagerie')} .cache/mujoco_menagerie
'''
    ssh(settings, 'bash -s', setup.encode())
    runner = f'''#!/usr/bin/env bash
set -euo pipefail
cd {q(remote+'/source')}
trap 'printf "%s\\n" "$?" > {q(remote+'/exit_code')}' EXIT
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD"
export JAX_COMPILATION_CACHE_DIR="$(mktemp -d /tmp/g1-jax-cache.XXXXXXXXXX)"
{runtime_setup(args.runtime_profile)}
nvidia-smi --query-gpu=uuid,pci.bus_id,name,memory.total,driver_version --format=csv
{q(shared+'/.venv-atlantis/bin/python')} -c "import jax; assert len(jax.devices('gpu')) == 1; print(jax.devices('gpu'))"
{q(shared+'/.venv-atlantis/bin/python')} -c {q(PACKAGE_VERSIONS_CODE)} > {q(remote+'/runtime_versions.txt')}
exec_queue() {{ {q(shared+'/.venv-atlantis/bin/python')} -u robocub/scripts/run_g1_queue.py --plan {q(args.plan)}; }}
exec_queue
'''
    ssh(settings, f'cat > {q(remote+"/run.sh")}', runner.encode())
    sbatch = ['sbatch','--parsable','--account='+settings['SLURM_ACCOUNT'],
              '--partition=long','--time=08:00:00','--gres=rtx2080ti:1',
              '--cpus-per-task='+str(args.cpus),'--mem='+str(args.memory_gb)+'G','--job-name=g1-'+args.run,
              '--output='+remote+'/launcher.log',remote+'/run.sh']
    # Scheduler validation allocates nothing; only the next command submits.
    ssh(settings, ' '.join(q(x) for x in sbatch[:1]+['--test-only']+sbatch[2:]))
    answer = ssh(settings, ' '.join(q(x) for x in sbatch)).strip()
    job = answer.split(';')[0]
    if not job.isdigit(): raise RuntimeError('Ambiguous scheduler response; inspect remote state, do not retry')
    meta.update(job_id=job,status='submitted'); save(manifest,meta)
    print(json.dumps({'run':args.run,'job_id':job,'status':'submitted','resources':meta['resources']}))


if __name__ == '__main__': main()
