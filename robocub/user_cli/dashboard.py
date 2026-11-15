"""Unified compute and frozen-queue dashboard; no simulator imports at startup."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shlex
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / 'App/Configs/dashboard_plans.json'


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def contained(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError('Path must stay inside this project')
    return path


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def display_state(data, now=None):
    """Never present an old queue heartbeat as proof that work is running."""
    state = data.get('status', 'unknown')
    if state != 'running':
        return state.replace('_', ' ')
    jobs = data.get('jobs', [])
    stamp = next((j.get('heartbeat_at') for j in reversed(jobs) if j.get('heartbeat_at')), data.get('started_at'))
    try:
        age = ((now or datetime.now(timezone.utc)) - datetime.fromisoformat(stamp.replace('Z', '+00:00'))).total_seconds()
    except (ValueError, TypeError, AttributeError):
        return 'unknown — no heartbeat'
    return 'running' if 0 <= age < 300 else 'stale — verify process'


@dataclass
class Run:
    name: str
    source: str
    state: str
    folder: Path
    status_path: Path | None = None
    receipt: Path | None = None
    stage: str = ''
    step: str = ''
    updated: str = ''


class Service:
    def __init__(self, root=ROOT):
        self.root = Path(root)

    def runs(self):
        results = self.root / 'Artifacts/Results'
        found = []
        for path in results.glob('*/queue/status.json'):
            data = read_json(path)
            # Cluster copies are represented by their submission receipt below.
            if path.parent.parent.name.startswith('cluster_'):
                continue
            jobs = data.get('jobs', [])
            current = jobs[-1] if jobs else {}
            checkpoints = path.parent.parent / 'train/logs/checkpoints'
            saved = max((int(p.name) for p in checkpoints.iterdir() if p.name.isdigit()), default=0) if checkpoints.is_dir() else 0
            found.append(Run(path.parent.parent.name, 'local', display_state(data), path.parent.parent,
                             path, stage=current.get('stage', ''),
                             step=str(current.get('last_training_step') or saved or ''),
                             updated=data.get('ended_at') or current.get('heartbeat_at') or data.get('started_at', '')))
        for path in (results / 'cluster_research').glob('*/submission.json'):
            data = read_json(path)
            if not data.get('run'):
                continue
            found.append(Run(data['run'], 'atlantis', 'unknown — check Slurm', path.parent,
                             receipt=path, stage='Job ' + str(data.get('job_id') or 'not confirmed'),
                             updated=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()))
        return sorted(found, key=lambda r: r.updated, reverse=True)

    def approved(self, source):
        entries = read_json(self.root / 'App/Configs/dashboard_plans.json').get('plans', [])
        return [e for e in entries if e.get('enabled') is True and e.get('source') == source]

    def validate_entry(self, entry):
        """Catalog approval binds exact bytes, provider and a never-used output."""
        if entry not in self.approved(entry.get('source')):
            raise ValueError('Plan is not enabled in the reviewed catalog')
        path = contained(self.root, entry['plan'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Frozen plan changed; review and register its new hash first')
        data = read_json(path)
        out = contained(self.root, data['output_dir'])
        if out.exists():
            raise ValueError('This output already exists; never repeat or overwrite a run')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', entry['id']):
            raise ValueError('Invalid plan identifier')
        if (self.root / 'Artifacts/Results/dashboard_launches' / entry['id']).exists():
            raise ValueError('Launch already attempted; reconcile its receipt before recovery')
        return path, data

    def check(self, source):
        if source not in ('local', 'atlantis'):
            raise ValueError('Unsupported compute provider; register a backend adapter first')
        if source == 'local':
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=15, check=True).stdout.strip()
            if os.name == 'nt':
                code = "import jax; assert jax.devices('gpu'); print(jax.devices('gpu'))"
                linux_root = '/mnt/' + self.root.drive[0].lower() + self.root.as_posix()[2:]
                cmd = ['wsl', '-d', 'Ubuntu', '--', 'bash', '-lc',
                       'cd ' + shlex.quote(linux_root) + ' && .cache/g1_wsl_venv/bin/python -c ' + shlex.quote(code)]
            else:
                cmd = [str(self.root / '.cache/g1_wsl_venv/bin/python'), '-c', "import jax; print(jax.devices('gpu'))"]
            subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=True)
            busy = any(r.source == 'local' and r.state == 'running' for r in self.runs())
            return ('Busy' if busy else 'Runtime verified') + ' — ' + gpu + '\nTraining capacity depends on the selected plan. Checked ' + utc()
        from robocub.tools.cluster_research import ssh
        from robocub.user_cli.cluster_dashboard import parse_env
        raw = ssh(parse_env(), 'scontrol show node zixian; scontrol show node matthew; sinfo -R')
        states = re.findall(r'State=(\S+)', raw)
        blocked = not states or all(any(x in s for x in ('DOWN', 'DRAIN', 'NOT_RESPONDING')) for s in states)
        return ('Unavailable' if blocked else 'Scheduler available — GPU runtime unverified') + '\nChecked ' + utc() + '\n' + raw

    def remote_status(self, run):
        from robocub.tools.cluster_research import ssh
        from robocub.user_cli.cluster_dashboard import parse_env
        job = str(read_json(run.receipt).get('job_id', ''))
        if not job.isdigit():
            raise ValueError('No confirmed job ID; reconcile submission before launching')
        raw = ssh(parse_env(), f"squeue -h -j {job} -o '%i %T %M %R'; sacct -n -X -j {job} --format=JobID,State,Elapsed,ExitCode")
        return raw.strip() or 'Unknown — scheduler returned no records'

    def stop(self, run):
        if run.source == 'local':
            if display_state(read_json(run.status_path)) != 'running':
                raise ValueError('No fresh running heartbeat; inspect the process before stopping')
            # Queue reads STOP in the per-job output or its parent queue folder.
            (run.status_path.parent / 'STOP').touch(exist_ok=False)
            return 'Stop requested. The current stage will terminate; only previously saved checkpoints are retained.'
        from robocub.tools.cluster_research import ssh
        from robocub.user_cli.cluster_dashboard import parse_env
        job = str(read_json(run.receipt).get('job_id', ''))
        if not job.isdigit():
            raise ValueError('No confirmed scheduler job')
        ssh(parse_env(), f'scancel {job}')
        return 'Cancellation requested for job ' + job + '; check Slurm to confirm.'

    def launch(self, entry):
        self.validate_entry(entry)
        # Local launching is delegated to the Linux path below, not Windows CPU JAX.
        args = ['-m', 'robocub.user_cli.dashboard', '--launch-local', entry['id']]
        if entry['source'] == 'local':
            linux_root = '/mnt/' + self.root.drive[0].lower() + self.root.as_posix()[2:] if os.name == 'nt' else str(self.root)
            command = 'cd ' + shlex.quote(linux_root) + ' && .cache/g1_wsl_venv/bin/python ' + shlex.join(args)
            cmd = ['wsl', '-d', 'Ubuntu', '--', 'bash', '-lc', command] if os.name == 'nt' else ['bash', '-lc', command]
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout
        # The existing snapshot launcher enforces clean commit, schema and duplicate receipts.
        resources = entry['resources']
        cmd = [sys.executable, str(self.root / 'robocub/tools/cluster_research.py'), 'submit', '--run', entry['id'],
               '--plan', entry['plan'], '--runtime-profile', 'reproducible', '--gpu-type', resources['gpu_type'],
               '--cpus', str(resources['cpus']), '--memory-gb', str(resources['memory_gb']), '--slurm-hours', str(resources['hours'])]
        return subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=360, check=True).stdout


def launch_local(identifier):
    if sys.platform != 'linux':
        raise ValueError('Local training requires the configured Linux/WSL runtime')
    service = Service()
    entry = next(e for e in service.approved('local') if e['id'] == identifier)
    path, plan = service.validate_entry(entry)
    from robocub.scripts.run_g1_queue import source_is_clean, validate_plan
    validate_plan(plan, ROOT)
    if not source_is_clean(ROOT):
        raise ValueError('Commit reviewed source and catalog before launch')
    if any(r.source == 'local' and ('running' in r.state or 'stale' in r.state) for r in service.runs()):
        raise ValueError('Existing running/stale queue must be reconciled before launching')
    import tempfile
    output = ROOT / 'Artifacts/Results/dashboard_launches' / identifier
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONUNBUFFERED='1', MUJOCO_GL='egl', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_autotune_level=0', JAX_COMPILATION_CACHE_DIR=tempfile.mkdtemp(prefix='g1-dashboard-'))
    env.pop('JAX_PLATFORMS', None)
    with (output / 'launcher.log').open('xb') as log:
        child = subprocess.Popen(['timeout', '--signal=INT', '--kill-after=60s', str(plan['max_wall_seconds'] + 30) + 's',
                                  sys.executable, '-u', str(ROOT / 'robocub/scripts/run_g1_queue.py'), '--plan', str(path)],
                                 cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    receipt = dict(wrapper_pid=child.pid, plan=entry['plan'], plan_sha256=entry['sha256'], started_at=utc())
    (output / 'launch_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


class Dashboard:
    def __init__(self, service=None):
        import tkinter as tk
        from tkinter import ttk, messagebox
        self.tk, self.ttk, self.messagebox = tk, ttk, messagebox
        self.service = service or Service()
        self.events = queue.Queue()
        self.busy = False
        self.checked_source = None
        self.health_cache = {}
        self.remote_cache = {}
        self.root = tk.Tk()
        self.root.title('Triton Droids | RoboCup control center')
        self.root.geometry('1100x760')
        self.root.minsize(900, 640)
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10))
        style.configure('Title.TLabel', font=('Segoe UI', 22, 'bold'))
        style.configure('Treeview', rowheight=29)
        outer = ttk.Frame(self.root, padding=22); outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='RoboCup control center', style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer, text='Unitree G1  /  Simulation research  /  Three-seed validation still pending').pack(anchor='w', pady=(2, 18))
        compute = ttk.LabelFrame(outer, text='1  Compute for the next launch', padding=12); compute.pack(fill='x')
        self.source = tk.StringVar(value='This laptop · RTX 4060 (WSL)')
        choices = ('This laptop · RTX 4060 (WSL)', 'Atlantis cluster')
        combo = ttk.Combobox(compute, textvariable=self.source, values=choices, state='readonly', width=36)
        combo.pack(side='left'); combo.bind('<<ComboboxSelected>>', lambda _: self.source_changed())
        ttk.Button(compute, text='Check compute', command=self.check).pack(side='left', padx=10)
        self.health = tk.StringVar(value='Unknown — click Check compute. Selection does not move existing work.')
        ttk.Label(outer, textvariable=self.health, wraplength=1000).pack(anchor='w', pady=8)
        actions = ttk.Frame(outer); actions.pack(fill='x', pady=8)
        ttk.Button(actions, text='Watch robot on field', command=self.watch).pack(side='left')
        ttk.Button(actions, text='Refresh saved progress', command=self.refresh).pack(side='left', padx=8)
        ttk.Button(actions, text='Open results', command=self.open_results).pack(side='left')
        ttk.Button(actions, text='Policies', command=lambda:self.open_path(ROOT/'App/Policies')).pack(side='left', padx=8)
        ttk.Button(actions, text='Recordings', command=self.recordings).pack(side='left')
        ttk.Button(actions, text='Help', command=lambda:self.open_path(ROOT/'START_HERE.md')).pack(side='left', padx=8)
        launch = ttk.Frame(outer); launch.pack(fill='x', pady=(0, 12))
        self.plan = tk.StringVar()
        self.plans = ttk.Combobox(launch, textvariable=self.plan, state='readonly', width=55); self.plans.pack(side='left')
        self.launch_button = ttk.Button(launch, text='Start approved plan', command=self.launch); self.launch_button.pack(side='left', padx=8)
        self.tabs = ttk.Notebook(outer); self.tabs.pack(fill='both', expand=True)
        work = ttk.Frame(self.tabs, padding=8); advanced = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(work, text='Current work & history'); self.tabs.add(advanced, text='Advanced & activity')
        self.table = ttk.Treeview(work, columns=('run','source','status','stage','step'), show='headings', height=11)
        for name, width in [('run',210),('source',80),('status',210),('stage',160),('step',120)]:
            self.table.heading(name, text=name.title()); self.table.column(name, width=width)
        bar = ttk.Scrollbar(work, orient='vertical', command=self.table.yview)
        self.table.configure(yscrollcommand=bar.set); bar.pack(side='right', fill='y'); self.table.pack(fill='both', expand=True)
        bottom = ttk.Frame(work); bottom.pack(fill='x', pady=(8,0))
        ttk.Button(bottom, text='Inspect selected', command=self.inspect).pack(side='left')
        ttk.Button(bottom, text='Check selected Slurm job', command=self.check_job).pack(side='left', padx=8)
        ttk.Button(bottom, text='Stop selected…', command=self.stop).pack(side='right')
        self.activity = tk.Text(advanced, wrap='word', font=('Consolas',10), height=16)
        self.activity.pack(fill='both', expand=True)
        self.log('Compute checks are explicit; saved local progress refreshes every 30 seconds.\n'
                 'Stale heartbeats are not running jobs. Cluster records require a Slurm check.\n'
                 'Plans are enabled by reviewed ID + SHA256 in App/Configs/dashboard_plans.json.\n'
                 'New compute providers require a health/launch/status/stop adapter, not just a display name.\n'
                 'Stopping keeps existing checkpoints but does not save an exact resumable training state.')
        self.rows = {}; self.refresh(); self.source_changed()
        self.root.after(100, self.drain); self.root.after(30000, self.tick)

    def source_id(self):
        return 'atlantis' if self.source.get() == 'Atlantis cluster' else 'local'

    def log(self, text):
        self.activity.configure(state='normal'); self.activity.insert('end', str(text) + '\n\n')
        self.activity.see('end'); self.activity.configure(state='disabled')

    def worker(self, task, kind='activity'):
        if self.busy:
            self.log('Another check or action is still in progress; wait for its result.')
            return
        self.busy = True
        def run():
            try:
                self.events.put((kind, task()))
            except subprocess.CalledProcessError:
                self.events.put(('error', 'Command failed. Inspect the recorded launcher/stage log; no automatic retry.'))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def drain(self):
        try:
            while True:
                kind, value = self.events.get_nowait(); self.busy = False
                if kind == 'health' and self.checked_source == self.source_id():
                    self.health.set(str(value).split('\n')[0] + ' · ' + utc())
                if kind == 'health':
                    self.health_cache[self.checked_source] = str(value).split('\n')[0] + ' · ' + utc()
                if kind.startswith('job:'):
                    match = re.search(r'\b(PENDING|RUNNING|COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)\b', str(value))
                    self.remote_cache[kind[4:]] = (match.group(0) if match else 'unknown', datetime.now(timezone.utc))
                    self.refresh()
                if kind == 'error': self.health.set('Check/action failed — see Advanced & activity')
                self.log(value)
        except queue.Empty:
            pass
        self.root.after(100, self.drain)

    def source_changed(self):
        self.health.set(self.health_cache.get(self.source_id(), 'Unknown — check this source. Existing runs stay on their original compute.'))
        self.entries = self.service.approved(self.source_id())
        labels = [e['id'] for e in self.entries]
        self.plans['values'] = labels
        self.plan.set(labels[0] if labels else 'No unlaunched approved plan registered')
        self.launch_button.configure(state='normal' if labels else 'disabled')

    def refresh(self):
        selected = self.table.selection()
        self.table.delete(*self.table.get_children())
        self.rows = {r.source + ':' + r.name:r for r in self.service.runs()}
        for key, r in self.rows.items():
            cached = self.remote_cache.get(key)
            if cached and (datetime.now(timezone.utc) - cached[1]).total_seconds() < 300:
                r.state = cached[0] + ' (checked ' + cached[1].strftime('%H:%M UTC') + ')'
            self.table.insert('', 'end', iid=key, values=(r.name,r.source,r.state,r.stage,r.step))
        if selected and selected[0] in self.rows: self.table.selection_set(selected[0])

    def tick(self):
        self.refresh(); self.root.after(30000, self.tick)

    def selected(self):
        keys = self.table.selection()
        return self.rows.get(keys[0]) if keys else None

    def check(self):
        if self.busy:
            return
        source = self.source_id()
        self.checked_source = source
        self.health.set('Checking ' + source + '…')
        self.worker(lambda:self.service.check(source), 'health')

    def inspect(self):
        r = self.selected()
        if r:
            self.log(json.dumps(read_json(r.status_path or r.receipt), indent=2)); self.tabs.select(1)

    def check_job(self):
        r = self.selected()
        if r and r.source == 'atlantis':
            self.worker(lambda: r.name + '\n' + self.service.remote_status(r), 'job:' + r.source + ':' + r.name)
            self.tabs.select(1)

    def stop(self):
        r = self.selected()
        if r and self.messagebox.askyesno('Stop this run?', f'Stop {r.name} on {r.source}?\nUnsaved progress is lost; saved checkpoints remain. No automatic restart.'):
            self.worker(lambda:self.service.stop(r))

    def launch(self):
        entry = next((e for e in self.entries if e['id'] == self.plan.get()), None)
        if entry and self.messagebox.askyesno('Launch frozen plan?', f"Launch {entry['id']} on {entry['source']}?\nOnly this registered plan and its fixed budget will run."):
            self.worker(lambda:self.service.launch(entry))

    def open_results(self):
        folder = self.selected().folder if self.selected() else ROOT / 'Artifacts/Results'
        self.open_path(folder)

    def open_path(self, folder):
        if os.name == 'nt': os.startfile(folder)
        else: subprocess.Popen(['xdg-open',str(folder)])

    def recordings(self):
        # An explicit user action; indexes files only, never loads a simulator.
        from App.Scripts.index_videos import build_index
        build_index(ROOT)
        self.open_path(ROOT/'Artifacts/Videos/index.html')

    def watch(self):
        log_path = ROOT / 'Artifacts/Results/dashboard_viewer.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a') as log:
            subprocess.Popen([sys.executable,str(ROOT/'App/robocub.py'),'watch','--scene','field'],cwd=ROOT,
                             stdout=log,stderr=subprocess.STDOUT)
        self.log('Viewer requested; startup diagnostics: ' + str(log_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-local')
    args = parser.parse_args(argv)
    if args.launch_local:
        launch_local(args.launch_local)
    else:
        Dashboard().root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
