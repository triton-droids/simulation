"""Provide the desktop dashboard and backend for Atlantis training sweeps.

This module gives ordinary users one visual place to submit RoboCup sweeps,
inspect running and completed jobs, download snapshots, read logs, and cancel
jobs. It deliberately delegates deployment and downloads to ``Scripts/sweep.sh``
and ``Scripts/return.sh`` so the tested shell workflow remains authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import queue
import re
import shutil
import subprocess
import threading
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"
OUTPUT_ROOT = ROOT / "Scripts" / "Outputs"
SWEEP_SCRIPT = ROOT / "Scripts" / "sweep.sh"
RETURN_SCRIPT = ROOT / "Scripts" / "return.sh"

RESOURCE_KEYS = (
    "SLURM_CPUS_PER_TASK",
    "SLURM_GPU_COUNT",
    "SLURM_GRES_TYPE",
    "SLURM_MEMORY",
    "SLURM_TIME",
)

JOB_LINE = re.compile(
    r"^\s*(Sweep_\d+) \| status=([^|]+) \| task=([^|]+) \| "
    r"policy=([^|]+) \| submitted=([^|]+) \| completed_trials=([^|]+) \| "
    r"exit_code=(.+)$"
)


@dataclass(frozen=True)
class ClusterJob:
    """Represent one sweep row reported by the remote return workflow.

    The dashboard uses this immutable record to populate its job table. Empty
    remote metadata is normalized by the shell script before this class is made.
    """

    label: str
    status: str
    task: str
    policy: str
    submitted: str
    completed_trials: int
    exit_code: str

    @property
    def active(self) -> bool:
        """Return whether Slurm may still be executing or queueing the job."""

        return self.status in {"launching", "starting", "running"}


def parse_env(path: Path = ENV_PATH) -> dict[str, str]:
    """Read simple ``KEY=value`` settings from the ignored project environment.

    Quotes around values are removed. Blank lines and comments are ignored.
    The function never prints values, which prevents key paths or future secret
    settings from leaking into the GUI log.
    """

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def update_env(updates: dict[str, str], path: Path = ENV_PATH) -> None:
    """Persist approved resource settings while preserving unrelated values.

    Only known Slurm resource keys may be changed. Existing order, comments,
    connection settings, and key paths remain untouched. Missing resource keys
    are appended so older local ``.env`` files upgrade cleanly.
    """

    unexpected = set(updates) - set(RESOURCE_KEYS)
    if unexpected:
        raise ValueError(f"Unsupported .env settings: {sorted(unexpected)}")

    lines = path.read_text(encoding="utf-8").splitlines()
    remaining = dict(updates)
    rendered: list[str] = []
    for line in lines:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=", line)
        if match and match.group(1) in remaining:
            name = match.group(1)
            rendered.append(f'{name}="{remaining.pop(name)}"')
        else:
            rendered.append(line)
    for name in RESOURCE_KEYS:
        if name in remaining:
            rendered.append(f'{name}="{remaining[name]}"')
    path.write_text("\n".join(rendered) + "\n", encoding="utf-8")


def parse_jobs(output: str) -> list[ClusterJob]:
    """Convert human-readable ``return.sh`` rows into structured job records."""

    jobs: list[ClusterJob] = []
    for line in output.splitlines():
        match = JOB_LINE.match(line)
        if not match:
            continue
        label, status, task, policy, submitted, completed, exit_code = (
            value.strip() for value in match.groups()
        )
        jobs.append(
            ClusterJob(
                label=label,
                status=status,
                task=task,
                policy=policy,
                submitted=submitted,
                completed_trials=int(completed),
                exit_code=exit_code,
            )
        )
    return jobs


def find_git_bash() -> Path:
    """Locate Git Bash on Windows or a compatible Bash executable elsewhere."""

    candidates = [
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    discovered = shutil.which("bash")
    if discovered:
        return Path(discovered)
    raise FileNotFoundError("Git Bash was not found. Install Git for Windows first.")


def windows_path(git_bash_path: str) -> str:
    """Convert a Git Bash ``/c/...`` path into a native Windows path."""

    match = re.match(r"^/([A-Za-z])/(.*)$", git_bash_path)
    if not match:
        return git_bash_path
    drive, remainder = match.groups()
    return str(Path(f"{drive.upper()}:/{remainder}"))


class ClusterService:
    """Own subprocess operations used by the visual dashboard.

    Long-running methods are safe to call from worker threads. They return
    captured output for display and raise ``CalledProcessError`` when the shell
    workflow reports a real failure.
    """

    def __init__(self, root: Path = ROOT) -> None:
        """Bind the service to one repository checkout and locate Git Bash."""

        self.root = root
        self.bash = find_git_bash()

    def _run(self, command: list[str], *, timeout: int | None = None) -> str:
        """Run one command without a shell and return combined readable output."""

        completed = subprocess.run(
            command,
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout

    def list_jobs(self) -> tuple[list[ClusterJob], str]:
        """Refresh remote status through ``return.sh`` without downloading files."""

        output = self._run([str(self.bash), str(RETURN_SCRIPT)], timeout=45)
        return parse_jobs(output), output

    def submit(self, *, smoke: bool, video: bool) -> str:
        """Commit, push, and submit a new sweep using the saved GUI resources."""

        arguments = [str(self.bash), str(SWEEP_SCRIPT)]
        if smoke:
            arguments.append("--smoke")
        arguments.append("--video" if video else "--no-video")
        return self._run(arguments, timeout=900)

    def download(self, label: str) -> str:
        """Download every currently available file for a validated sweep label."""

        self._validate_label(label)
        return self._run([str(self.bash), str(RETURN_SCRIPT), label], timeout=180)

    def cancel(self, label: str) -> str:
        """Cancel the Slurm job recorded for a sweep without deleting its files."""

        self._validate_label(label)
        settings = parse_env(self.root / ".env")
        ssh_key = windows_path(settings["SSH_KEY_PATH"])
        remote_pid = f'{settings["SSH_DIRECTORY"]}/Scripts/Outputs/{label}/.remote_pid'
        remote_command = (
            f'test -f "{remote_pid}" && job_id=$(cat "{remote_pid}") && '
            'case "$job_id" in (*[!0-9]*|"") exit 2;; esac && '
            'scancel "$job_id" && printf "Cancelled Slurm job %s\\n" "$job_id"'
        )
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-i",
            ssh_key,
            "-p",
            settings["SSH_PORT"],
            settings["ATLANTIS_HOST"],
            remote_command,
        ]
        return self._run(command, timeout=45)

    def local_log(self, label: str, *, lines: int = 160) -> str:
        """Return the tail of the latest downloaded runner log, if available."""

        self._validate_label(label)
        path = self.root / "Scripts" / "Outputs" / label / "remote_runner.log"
        if not path.is_file():
            return "No downloaded runner log yet. Click Download Snapshot first."
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])

    @staticmethod
    def _validate_label(label: str) -> None:
        """Reject values that could escape the expected remote sweep directory."""

        if not re.fullmatch(r"Sweep_[0-9]+", label):
            raise ValueError(f"Invalid sweep label: {label!r}")


def _import_tkinter():
    """Import Tkinter lazily so backend tests work on headless systems."""

    import tkinter as tk
    from tkinter import messagebox, ttk

    return tk, ttk, messagebox


class ClusterDashboard:
    """Render and coordinate the beginner-facing Atlantis desktop interface."""

    def __init__(self, service: ClusterService | None = None) -> None:
        """Create widgets, load saved resources, and start the first refresh."""

        tk, ttk, messagebox = _import_tkinter()
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.service = service or ClusterService()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.jobs: dict[str, ClusterJob] = {}

        self.root = tk.Tk()
        self.root.title("RoboCup Atlantis Training Dashboard")
        self.root.geometry("1180x760")
        self.root.minsize(960, 620)
        self._build_widgets()
        self._load_resources()
        self.root.after(100, self._drain_events)
        self.refresh()

    def _build_widgets(self) -> None:
        """Build the resource controls, job table, actions, and live log panel."""

        ttk = self.ttk
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        resources = ttk.LabelFrame(outer, text="New sweep resources", padding=10)
        resources.pack(fill="x")

        self.cpu = self._combo(resources, "CPU cores", ("1", "2", "4", "8", "16"), 0)
        self.gpu_count = self._combo(resources, "GPUs", ("1", "2", "4"), 2)
        self.gpu_type = self._combo(resources, "GPU type", ("rtx2080ti",), 4)
        self.memory = self._combo(resources, "Memory", ("16G", "32G", "64G", "128G"), 6)
        self.time_limit = self._combo(resources, "Time limit", ("00:30:00", "01:00:00", "04:00:00", "08:00:00"), 8)

        self.smoke = self.tk.BooleanVar(value=True)
        self.video = self.tk.BooleanVar(value=False)
        ttk.Checkbutton(resources, text="Smoke sweep", variable=self.smoke).grid(row=0, column=10, padx=8)
        ttk.Checkbutton(resources, text="Record videos", variable=self.video).grid(row=0, column=11, padx=8)
        ttk.Button(resources, text="Save settings", command=self.save_resources).grid(row=0, column=12, padx=5)
        ttk.Button(resources, text="Submit new sweep", command=self.submit).grid(row=0, column=13, padx=5)

        note = ttk.Label(
            outer,
            text="Running Slurm jobs cannot be resized. Change resources, then cancel and submit a replacement if needed.",
        )
        note.pack(anchor="w", pady=(6, 8))

        table_frame = ttk.LabelFrame(outer, text="Atlantis sweeps", padding=8)
        table_frame.pack(fill="both", expand=True)
        columns = ("status", "job", "policy", "submitted", "trials", "exit")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=13)
        headings = {
            "status": "Status",
            "job": "Sweep",
            "policy": "Policy",
            "submitted": "Submitted (UTC)",
            "trials": "Completed trials",
            "exit": "Exit code",
        }
        widths = {"status": 100, "job": 100, "policy": 120, "submitted": 190, "trials": 120, "exit": 90}
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], anchor="center")
        self.table.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(table_frame, command=self.table.yview)
        scrollbar.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.bind("<<TreeviewSelect>>", lambda _event: self.show_selected_log())

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=8)
        ttk.Button(actions, text="Refresh", command=self.refresh).pack(side="left", padx=3)
        ttk.Button(actions, text="Download snapshot", command=self.download).pack(side="left", padx=3)
        ttk.Button(actions, text="Show downloaded log", command=self.show_selected_log).pack(side="left", padx=3)
        ttk.Button(actions, text="Cancel selected job", command=self.cancel).pack(side="left", padx=3)
        ttk.Button(actions, text="Cancel + resubmit", command=self.resubmit).pack(side="left", padx=3)

        self.connection = self.tk.StringVar(value="Connecting to Atlantis...")
        ttk.Label(actions, textvariable=self.connection).pack(side="right")

        log_frame = ttk.LabelFrame(outer, text="Activity and training log", padding=8)
        log_frame.pack(fill="both", expand=True)
        self.log = self.tk.Text(log_frame, height=13, wrap="none", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")

    def _combo(self, parent, label: str, values: Iterable[str], column: int):
        """Create one labeled read-only resource dropdown and return its variable."""

        self.ttk.Label(parent, text=label).grid(row=0, column=column, padx=(4, 2))
        variable = self.tk.StringVar()
        combo = self.ttk.Combobox(parent, textvariable=variable, values=tuple(values), state="readonly", width=10)
        combo.grid(row=0, column=column + 1, padx=(0, 8))
        return variable

    def _load_resources(self) -> None:
        """Populate dropdowns from the ignored local ``.env`` configuration."""

        settings = parse_env()
        self.cpu.set(settings.get("SLURM_CPUS_PER_TASK", "8"))
        self.gpu_count.set(settings.get("SLURM_GPU_COUNT", "1"))
        self.gpu_type.set(settings.get("SLURM_GRES_TYPE", "rtx2080ti"))
        self.memory.set(settings.get("SLURM_MEMORY", "32G"))
        self.time_limit.set(settings.get("SLURM_TIME", "04:00:00"))

    def save_resources(self) -> bool:
        """Validate dropdown values and persist them for the next submission."""

        updates = {
            "SLURM_CPUS_PER_TASK": self.cpu.get(),
            "SLURM_GPU_COUNT": self.gpu_count.get(),
            "SLURM_GRES_TYPE": self.gpu_type.get(),
            "SLURM_MEMORY": self.memory.get(),
            "SLURM_TIME": self.time_limit.get(),
        }
        if not updates["SLURM_CPUS_PER_TASK"].isdigit() or not updates["SLURM_GPU_COUNT"].isdigit():
            self.messagebox.showerror("Invalid resources", "CPU and GPU counts must be positive integers.")
            return False
        update_env(updates)
        self._append("Saved resource settings for the next sweep.\n")
        return True

    def refresh(self) -> None:
        """Fetch current remote jobs without blocking Tkinter's event loop."""

        self.connection.set("Refreshing...")
        self._worker("jobs", self.service.list_jobs)

    def submit(self) -> None:
        """Save selections and submit a fresh sweep in a background thread."""

        if not self.save_resources():
            return
        self.connection.set("Submitting sweep...")
        self._append("Submitting a new sweep. This may commit and push local changes.\n")
        self._worker("submit", lambda: self.service.submit(smoke=self.smoke.get(), video=self.video.get()))

    def download(self) -> None:
        """Download a snapshot for the selected sweep and show its local log."""

        label = self._selected_label()
        if not label:
            return
        self.connection.set(f"Downloading {label}...")
        self._worker("download", lambda: self.service.download(label))

    def cancel(self) -> None:
        """Confirm and cancel the selected active job without deleting outputs."""

        label = self._selected_label()
        if not label:
            return
        job = self.jobs[label]
        if not job.active:
            self.messagebox.showinfo("Job already finished", f"{label} is {job.status}; there is nothing to cancel.")
            return
        if not self.messagebox.askyesno("Cancel Slurm job", f"Cancel active job {label}? Remote files will be kept."):
            return
        self.connection.set(f"Cancelling {label}...")
        self._worker("cancel", lambda: self.service.cancel(label))

    def resubmit(self) -> None:
        """Cancel the selected active job, then submit with current resources."""

        label = self._selected_label()
        if not label or not self.save_resources():
            return
        job = self.jobs[label]
        action = "cancel it and submit a replacement" if job.active else "submit a replacement"
        if not self.messagebox.askyesno("Resubmit sweep", f"{label} is {job.status}. {action.capitalize()}?"):
            return

        def replace() -> str:
            output = ""
            if job.active:
                output += self.service.cancel(label)
            output += self.service.submit(smoke=self.smoke.get(), video=self.video.get())
            return output

        self.connection.set(f"Replacing {label}...")
        self._worker("submit", replace)

    def show_selected_log(self) -> None:
        """Display the latest locally downloaded runner log for the selection."""

        label = self._selected_label(show_error=False)
        if label:
            self._replace_log(self.service.local_log(label))

    def _selected_label(self, *, show_error: bool = True) -> str | None:
        """Return the selected sweep label or explain that selection is required."""

        selected = self.table.selection()
        if selected:
            return selected[0]
        if show_error:
            self.messagebox.showinfo("Select a sweep", "Select a sweep row first.")
        return None

    def _worker(self, event_name: str, operation: Callable[[], object]) -> None:
        """Run network or deployment work off the GUI thread and report results."""

        def run() -> None:
            try:
                self.events.put((event_name, operation()))
            except Exception as error:  # Tkinter must present subprocess failures.
                self.events.put(("error", error))

        threading.Thread(target=run, daemon=True).start()

    def _drain_events(self) -> None:
        """Apply completed worker results on Tkinter's single UI thread."""

        try:
            while True:
                event_name, payload = self.events.get_nowait()
                if event_name == "jobs":
                    jobs, output = payload
                    self._populate(jobs)
                    self._append(output)
                    self.connection.set(f"Connected: {len(jobs)} sweep(s)")
                elif event_name in {"submit", "cancel"}:
                    self._append(str(payload))
                    self.connection.set("Operation completed; refreshing...")
                    self.refresh()
                elif event_name == "download":
                    self._append(str(payload))
                    self.connection.set("Snapshot downloaded")
                    self.show_selected_log()
                    self.refresh()
                elif event_name == "error":
                    self.connection.set("Operation failed")
                    self._append(f"ERROR: {payload}\n")
                    self.messagebox.showerror("Cluster operation failed", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _populate(self, jobs: list[ClusterJob]) -> None:
        """Replace table rows while keeping running jobs above finished jobs."""

        selected = self.table.selection()
        for item in self.table.get_children():
            self.table.delete(item)
        order = {"running": 0, "starting": 0, "launching": 0, "succeeded": 1, "failed": 2}
        for job in sorted(jobs, key=lambda item: (order.get(item.status, 3), item.label)):
            self.jobs[job.label] = job
            self.table.insert(
                "",
                "end",
                iid=job.label,
                values=(job.status, job.label, job.policy, job.submitted, job.completed_trials, job.exit_code),
            )
        if selected and self.table.exists(selected[0]):
            self.table.selection_set(selected[0])

    def _append(self, text: str) -> None:
        """Append output to the activity panel and keep the latest line visible."""

        self.log.configure(state="normal")
        self.log.insert("end", text if text.endswith("\n") else text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _replace_log(self, text: str) -> None:
        """Replace the activity panel with the selected downloaded training log."""

        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("1.0", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def run(self) -> None:
        """Enter Tkinter's event loop and keep the dashboard open."""

        self.root.mainloop()


def main() -> int:
    """Launch the visual Atlantis dashboard and return after its window closes."""

    ClusterDashboard().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
