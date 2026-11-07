"""Coordinate the small public scripts without loading simulator dependencies.

This module owns command parsing, canonical model paths, run-directory naming,
and subprocess launching. Keeping that work here lets every file in `App/Scripts/`
remain an obvious one-purpose button for non-expert users.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
T02_RUN = ROOT / "App" / "Models" / "T02"
T02_CHECKPOINT = 1_003_520
T02_CHECKPOINT_DIR = T02_RUN / "logs" / "checkpoints" / str(T02_CHECKPOINT)
T02_POLICY = T02_CHECKPOINT_DIR / "policy"


def _python() -> str:
    """Return the active interpreter so child commands reuse this environment."""

    return sys.executable


def _run(script: Path, arguments: Sequence[str] = ()) -> int:
    """Run one repository script from the repository root."""

    command = [_python(), str(script), *arguments]
    print("\nLaunching:", " ".join(command), flush=True)
    print(f"Working directory: {ROOT}\n", flush=True)
    return subprocess.call(command, cwd=ROOT, env=os.environ.copy())


def _checkpoint_status() -> dict[str, object]:
    """Describe the canonical T02 checkpoint without loading JAX or MuJoCo."""

    return {
        "policy": "T02 rate_cost",
        "checkpoint": T02_CHECKPOINT,
        "run_directory": str(T02_RUN),
        "checkpoint_directory": str(T02_CHECKPOINT_DIR),
        "weights_file": str(T02_POLICY),
        "weights_present": T02_POLICY.is_file(),
        "config_present": (T02_RUN / "resolved_config.json").is_file(),
    }


def command_watch(args: argparse.Namespace) -> int:
    """Validate T02 and open either the field or flat MuJoCo viewer."""

    status = _checkpoint_status()
    if not status["weights_present"]:
        print(f"Missing policy weights: {T02_POLICY}", file=sys.stderr)
        print("Provision the T02 handoff artifacts before starting the viewer.", file=sys.stderr)
        return 2

    script = ROOT / "robocub" / "tools" / (
        "watch_g1_soccer.py" if args.scene == "field" else "watch_g1_walk.py"
    )
    print("Robot: Unitree G1")
    print("Simulator: MuJoCo")
    print(f"Scene: {args.scene}")
    print(f"Policy: T02 checkpoint {T02_CHECKPOINT}")
    print(f"Weights: {T02_POLICY}")
    print("A separate window named 'MuJoCo : g1 scene' will open after JAX compilation.")
    print("If you only see command text, use Alt-Tab and select the MuJoCo window.")
    forwarded = list(args.extra)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    if args.scene == "field" and args.record:
        forwarded.extend(["--record", args.record])
    return _run(script, forwarded)


def command_train(args: argparse.Namespace) -> int:
    """Create an isolated run directory and launch the selected PPO profile."""

    profile = "ppo_g1_smoke" if args.smoke else "ppo_g1"
    print("Robot: Unitree G1")
    print("Simulator/trainer: MJX + Brax PPO")
    print(f"Training profile: {profile}")
    if not args.smoke:
        print("WARNING: this starts the historical 5,000,000-step profile.")
        print("The repository does not claim this recipe is a validated locomotion baseline.")
    forwarded = list(args.extra)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    overrides = [
        "--logger", args.logger,
        "--video" if args.video else "--no-video",
        "--seed", str(args.seed),
        "env=unitree_g1",
        f"agent={profile}",
        "robot=unitree_g1",
        "sim=unitree_g1",
        *forwarded,
    ]
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        base = ROOT / "Artifacts" / "Runs" / time.strftime("run_%Y%m%d_%H%M%S")
        run_dir = base
        suffix = 2
        while run_dir.exists():
            run_dir = base.with_name(f"{base.name}_{suffix:02d}")
            suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    overrides.extend([f"hydra.run.dir={run_dir}", "hydra.job.chdir=true"])
    print(f"Run artifacts: {run_dir}")
    return _run(ROOT / "robocub" / "scripts" / "train.py", overrides)


def command_evaluate(args: argparse.Namespace) -> int:
    """Run the audited fixed-command evaluator against the canonical T02 policy."""

    if not T02_POLICY.is_file():
        print(f"Missing policy weights: {T02_POLICY}", file=sys.stderr)
        return 2
    arguments = [
        "--run-dir", str(T02_RUN),
        "--checkpoint", str(T02_CHECKPOINT),
        "--steps", str(args.steps),
        "--seeds", args.seeds,
        "--nominal-reset",
        *args.extra,
    ]
    if args.video:
        arguments.append("--video")
    return _run(ROOT / "robocub" / "scripts" / "evaluate_g1.py", arguments)


def command_status(args: argparse.Namespace) -> int:
    """Print the selected runtimes, checkpoint, and artifact availability."""

    status = _checkpoint_status()
    latest_status = ROOT / "Artifacts" / "Runs" / "latest" / "status.json"
    print("ROBOCUP STATUS")
    print("==============")
    print("Robot: Unitree G1")
    print("Walking runtime: MuJoCo")
    print("Training runtime: MJX + Brax PPO")
    print("Field runtime: Isaac Lab/MJWarp infrastructure; T02 policy transfer not implemented")
    print(f"Selected policy: {status['policy']}")
    print(f"Checkpoint: {status['checkpoint']:,}")
    print(f"Weights: {status['weights_file']}")
    print(f"Weights present: {'YES' if status['weights_present'] else 'NO'}")
    print(f"Resolved config present: {'YES' if status['config_present'] else 'NO'}")
    print(f"Active standardized run status: {latest_status if latest_status.exists() else 'none'}")
    if args.json:
        print("\n" + json.dumps(status, indent=2))
    return 0 if status["weights_present"] else 2


def command_doctor(_args: argparse.Namespace) -> int:
    """Check the local files required by every public workflow."""

    checks = {
        "Python 3.12+": sys.version_info >= (3, 12),
        "T02 weights": T02_POLICY.is_file(),
        "T02 resolved config": (T02_RUN / "resolved_config.json").is_file(),
        "MuJoCo walking launcher": (ROOT / "robocub" / "tools" / "watch_g1_walk.py").is_file(),
        "Field walking launcher": (ROOT / "robocub" / "tools" / "watch_g1_soccer.py").is_file(),
        "Training launcher": (ROOT / "robocub" / "scripts" / "train.py").is_file(),
    }
    for label, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
    return 0 if all(checks.values()) else 2


def build_parser() -> argparse.ArgumentParser:
    """Build the shared command parser used by the public Scripts wrappers."""

    parser = argparse.ArgumentParser(
        description="Watch, train, evaluate, and inspect the Triton Droids G1 simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    watch = commands.add_parser("watch", help="Open a live MuJoCo camera for the saved T02 walker.")
    watch.add_argument("--scene", choices=("field", "flat"), default="field")
    watch.add_argument("--record", default=None, help="Field mode recording output directory.")
    watch.add_argument("extra", nargs=argparse.REMAINDER, help="Arguments forwarded to the viewer.")
    watch.set_defaults(handler=command_watch)

    train = commands.add_parser("train", help="Start G1 PPO training.")
    train.add_argument("--smoke", action="store_true", help="Run the 1,024-step integration profile.")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--run-dir", default=None, help="Exact output directory; defaults to Artifacts/Runs/run_TIMESTAMP.")
    train.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create periodic and final rollout videos.",
    )
    train.add_argument("--logger", choices=("local", "none", "wandb-offline", "wandb"), default="local")
    train.add_argument("extra", nargs=argparse.REMAINDER, help="Hydra overrides forwarded to training.")
    train.set_defaults(handler=command_train)

    evaluate = commands.add_parser("evaluate", help="Evaluate the saved T02 checkpoint.")
    evaluate.add_argument("--steps", type=int, default=500)
    evaluate.add_argument("--seeds", default="2000")
    evaluate.add_argument("--video", action="store_true")
    evaluate.add_argument("extra", nargs=argparse.REMAINDER)
    evaluate.set_defaults(handler=command_evaluate)

    status = commands.add_parser("status", help="Show the active policy, weights, and runtime state.")
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=command_status)

    doctor = commands.add_parser("doctor", help="Check launchers and required local artifacts.")
    doctor.set_defaults(handler=command_doctor)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse a public command and return its process exit status."""

    args = build_parser().parse_args(argv)
    return args.handler(args)
