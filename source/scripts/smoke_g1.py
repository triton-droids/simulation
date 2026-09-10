"""Run deterministic Gate 3 G1 MJX smoke diagnostics and optional videos."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jp
import mediapy as media
import numpy as np
from omegaconf import OmegaConf

from source.config.g1 import G1MJXConfig
from source.locomotion.unitree_g1.joystick import Joystick
from source.robots.unitree_g1 import UnitreeG1Model


def _git_record() -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _bounded_action(index: jax.Array, action_size: int, amplitude: float) -> jax.Array:
    return amplitude * jp.sin(0.035 * index + jp.arange(action_size) * 0.17)


def run_scan(
    env: Joystick, initial_state, steps: int, amplitude: float
) -> tuple[object, dict[str, np.ndarray]]:
    """JIT one fixed bounded-action rollout and return diagnostic traces."""

    def rollout(state):
        def one_step(carry, index):
            action = _bounded_action(index, env.nu, amplitude)
            carry = env.step(carry, action)
            finite = (
                jp.isfinite(carry.pipeline_state.q).all()
                & jp.isfinite(carry.pipeline_state.qd).all()
                & jp.isfinite(carry.obs["state"]).all()
                & jp.isfinite(carry.obs["privileged_state"]).all()
            )
            values = {
                "finite": finite,
                "done": carry.done,
                "pelvis_height": carry.pipeline_state.q[2],
                "undesired_contact": carry.info["undesired_contact"],
            }
            return carry, values

        return jax.lax.scan(one_step, state, jp.arange(steps))

    final_state, traces = jax.jit(rollout)(initial_state)
    jax.block_until_ready(traces["finite"])
    return final_state, {key: np.asarray(value) for key, value in traces.items()}


def video_rollout(env: Joystick, initial_state, steps: int, amplitude: float):
    step = jax.jit(env.step)
    state = initial_state
    rollout = [state.pipeline_state]
    for index in range(steps):
        action = _bounded_action(jp.asarray(index), env.nu, amplitude)
        state = step(state, action)
        rollout.append(state.pipeline_state)
    return state, rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-amplitude", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=Path("results/gate3"))
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--no-fetch-model", action="store_false", dest="fetch_model")
    parser.set_defaults(fetch_model=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    cfg = OmegaConf.structured(G1MJXConfig())
    cfg.reset.randomize = False
    cfg.noise.add_noise = False
    cfg.push.add_push = False
    cfg.domain_rand.add_domain_rand = False
    robot = UnitreeG1Model(fetch=args.fetch_model)
    env = Joystick("unitree_g1", robot, "flat", cfg)
    initial_state = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
    final_state, traces = run_scan(env, initial_state, args.steps, args.action_amplitude)

    done_indices = np.flatnonzero(traces["done"] > 0)
    result = {
        "kind": "gate3_pipeline_smoke_not_locomotion_evidence",
        "seed": args.seed,
        "steps": args.steps,
        "control_dt_seconds": env.dt,
        "action_amplitude": args.action_amplitude,
        "all_finite": bool(traces["finite"].all()),
        "first_done_step": int(done_indices[0] + 1) if done_indices.size else None,
        "done_step_count": int(done_indices.size),
        "final_pelvis_height": float(final_state.pipeline_state.q[2]),
        "actor_observation_size": env.obs_size,
        "privileged_observation_size": env.privileged_obs_size,
        "action_size": env.action_size,
        "model": robot.source_record,
        "git": _git_record(),
        "platform": platform.platform(),
        "jax_devices": [str(device) for device in jax.devices()],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mujoco", "jax", "jaxlib", "brax")
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.video:
        _, standing = video_rollout(env, initial_state, 50, 0.0)
        _, bounded = video_rollout(env, initial_state, 100, args.action_amplitude)
        media.write_video(
            args.output_dir / "standing_zero_action.mp4",
            env.render(standing, height=480, width=640, camera="track"),
            fps=1.0 / env.dt,
        )
        media.write_video(
            args.output_dir / "bounded_action.mp4",
            env.render(bounded, height=480, width=640, camera="track"),
            fps=1.0 / env.dt,
        )
        result["videos"] = ["standing_zero_action.mp4", "bounded_action.mp4"]

    result["wall_time_seconds"] = time.time() - started
    output = args.output_dir / "smoke.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_finite"]:
        raise SystemExit("Gate 3 smoke produced a nonfinite value.")


if __name__ == "__main__":
    main()
