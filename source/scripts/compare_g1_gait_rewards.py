"""Compare a saved local policy and the exact ONNX oracle in the SAME MJX task.

Diagnostic only: no policy fitting or imitation. Saves all per-step terms,
actions, positions, contacts and velocities for an explicitly fixed command.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from source.scripts import evaluate_g1 as ev
from source.scripts import evaluate_playground_onnx as oracle
from source.utils.g1_terminal_diagnostic import terminal_signals
from brax.io import model
from brax.training.agents.ppo import networks
import jax
import jax.numpy as jp
import numpy as np
from omegaconf import OmegaConf
import onnxruntime as ort


def _policy_label(value: str) -> str:
    if not value.isidentifier() or value.lower() == "oracle":
        raise argparse.ArgumentTypeError("Use an identifier other than oracle for the local policy")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--policy-label", type=_policy_label, default="C06")
    parser.add_argument("--command", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                        metavar=("VX", "VY", "YAW"))
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--randomized-reset", action="store_true")
    parser.add_argument("--evaluation-reset", action="store_true",
                        help="Use evaluate_g1's split reset key and sampled gait frequency.")
    parser.add_argument("--reset-ablation", choices=("full", "no_velocity", "no_linear", "no_angular", "no_horizontal", "no_vertical", "nominal_joints"), default="full",
                        help="Diagnostic only: remove one reset perturbation; never a final validation gate.")
    args = parser.parse_args()
    if not np.isfinite(args.command).all():
        parser.error("--command values must be finite")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    cfg = OmegaConf.load(args.run_dir / "resolved_config.json")
    if "recovery_reset_candidates" in cfg.sim.playground:
        cfg.sim.playground.recovery_reset_candidates = 1
    cfg.robot.fetch_model = False
    cfg.sim.playground.fetch_source = False
    cfg.sim.reset.randomize = args.randomized_reset
    if args.reset_ablation == "no_velocity":
        cfg.sim.reset.base_velocity_range = 0.0
    elif args.reset_ablation == "nominal_joints":
        cfg.sim.reset.joint_scale_range = [1.0, 1.0]
    cfg.sim.noise.add_noise = False
    cfg.sim.push.add_push = False
    env = ev.get_env_class(cfg.env.name)(cfg.robot.name, ev.make_robot(cfg.robot), cfg.env.terrain, cfg.sim)
    factory = functools.partial(networks.make_ppo_networks,
        policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=cfg.agent.value_hidden_layer_sizes,
        policy_obs_key="state", value_obs_key="privileged_state")
    net = ev._make_evaluation_network(factory, env.observation_size, 29,
        normalize_observations=cfg.agent.normalize_observations)
    params = model.load_params(args.run_dir / "logs/checkpoints" / str(args.checkpoint) / "policy")
    policy = jax.jit(networks.make_inference_fn(net)(params, deterministic=True))
    path = oracle._default_policy_path()
    assert oracle._sha256(path) == oracle.EXPECTED_ONNX_SHA256
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    command = jp.array(args.command)
    def diagnostic_reset(state):
        if args.reset_ablation == "full":
            return state
        from mujoco import mjx
        qpos, qvel = state.data.qpos, state.data.qvel
        if args.reset_ablation == "no_velocity":
            qvel = jp.zeros_like(qvel)
        elif args.reset_ablation == "no_linear":
            qvel = qvel.at[:3].set(0)
        elif args.reset_ablation == "no_angular":
            qvel = qvel.at[3:6].set(0)
        elif args.reset_ablation == "no_horizontal":
            qvel = qvel.at[:2].set(0)
        elif args.reset_ablation == "no_vertical":
            qvel = qvel.at[2:3].set(0)
        else:
            qpos = qpos.at[7:].set(env._env._init_q[7:])
        data = env._mjx_env_module.make_data(
            env._env.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
            impl=env._env.mjx_model.impl.value,
            naconmax=env._env._config.naconmax, njmax=env._env._config.njmax)
        data = mjx.forward(env._env.mjx_model, data)
        contact = env._contact(data)
        info = dict(state.info)
        info.update(last_contact=contact, feet_air_time=jp.zeros(2),
                    swing_peak=jp.zeros(2), motor_targets=data.ctrl)
        obs = env._env._get_obs(data, info, contact)
        return state.replace(data=data, obs=obs, info=info)
    reset = jax.jit(diagnostic_reset)
    unmodified_reset = jax.jit(env.reset)
    step = jax.jit(lambda state, action: ev._step_with_held_command(env, state, action, command))
    rows = {}
    initial_states = {}
    for controller in (args.policy_label, "oracle"):
        reset_key = jax.random.PRNGKey(args.seed)
        if args.evaluation_reset:
            reset_key = jax.random.split(reset_key)[0]
        original = unmodified_reset(reset_key)
        state = ev._replace_command(reset(original), command)
        if args.reset_ablation == "no_velocity":
            np.testing.assert_allclose(np.asarray(state.data.qpos), np.asarray(original.data.qpos), rtol=0, atol=1e-6)
            assert np.all(np.asarray(state.data.qvel) == 0)
            assert not np.array_equal(np.asarray(state.data.qvel), np.asarray(original.data.qvel))
        elif args.reset_ablation in ("no_linear", "no_angular", "no_horizontal", "no_vertical"):
            removed = {"no_linear": slice(0, 3), "no_angular": slice(3, 6), "no_horizontal": slice(0, 2), "no_vertical": slice(2, 3)}[args.reset_ablation]
            retained = [i for i in range(env.nv) if i not in range(removed.start, removed.stop)]
            np.testing.assert_allclose(np.asarray(state.data.qpos), np.asarray(original.data.qpos), rtol=0, atol=1e-6)
            assert np.all(np.asarray(state.data.qvel)[removed] == 0)
            assert not np.array_equal(np.asarray(state.data.qvel)[removed], np.asarray(original.data.qvel)[removed])
            np.testing.assert_allclose(np.asarray(state.data.qvel)[retained], np.asarray(original.data.qvel)[retained], rtol=0, atol=1e-6)
        elif args.reset_ablation == "nominal_joints":
            np.testing.assert_allclose(np.asarray(state.data.qpos[:7]), np.asarray(original.data.qpos[:7]), rtol=0, atol=1e-6)
            np.testing.assert_allclose(np.asarray(state.data.qvel), np.asarray(original.data.qvel), rtol=0, atol=1e-6)
            assert np.array_equal(np.asarray(state.data.qpos[7:]), np.asarray(env._env._init_q[7:]))
            assert not np.array_equal(np.asarray(state.data.qpos[7:]), np.asarray(original.data.qpos[7:]))
        info = dict(state.info)
        if not args.evaluation_reset:
            info["phase_dt"] = jp.asarray([2 * np.pi * 1.5 * env.dt])
        state = state.replace(info=info)
        initial_states[controller] = {
            "qpos": np.asarray(state.data.qpos).tolist(),
            "qvel": np.asarray(state.data.qvel).tolist(),
            "phase_dt": np.asarray(state.info["phase_dt"]).tolist(),
        }
        trace = []
        for index in range(args.steps):
            if controller != "oracle":
                action = policy(state.obs, jax.random.PRNGKey(index))[0]
            else:
                action = jp.asarray(session.run(None, {"obs": np.asarray(state.obs["state"])[None]})[0][0])
            state = step(state, action)
            sample = {"action": action, "qpos": state.data.qpos,
                "foot_positions": state.data.site_xpos[env._feet_site_id],
                "velocity": env.get_local_linvel(state.data, "pelvis"),
                "gyro": env.get_gyro(state.data, "pelvis"),
                "contact": env._contact(state.data), "phase": state.info["phase"],
                "reward": state.reward, "done": state.done, **state.metrics}
            signals = terminal_signals(env, state.data)
            assert bool(state.done) == any(bool(v) for k, v in signals.items() if k.startswith("terminal/")), "Termination decomposition mismatch"
            sample.update(signals)
            trace.append(jax.tree.map(np.asarray, sample))
            if bool(state.done):
                break
        arrays = {name: np.stack([item[name] for item in trace]) for name in trace[0]}
        np.savez_compressed(args.output_dir / f"{controller}_trace.npz", **arrays)
        contact = arrays["contact"]
        air_intervals = [ev._completed_air_intervals(contact[:, foot], env.dt) for foot in range(2)]
        rows[controller] = {
            "steps": len(trace), "terminal": bool(arrays["done"][-1]),
            "terminal_causes": [k.removeprefix("terminal/") for k, v in arrays.items() if k.startswith("terminal/") and bool(v[-1])],
            "first_trigger_step": {k.removeprefix("terminal/"): int(np.flatnonzero(v)[0])+1 if np.any(v) else None for k, v in arrays.items() if k.startswith("terminal/")},
            "mean_velocity": arrays["velocity"].mean(0).tolist(),
            "linear_rmse": float(np.sqrt(np.mean(np.sum((arrays["velocity"][:, :2] - np.asarray(args.command[:2])) ** 2, axis=1)))),
            "mean_yaw_rate": float(arrays["gyro"][:, 2].mean()),
            "yaw_rate_std": float(arrays["gyro"][:, 2].std()),
            "yaw_rmse": float(np.sqrt(np.mean((arrays["gyro"][:, 2] - args.command[2]) ** 2))),
            "minimum_pelvis_height": float(arrays["qpos"][:, 2].min()),
            "single_support_fraction": float(np.mean(contact.sum(1) == 1)),
            "foot_transitions": np.sum(contact[1:] != contact[:-1], axis=0).tolist(),
            "completed_air_intervals_seconds": air_intervals,
            "median_completed_air_interval_seconds": [float(np.median(v)) if v else 0.0 for v in air_intervals],
            "raw_action_saturation_fraction": float(np.mean(np.abs(arrays["action"]) >= 1)),
            "max_abs_raw_action": float(np.max(np.abs(arrays["action"]))),
            "mean_reward": float(arrays["reward"].mean()),
            "mean_weighted_terms_before_dt": {k: float(v.mean()) for k, v in arrays.items() if k.startswith("reward/")},
        }
        print(controller, json.dumps(rows[controller]), flush=True)
    result = {"kind": "same_MJX_reward_diagnostic", "command": args.command,
        "run_dir": str(args.run_dir), "checkpoint": args.checkpoint,
        "policy_label": args.policy_label,
        "phase_frequency_hz": float(np.asarray(state.info["phase_dt"])[0] / (2 * np.pi * env.dt)),
        "seed": args.seed, "reset_randomized": args.randomized_reset,
        "evaluation_reset": args.evaluation_reset, "initial_states": initial_states,
        "reset_ablation": args.reset_ablation,
        "reset_config": OmegaConf.to_container(cfg.sim.reset, resolve=True),
        "results": rows,
        "oracle_sha256": oracle.EXPECTED_ONNX_SHA256, "git": ev._git_record(),
        "wall_time_seconds": time.perf_counter() - start}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
