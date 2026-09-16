"""Compare C06 stance and the exact ONNX oracle in the SAME MJX task.

Diagnostic only: no policy fitting or imitation. Saves all per-step terms,
actions, positions, contacts and velocities for a fixed forward command.
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
from brax.io import model
from brax.training.agents.ppo import networks
import jax
import jax.numpy as jp
import numpy as np
from omegaconf import OmegaConf
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    cfg = OmegaConf.load(args.run_dir / "resolved_config.json")
    cfg.robot.fetch_model = False
    cfg.sim.playground.fetch_source = False
    cfg.sim.reset.randomize = False
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
    command = jp.array([0.5, 0.0, 0.0])
    reset = jax.jit(env.reset)
    step = jax.jit(lambda state, action: ev._step_with_held_command(env, state, action, command))
    rows = {}
    for controller in ("C06", "oracle"):
        state = ev._replace_command(reset(jax.random.PRNGKey(2000)), command)
        info = dict(state.info)
        info["phase_dt"] = jp.asarray([2 * np.pi * 1.5 * env.dt])
        state = state.replace(info=info)
        trace = []
        for index in range(args.steps):
            if controller == "C06":
                action = policy(state.obs, jax.random.PRNGKey(index))[0]
            else:
                action = jp.asarray(session.run(None, {"obs": np.asarray(state.obs["state"])[None]})[0][0])
            state = step(state, action)
            sample = {"action": action, "qpos": state.data.qpos,
                "velocity": env.get_local_linvel(state.data, "pelvis"),
                "contact": env._contact(state.data), "phase": state.info["phase"],
                "reward": state.reward, "done": state.done, **state.metrics}
            trace.append(jax.tree.map(np.asarray, sample))
            if bool(state.done):
                break
        arrays = {name: np.stack([item[name] for item in trace]) for name in trace[0]}
        np.savez_compressed(args.output_dir / f"{controller}_trace.npz", **arrays)
        contact = arrays["contact"]
        rows[controller] = {
            "steps": len(trace), "terminal": bool(arrays["done"][-1]),
            "mean_velocity": arrays["velocity"].mean(0).tolist(),
            "linear_rmse": float(np.sqrt(np.mean(np.sum((arrays["velocity"][:, :2] - [0.5, 0]) ** 2, axis=1)))),
            "minimum_pelvis_height": float(arrays["qpos"][:, 2].min()),
            "single_support_fraction": float(np.mean(contact.sum(1) == 1)),
            "foot_transitions": np.sum(contact[1:] != contact[:-1], axis=0).tolist(),
            "raw_action_saturation_fraction": float(np.mean(np.abs(arrays["action"]) >= 1)),
            "max_abs_raw_action": float(np.max(np.abs(arrays["action"]))),
            "mean_reward": float(arrays["reward"].mean()),
            "mean_weighted_terms_before_dt": {k: float(v.mean()) for k, v in arrays.items() if k.startswith("reward/")},
        }
        print(controller, json.dumps(rows[controller]), flush=True)
    result = {"kind": "same_MJX_reward_diagnostic", "command": [0.5, 0, 0],
        "phase_frequency_hz": 1.5, "seed": 2000, "results": rows,
        "oracle_sha256": oracle.EXPECTED_ONNX_SHA256, "git": ev._git_record(),
        "wall_time_seconds": time.perf_counter() - start}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
