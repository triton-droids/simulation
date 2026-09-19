"""Prepare an explicit command-variance prior without fitting policy weights.

For a forward-only checkpoint, preserve its function on vy=yaw=0 while
preventing the newly varied command channels from normalizing to ~100,000.
Writes a separate full warm-start checkpoint and an inference parity audit.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from brax.io import model
from brax.training.agents.ppo import checkpoint, networks
from flax.training import orbax_utils
import jax
import jax.numpy as jp
import numpy as np
from omegaconf import OmegaConf
import orbax.checkpoint as ocp
from source.scripts import evaluate_g1 as ev
from source.utils.checkpoints import inference_params_from_training_params, set_unseen_g1_command_std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--lateral-std", type=float, required=True)
    parser.add_argument("--yaw-std", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.run_dir / "resolved_config.json")
    if cfg.env.name != "unitree_g1_playground" or not cfg.agent.normalize_observations:
        raise ValueError("Expected normalized pinned G1 Playground checkpoint")
    source = (args.run_dir / "logs/checkpoints" / str(args.checkpoint)).resolve()
    params = checkpoint.load(source)
    prepared = set_unseen_g1_command_std(params, args.lateral_std, args.yaw_std)
    factory = functools.partial(networks.make_ppo_networks,
        policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=cfg.agent.value_hidden_layer_sizes,
        policy_obs_key="state", value_obs_key="privileged_state")
    sizes = {key: len(value) for key,value in params[0].mean.items()}
    net = ev._make_evaluation_network(factory, sizes, 29, normalize_observations=True)
    rng = np.random.default_rng(1212)
    samples = {}
    for key, size in sizes.items():
        values = np.asarray(params[0].mean[key]) + rng.normal(size=(64,size))*np.asarray(params[0].std[key])
        values[:, 10:12] = 0
        samples[key] = jp.asarray(values, dtype=jp.float32)
    make_policy = networks.make_inference_fn(net)
    before = make_policy(inference_params_from_training_params(params), deterministic=True)(samples, jax.random.PRNGKey(0))[0]
    after = make_policy(inference_params_from_training_params(prepared), deterministic=True)(samples, jax.random.PRNGKey(0))[0]
    difference = float(np.max(np.abs(np.asarray(before)-np.asarray(after))))
    if difference != 0 or not np.isfinite(np.asarray(after)).all():
        raise ValueError(f"Preparation changed old-command inference: {difference}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    target = output / "checkpoint"
    ocp.PyTreeCheckpointer().save(str(target), prepared,
        save_args=orbax_utils.save_args_from_target(prepared))
    model.save_params(target / "policy", inference_params_from_training_params(prepared))
    restored = checkpoint.load(target)
    original_leaves = jax.tree_util.tree_leaves(prepared)
    restored_leaves = jax.tree_util.tree_leaves(restored)
    if (len(original_leaves) != len(restored_leaves)
            or any(np.shape(a) != np.shape(b) for a,b in zip(original_leaves, restored_leaves))):
        raise ValueError("Prepared checkpoint changed leaf count or shapes on reload")
    errors = [float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
              for a,b in zip(original_leaves, restored_leaves)]
    if max(errors) != 0:
        raise ValueError("Prepared checkpoint failed exact round-trip")
    shutil.copyfile(args.run_dir / "resolved_config.json", output / "source_resolved_config.json")
    report = {"kind": "explicit_unseen_command_variance_prior", "source": str(source),
        "prepared_checkpoint": str(target), "new_lateral_std": args.lateral_std,
        "new_yaw_std": args.yaw_std, "old_subspace": "vy=0,yaw=0",
        "synthetic_inputs": 64, "max_old_subspace_action_difference": difference,
        "max_full_checkpoint_round_trip_difference": max(errors),
        "weights_fitted": False, "actor_and_critic_weights_unchanged": True,
        "git": ev._git_record(), "command": sys.argv,
        "old_command_std": {k: np.asarray(v)[9:12].tolist() for k,v in params[0].std.items()},
        "new_command_std": {k: np.asarray(v)[9:12].tolist() for k,v in prepared[0].std.items()}}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
