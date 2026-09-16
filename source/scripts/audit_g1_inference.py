"""Check saved PPO actions against the pinned Brax training construction."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from source.scripts import evaluate_g1 as evaluation
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks
import jax
import jax.numpy as jp
import numpy as np
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.run_dir / "resolved_config.json")
    params = model.load_params(args.run_dir / "logs/checkpoints" / str(args.checkpoint) / "policy")
    factory = functools.partial(
        networks.make_ppo_networks,
        policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=cfg.agent.value_hidden_layer_sizes,
        policy_obs_key="state", value_obs_key="privileged_state",
    )
    sizes = {"state": 103, "privileged_state": 216}
    # Matches Brax 0.14.2 train.py:432-450 and repository train.py's factory.
    normalize = running_statistics.normalize if cfg.agent.normalize_observations else lambda x, _: x
    training_net = factory(sizes, 29, preprocess_observations_fn=normalize)
    eval_net = evaluation._make_evaluation_network(
        factory, sizes, 29, normalize_observations=cfg.agent.normalize_observations,
    )
    wrong_net = factory(sizes, 29)
    policies = [networks.make_inference_fn(net)(params, deterministic=True)
                for net in (training_net, eval_net, wrong_net)]
    errors, sensitivity = [], []
    key = jax.random.PRNGKey(707)
    # Nontrivial batches exercise every input, command and running statistic.
    for scale in (0.0, 0.1, 1.0, 3.0):
        key, obs_key, action_key = jax.random.split(key, 3)
        obs = {name: scale * jax.random.normal(obs_key, (16, size)) for name, size in sizes.items()}
        actions = [np.asarray(policy(obs, action_key)[0]) for policy in policies]
        assert all(np.isfinite(action).all() for action in actions)
        errors.append(float(np.max(np.abs(actions[0] - actions[1]))))
        sensitivity.append(float(np.max(np.abs(actions[0] - actions[2]))))
    result = {"checkpoint": str(args.run_dir / "logs/checkpoints" / str(args.checkpoint)),
              "max_action_difference": max(errors), "identity_preprocessor_difference": max(sensitivity),
              "observations_checked": 64, "git": evaluation._git_record()}
    assert max(errors) < 1e-6, result
    if cfg.agent.normalize_observations:
        assert max(sensitivity) > 1e-3, "Audit inputs must detect the historical normalization bug"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
