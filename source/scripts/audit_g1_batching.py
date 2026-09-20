"""Compare serial and batched numeric evaluation on short real MJX rollouts."""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from source.scripts import evaluate_g1 as ev
from brax.io import model
from brax.training.agents.ppo import networks
import numpy as np
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(args.run_dir / "resolved_config.json")
    cfg.robot.fetch_model = cfg.sim.playground.fetch_source = False
    cfg.sim.reset.randomize = cfg.sim.noise.add_noise = cfg.sim.push.add_push = False
    robot = ev.make_robot(cfg.robot)
    env = ev.get_env_class(cfg.env.name)(cfg.robot.name, robot, cfg.env.terrain, cfg.sim)
    factory = functools.partial(networks.make_ppo_networks,
        policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=cfg.agent.value_hidden_layer_sizes,
        policy_obs_key="state", value_obs_key="privileged_state")
    net = ev._make_evaluation_network(factory, env.observation_size, env.action_size,
        normalize_observations=bool(cfg.agent.normalize_observations))
    params = model.load_params(args.run_dir / "logs/checkpoints" / str(args.checkpoint) / "policy")
    commands = tuple(x for x in ev.COMMANDS if x[0] in {"stand", "forward", "left", "turn_left"})
    traces, times = {}, {}
    for batch_size in (1, 4):
        start = time.perf_counter()
        rollout = ev._build_rollout(env, net, 50, batch_size=batch_size)
        traces[batch_size] = list(ev._numeric_episode_traces(
            rollout, params, True, commands, [4000], batch_size))
        times[batch_size] = time.perf_counter() - start
        print(f"batch={batch_size}, seconds including compile={times[batch_size]:.2f}", flush=True)
    # Fixed, prospective tolerance: 0.01 in physical SI/error/reward channels,
    # identical termination masks, at most two contact differences per 50 steps.
    tolerances = {key: .01 for key in ("linear_error", "yaw_error", "pelvis_height", "reward")}
    results = []
    passed = True
    for (name, _), serial, batch in zip(commands, traces[1], traces[4]):
        differences = {key: float(np.max(np.abs(serial[key] - batch[key]))) for key in tolerances}
        masks_equal = all(np.array_equal(serial[k], batch[k]) for k in ("valid", "done"))
        contact_differences = {k: int(np.count_nonzero(serial[k] != batch[k]))
                               for k in ("left_contact", "right_contact")}
        finite = all(np.isfinite(v).all() for t in (serial, batch) for v in t.values())
        ok = (finite and masks_equal and all(differences[k] <= v for k, v in tolerances.items())
              and max(contact_differences.values()) <= 2)
        passed &= ok
        results.append(dict(command=name, max_abs_difference=differences,
                            masks_equal=masks_equal, contact_differences=contact_differences,
                            finite=finite, passed=ok))
    result = dict(kind="short_MJX_numeric_batch_audit", run_dir=str(args.run_dir),
                  checkpoint=args.checkpoint, steps=50, seed=4000, batch_size=4,
                  tolerances=tolerances, results=results, passed=passed,
                  seconds_including_compile=times, git=ev._git_record())
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("Serial/batched physics comparison failed; keep serial research evaluation")


if __name__ == "__main__":
    main()
