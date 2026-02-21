"""Script to run a trained policy in simulation."""

import argparse

parser = argparse.ArgumentParser(description="Run a trained policy in MuJoCo or MJX simulation.")
parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing the trained policy.")
parser.add_argument("--checkpoint", type=int, required=True, help="Checkpoint step to load.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to run the policy.")
parser.add_argument("--save_video", action="store_true", default=False, help="Save a video of the rollout.")
parser.add_argument("--output", type=str, default=None, help="Output path for the saved video (without extension).")

args = parser.parse_args()

import functools
from brax.training.agents.ppo import networks as ppo_networks
from etils import epath
from omegaconf import OmegaConf
from src.locomotion import get_env_class
from src.robots.robot import Robot
from src.tools.rollouts import get_rollout, save_rollout


def main():
    run_dir = epath.Path(args.run_dir).resolve()
    policy_path = run_dir / "logs" / "checkpoints" / f"{args.checkpoint}" / "policy"
    config_dir = str(run_dir / ".hydra" / "config.yaml")

    try:
        config = OmegaConf.load(config_dir)
    except FileNotFoundError as e:
        print(f"Error: Could not find required file: {e}")
        print(f"Please check that the run_dir '{args.run_dir}' and checkpoint files exist.")
        return

    env_cfg = config.sim
    agent_cfg = config.agent

    EnvClass = get_env_class(config.env.name)
    robot = Robot(config.robot.name)
    env = EnvClass(
        config.robot.name,
        robot,
        config.env.terrain,
        env_cfg,
    )

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=agent_cfg.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=agent_cfg.value_hidden_layer_sizes,
    )

    if args.save_video:
        output_path = args.output if args.output else str(run_dir / f"rollout_{args.checkpoint}")
        print(f"Running policy for {args.num_steps} steps and saving video to '{output_path}'...")
        save_rollout(output_path, str(policy_path), env, make_networks_factory, args.num_steps)
        print("Done.")
    else:
        print(f"Running policy for {args.num_steps} steps...")
        rollout = get_rollout(str(policy_path), env, make_networks_factory, args.num_steps)
        print(f"Rollout complete. Total frames: {len(rollout)}")


if __name__ == "__main__":
    main()
