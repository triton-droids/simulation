"""Train the existing default humanoid locomotion policy with Brax PPO.

This remains the advanced training entry point for the simulator internals
under `source/`. It is intentionally separate from the beginner Unitree G1
loader in `scripts/` because training requires the MJX/Brax stack, logging,
Hydra configuration, and policy checkpoint handling.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

ORIGINAL_ARGV = tuple(sys.argv)

# JAX reads allocator and XLA settings during backend initialization, so set
# them before the first JAX import. This matters on the 8 GiB target GPU.
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "glfw" if sys.platform == "win32" else "egl"
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / ".cache" / "jax_compilation_cache"),
)

# Make `source.*` imports work when this file is launched directly from an IDE
# or with `python source/scripts/train.py` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.scripts import cli_args
import jax
from source.utils.jax_compat import install_brax_pmap_compatibility

install_brax_pmap_compatibility()


# Collect the command-line flags that should be handled before Hydra reads the
# remaining config overrides.
parser = argparse.ArgumentParser(description="Train a RL agent with Brax")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--logger",
    choices=("local", "none", "wandb-offline", "wandb"),
    default="local",
    help="Metrics backend. The default writes JSONL locally and needs no credentials.",
)

cli_args.add_rl_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Leave only Hydra overrides in sys.argv so Hydra does not try to parse the
# training script's ordinary argparse flags.
sys.argv = [sys.argv[0]] + hydra_args

from source.locomotion import get_env_class

import time
import json
import hydra
import warnings
import functools
from typing import Any
from etils import epath
from absl import logging

from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from omegaconf import OmegaConf, DictConfig
from orbax import checkpoint as ocp
from flax.training import orbax_utils

from source.config.config import Config 
from source.tools.rollouts import save_rollout
from source.robots import make_robot
from source.randomize import domain_randomize
from source.utils.experiment_logging import create_metric_logger
from source.utils.checkpoints import inference_params_from_training_params
from source.utils.ppo_config import trainer_runtime_controls

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)
import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")
# Supress Hydra warnings
warnings.filterwarnings("ignore", category=UserWarning)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_record() -> dict[str, object]:
    """Return semantic Git state without WSL/Windows CRLF false positives."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "core.autocrlf=true", *args],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    try:
        return {
            "available": True,
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"available": False, "commit": None, "branch": None, "dirty": None}


def _gpu_record() -> list[str]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _runtime_manifest() -> dict[str, object]:
    packages = (
        "brax",
        "etils",
        "flax",
        "hydra-core",
        "jax",
        "jaxlib",
        "ml-collections",
        "mujoco",
        "omegaconf",
        "orbax-checkpoint",
    )
    return {
        "started_at_utc": _utc_timestamp(),
        "command": list(ORIGINAL_ARGV),
        "project_root": str(PROJECT_ROOT),
        "git": _git_record(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "gpu": _gpu_record(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "environment": {
            name: os.environ.get(name)
            for name in (
                "JAX_COMPILATION_CACHE_DIR",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "MUJOCO_GL",
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
            )
        },
    }


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Train or parameter-warm-start a PPO policy for the configured task.

    Args:
        cfg: Hydra configuration containing robot, environment, simulator, and
            agent settings.

    Side effects:
        Creates local `logs/` and optional `results/` folders, writes copied
        training and robot configs, logs to Weights & Biases, and saves policy
        checkpoints. The pinned Brax restore interface reloads normalizer,
        policy, and value parameters but not optimizer/step/PRNG state. W&B is
        used only when explicitly selected with
        ``--logger wandb`` or ``--logger wandb-offline``.

    Failure cases:
        Missing dependencies, invalid robot/config names, unavailable GPU/JAX
        backends, or missing warm-start checkpoints will stop training before a
        policy is produced.
    """

    print("=" * 100)
    checkpoint_path = None
    robot_xml_path = None
    robot_config_path = None
    if args_cli.resume:
        checkpoint_path = epath.Path(args_cli.checkpoint).resolve()
        print(f"Warm-starting parameters from checkpoint: {checkpoint_path}")

        run_dir = checkpoint_path
        while run_dir.name != "logs" and run_dir != run_dir.parent:
            run_dir = run_dir.parent
        run_dir = run_dir.parent
        
        if cfg.robot.name == "default_humanoid_legs":
            robot_config_path = run_dir / "robot_config.json"
            robot_xml_path = str(run_dir / (cfg.robot.name + ".xml"))

        print("Successfully loaded configuration from previous run")

    else:
        print("No checkpoint provided. Training from scratch.")

    robot = make_robot(
        cfg.robot,
        config_path=robot_config_path,
        xml_path=robot_xml_path,
    )

    EnvClass = get_env_class(cfg.env.name)
    env_cfg = cfg.sim
    train_cfg = cfg.agent
    if args_cli.seed is not None:
        train_cfg.seed = args_cli.seed
        cfg.seed = args_cli.seed

    
    env = EnvClass(
        cfg.robot.name,
        robot,
        cfg.env.terrain, 
        env_cfg)
    
    eval_env = EnvClass(
        cfg.robot.name,
        robot,
        cfg.env.terrain,
        env_cfg)
    
    test_env = EnvClass(
        cfg.robot.name,
        robot,
        cfg.env.terrain,
        env_cfg
    )

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=train_cfg.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=train_cfg.value_hidden_layer_sizes,
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

    time_str = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{robot.name}_{cfg.env.name}_{cfg.agent.name}_{time_str}"

    logdir = epath.Path("logs").resolve() 

    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logs are being stored to {logdir}")

    if args_cli.video:
        run_dir = logdir.parent
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)


    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    #Save environment configuration
    with open(logdir.parent / "train_config.json", "w") as f:
        json.dump(OmegaConf.to_container(train_cfg), f, indent=4)

    with open(logdir.parent / "resolved_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)

    #Save robot configuration
    with open(logdir.parent / "robot_config.json", "w") as f:
        json.dump(robot.model_config, f, indent=4)
    
    if hasattr(robot, "source_record"):
        # An include-based external model is reconstructed from provenance;
        # copying its top-level XML would not create a self-contained asset.
        with open(logdir.parent / "robot_source.json", "w") as f:
            json.dump(robot.source_record, f, indent=4)
    else:
        with open(logdir.parent / Path(robot.name + ".xml"), "w") as f:
            f.write(robot.xml)

    if hasattr(env, "source_record"):
        with open(logdir.parent / "environment_source.json", "w") as f:
            json.dump(env.source_record, f, indent=4)
    if hasattr(env, "effective_config"):
        with open(logdir.parent / "environment_effective_config.json", "w") as f:
            json.dump(env.effective_config, f, indent=4)

    manifest_path = Path(logdir.parent) / "run_manifest.json"
    run_manifest = _runtime_manifest()
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2, sort_keys=True)

    print("=" * 100)

    metric_logger = create_metric_logger(
        mode=args_cli.logger,
        output_dir=Path(logdir),
        run_name=run_name,
        project=args_cli.log_project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print("=" * 100)

    def policy_params_fn(current_step: int, make_policy: Any, params: Any):
        """Persist PPO checkpoints and policy parameters during training.

        Args:
            current_step: Training step reported by Brax PPO.
            make_policy: Brax policy factory passed by the PPO trainer.
            params: Current training parameters to checkpoint.

        Side effects:
            Writes checkpoint folders under `logs/checkpoints/`.
        """

        # Save both the Orbax checkpoint and the smaller policy params used by
        # playback so a policy can be inspected or parameter-warm-started.
        # Brax 0.14.2 does not restore optimizer, step, or PRNG state here.
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = os.path.abspath(os.path.join(ckpt_path, f"{current_step}"))        
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        policy_path = os.path.join(path, "policy")
        model.save_params(policy_path, inference_params_from_training_params(params))


    domain_randomize_fn = None
    if env.add_domain_rand:
        domain_randomize_fn = functools.partial(
            domain_randomize,
            robot=robot,
            friction_range=env_cfg.domain_rand.friction_range,
            frictionloss_range=env_cfg.domain_rand.frictionloss_range,
            armature_range=env_cfg.domain_rand.armature_range,
            body_mass_range=env_cfg.domain_rand.body_mass_range,
            torso_mass_range=env_cfg.domain_rand.torso_mass_range,
            qpos0_range=env_cfg.domain_rand.qpos0_range,
        )

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=train_cfg.num_timesteps,
        num_evals=train_cfg.num_evals,
        episode_length=train_cfg.episode_length,
        unroll_length=train_cfg.unroll_length,
        num_minibatches=train_cfg.num_minibatches,
        num_updates_per_batch=train_cfg.num_updates_per_batch,
        discounting=train_cfg.discounting,
        learning_rate=train_cfg.learning_rate,
        entropy_cost=train_cfg.entropy_cost,
        clipping_epsilon=train_cfg.clipping_epsilon,
        num_envs=train_cfg.num_envs,
        batch_size=train_cfg.batch_size,
        seed=train_cfg.seed,
        num_resets_per_eval=train_cfg.num_resets_per_eval,
        num_eval_envs=train_cfg.num_eval_envs,
        deterministic_eval=train_cfg.deterministic_eval,
        reward_scaling=train_cfg.reward_scaling,
        gae_lambda=train_cfg.gae_lambda,
        use_pmap_on_reset=train_cfg.use_pmap_on_reset,
        **trainer_runtime_controls(train_cfg),
        network_factory=make_networks_factory,
        randomization_fn=domain_randomize_fn,
        policy_params_fn=policy_params_fn,
        restore_checkpoint_path=checkpoint_path,
        wrap_env_fn=getattr(env, "brax_training_wrapper", None),
    )


    times = [time.time()]
    
    last_ckpt_step = 0
    best_ckpt_step = 0
    best_episode_reward = -float("inf")
    last_video_step = 0

    def progress(num_steps, metrics):
        """Handle PPO progress updates, metric logging, and optional videos.

        Args:
            num_steps: Current PPO training step.
            metrics: Evaluation and training metrics reported by Brax.

        Side effects:
            Logs metrics to the explicitly selected backend and may write
            rollout videos when video capture is enabled.
        """

        nonlocal best_episode_reward, best_ckpt_step, last_ckpt_step, last_video_step

        times.append(time.time())
        metric_logger.log(metrics, step=num_steps)

        if args_cli.video and last_ckpt_step != 0 and (num_steps - last_video_step >= args_cli.video_interval):
            print(f"Saving rollout at step {last_ckpt_step}")
            current_ckpt_path = os.path.join(logdir, "checkpoints")
            current_policy_path = os.path.join(current_ckpt_path, f"{last_ckpt_step}", "policy")
            save_path = str(results_dir / f"{last_ckpt_step}")
            save_rollout(save_path, current_policy_path, test_env, make_networks_factory, args_cli.video_length)
            last_video_step = last_ckpt_step

        last_ckpt_step = num_steps

        episode_reward = float(metrics.get("eval/episode_reward", 0))
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_ckpt_step = num_steps
        print(f"{num_steps}: {metrics['eval/episode_reward']}")
    
    try:
        try:
            make_policy, params, _ = train_fn(
                environment=env, eval_env=eval_env, progress_fn=progress
            )
        except KeyboardInterrupt:
            pass
    finally:
        metric_logger.close()

    if len(times) > 1:
        print(f"time to first progress callback: {times[1] - times[0]}")
        print(f"time after first callback: {times[-1] - times[1]}")
    else:
        print("No PPO progress callback was emitted.")
    print(f"best checkpoint step: {best_ckpt_step}")
    print(f"best episode reward: {best_episode_reward}")

    run_manifest.update(
        {
            "ended_at_utc": _utc_timestamp(),
            "wall_time_seconds": time.time() - times[0],
            "best_checkpoint_step": best_ckpt_step,
            "best_episode_reward": best_episode_reward,
        }
    )
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2, sort_keys=True)


    if args_cli.video and best_ckpt_step:
        print("Saving rollout for best checkpoint")
        best_ckpt_path = os.path.join(logdir.parent, f"best_policy-{best_ckpt_step}")
        best_policy_path = os.path.join(logdir, "checkpoints", f"{best_ckpt_step}", "policy")
        save_rollout(best_ckpt_path, best_policy_path, test_env, make_networks_factory, 1000)

if __name__ == "__main__":
    main()
