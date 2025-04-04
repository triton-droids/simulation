"""Train a PPO agent using JAX on the specified environment."""
from datetime import datetime
from absl import logging
from ml_collections import config_dict
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from orbax import checkpoint as ocp
from flax.training import orbax_utils
from etils import epath
from tensorboardX import SummaryWriter
from mujoco_playground._src import mjx_env
from mujoco_playground import wrapper
from ..utils import registry
from ..utils.rollouts import save_mjx_rollout
import jax.numpy as jp
import functools
import json
import time
import wandb
import warnings

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

def brax_train_policy(
    env_name: str,
    cfg: config_dict.ConfigDict,
    checkpoint: str = None,
    exp_name: str = None,
    record: bool = False,
) -> None:

    env = registry.load(env_name, cfg.env)
    print(env)
    env_cfg = cfg.env
    ppo_cfg = cfg.brax_ppo_agent
    print(ppo_cfg)
    print(env_cfg)

    # Generate unique experiment name
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{env_name}-{timestamp}"
    if exp_name is not None:
        experiment_name += f"-{exp_name}"
    print(f"Experiment name: {exp_name}")

    # Set up logging directory
    logdir = epath.Path("logs").resolve() / experiment_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(logdir)
    print(f"Logs are being stored in: {logdir}")

    # Initialize Weights & Biases if required
    if cfg.logger.wandb:
        wandb.init(project=cfg.logger.project_name, name=exp_name)
        wandb.config.update(env_cfg.to_dict())
        wandb.config.update({"env_name": env_cfg.name})

    # Initialize TensorBoard if required
    if cfg.logger.tensorboard:
        writer = SummaryWriter(logdir)

    # Handle checkpoint loading
    if checkpoint is not None:
        # Convert to absolute path
        checkpoint = epath.Path(checkpoint).resolve()
        print(f"Restoring from checkpoint: {checkpoint}")
    else:
        print("No checkpoint path provided, not restoring from checkpoint")
        restore_checkpoint_path = None

    # Set up checkpoint directory
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    if record:
        # Set up training vid directory
        vid_path = logdir / "videos"
        vid_path.mkdir(parents=True, exist_ok=True)
        print(f"Video path: {vid_path}")

        # Load recording environment
        rec_env = registry.load(env_name, env_cfg)

    # Save environment configuration
    with open(ckpt_path / "config.json", "w") as fp:
        json.dump(cfg.to_dict(), fp, indent=2)

    # Define policy parameters function for saving checkpoints and recording rollouts
    def policy_params_fn(current_step, make_policy, params):
        inference_fn = make_policy(params)
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f"{current_step}"
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        if record:
            path = vid_path / f"{current_step}"
            print("Saving rollout at step {current_step} to videos/")
            save_mjx_rollout(rec_env, inference_fn, path, seed=cfg.seed)

    training_params = dict(ppo_cfg)
    if "network_factory" in training_params:
        del training_params["network_factory"]

    network_fn = ppo_networks.make_ppo_networks
    network_factory = functools.partial(network_fn, **ppo_cfg.network_factory)

    if env_cfg.d_randomization:
        training_params["randomization_fn"] = registry.get_domain_randomizer(env_name)
    num_eval_envs = ppo_cfg.num_envs

    if "num_eval_envs" in training_params:
        del training_params["num_eval_envs"]

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        policy_params_fn=policy_params_fn,
        seed=cfg.seed,
        restore_checkpoint_path=restore_checkpoint_path,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
    )

    times = [time.monotonic()]

    # Progress function for logging
    def progress(num_steps, metrics):
        times.append(time.monotonic())

        # Log to Weights & Biases
        if cfg.logger.wandb:
            wandb.log(metrics, step=num_steps)

        # Log to TensorBoard
        if cfg.logger.tensorboard:
            for key, value in metrics.items():
                writer.add_scalar(key, value, num_steps)
            writer.flush()

        print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")

    # Load evaluation environment
    eval_env = registry.load(env_name, env_cfg)

    # Train or load the model
    make_inference_fn, params, _ = train_fn(
        environment=env, progress_fn=progress, eval_env=eval_env
    )

    print("Done training.")
    if len(times) > 1:
        print(f"Time to JIT compile: {times[1] - times[0]}")
        print(f"Time to train: {times[-1] - times[1]}")

    return make_inference_fn
