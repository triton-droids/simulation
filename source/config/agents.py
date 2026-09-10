"""Agent hyperparameter defaults for the existing Brax PPO training path."""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class PPOConfig:
    """Default Brax PPO hyperparameters for locomotion training.

    The training script reads this dataclass through Hydra and passes the
    values into Brax's PPO trainer. It represents the advanced RL workflow, not
    the beginner Unitree G1 viewer path.
    """

    name: str = "ppo"
    num_timesteps: int = 200_000_000
    num_evals: int = 100
    episode_length: int = 1000
    unroll_length: int = 20
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    discounting: float = 0.97
    learning_rate: float = 3e-4
    entropy_cost: float = 0.01
    clipping_epsilon: float = 0.2
    num_envs: int = 8192
    batch_size: int = 256
    seed: int = 0
    render_interval: int = 50
    normalize_observations: bool = True
    action_repeat: float = 1.0
    max_grad_norm: float = 1.0
    policy_hidden_layer_sizes: Tuple[int, ...] = (512, 256, 128)
    value_hidden_layer_sizes: Tuple[int, ...] = (512, 256, 128)
    num_resets_per_eval: int = 1
    num_eval_envs: int = 128
    deterministic_eval: bool = True
    reward_scaling: float = 1.0
    gae_lambda: float = 0.95
    # Brax 0.14.2's single-device reset branch passes the environment batch
    # straight through a jitted reset. Its default pmap branch correctly
    # preserves the device and environment axes, including on one device.
    use_pmap_on_reset: bool = True


@dataclass
class G1PPOSmokeConfig(PPOConfig):
    """Minimum profile that must checkpoint before any longer G1 run."""

    name: str = "ppo_g1_smoke"
    num_timesteps: int = 1024
    num_evals: int = 2
    episode_length: int = 64
    unroll_length: int = 4
    num_minibatches: int = 4
    num_updates_per_batch: int = 1
    discounting: float = 0.98
    learning_rate: float = 1e-4
    entropy_cost: float = 0.0
    num_envs: int = 16
    num_eval_envs: int = 4
    batch_size: int = 4
    reward_scaling: float = 0.1
    policy_hidden_layer_sizes: Tuple[int, ...] = (32, 32)
    value_hidden_layer_sizes: Tuple[int, ...] = (32, 32)


@dataclass
class G1PPOConfig(PPOConfig):
    """Historical profile used by the preserved failed Gate 4 family."""

    name: str = "ppo_g1"
    num_timesteps: int = 5_000_000
    num_evals: int = 11
    episode_length: int = 1000
    unroll_length: int = 32
    num_minibatches: int = 16
    num_updates_per_batch: int = 2
    discounting: float = 0.98
    learning_rate: float = 5e-5
    entropy_cost: float = 0.0
    num_envs: int = 512
    num_eval_envs: int = 32
    batch_size: int = 32
    reward_scaling: float = 0.1
    policy_hidden_layer_sizes: Tuple[int, ...] = (512, 256, 64)
    value_hidden_layer_sizes: Tuple[int, ...] = (256, 256, 256, 256)


@dataclass
class G1PPOCorrectiveConfig(PPOConfig):
    """Diagnostic-first G1 profile aligned to pinned Playground PPO semantics.

    The 512-environment laptop batch keeps exactly one 20-step rollout per
    epoch (``32 * 16 == 512``), while retaining Playground's optimizer,
    entropy, discount, update count, reward scale, and asymmetric networks.
    Timesteps and evaluation count remain deliberately diagnostic-sized and
    are overridden only after short-run behavior justifies scaling.
    """

    name: str = "ppo_g1_corrective"
    num_timesteps: int = 262_144
    num_evals: int = 5
    episode_length: int = 1000
    unroll_length: int = 20
    num_minibatches: int = 16
    num_updates_per_batch: int = 4
    discounting: float = 0.97
    learning_rate: float = 3e-4
    entropy_cost: float = 0.005
    clipping_epsilon: float = 0.2
    num_envs: int = 512
    num_eval_envs: int = 32
    batch_size: int = 32
    reward_scaling: float = 1.0
    policy_hidden_layer_sizes: Tuple[int, ...] = (512, 256, 128)
    value_hidden_layer_sizes: Tuple[int, ...] = (512, 256, 128)
