from dataclasses import dataclass
from typing import Tuple

@dataclass
class PPOConfig:
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