# PPO Reference

Rewards define the behavior to prefer. PPO values define how experience is
collected and how aggressively the neural network learns from it.

## Profiles

| Value | Smoke | Historical G1 | Corrective diagnostic |
|---|---:|---:|---:|
| Timesteps | 1,024 | 5,000,000 | 262,144 |
| Environments | 16 | 512 | 512 |
| Evaluation environments | 4 | 32 | 32 |
| Episode length | 64 | 1,000 | 1,000 |
| Rollout length | 4 | 32 | 20 |
| Minibatches | 4 | 16 | 16 |
| Updates per batch | 1 | 2 | 4 |
| Batch size | 4 | 32 | 32 |
| Learning rate | 0.0001 | 0.00005 | 0.0003 |
| Discount | 0.98 | 0.98 | 0.97 |
| Entropy cost | 0 | 0 | 0.005 |
| Reward scaling | 0.1 | 0.1 | 1.0 |
| Policy network | 32, 32 | 512, 256, 64 | 512, 256, 128 |
| Value network | 32, 32 | 256, 256, 256, 256 | 512, 256, 128 |

## What each value does

| Setting | Plain-English meaning | Higher usually does | Lower usually does |
|---|---|---|---|
| `num_timesteps` | Total experience budget | Trains longer, costs more | Ends sooner, may undertrain |
| `num_envs` | Parallel robot copies | More throughput and memory use | Less memory, slower collection |
| `episode_length` | Maximum steps before reset | Allows longer behaviors | Resets more often |
| `unroll_length` | Consecutive rollout steps | Longer temporal chunks | Shorter chunks |
| `num_updates_per_batch` | Reuses collected data | More learning, possible overfit | More conservative updates |
| `learning_rate` | Optimizer step size | Faster but less stable | Slower but steadier |
| `discounting` | Importance of future reward | More long-term focus | More immediate focus |
| `entropy_cost` | Exploration pressure | More action exploration | More deterministic actions |
| `clipping_epsilon` | PPO policy-change region | Permits larger updates | Restricts updates |
| `reward_scaling` | Reward multiplier for PPO | Enlarges learning signal | Shrinks learning signal |
| `gae_lambda` | Advantage bias/variance balance | Longer, noisier credit | Shorter, more biased credit |
| `max_grad_norm` | Gradient clipping threshold | Allows larger gradients | Clips more strongly |
| `policy_hidden_layer_sizes` | Policy network shape; each tuple item is one hidden layer's neuron count | More/larger entries add policy capacity and compute | Fewer/smaller entries reduce policy capacity and compute |
| `value_hidden_layer_sizes` | Value network shape; each tuple item is one hidden layer's neuron count | More/larger entries add critic capacity and compute | Fewer/smaller entries reduce critic capacity and compute |
| `normalize_observations` | Normalize network inputs | Stabilizes unlike-scaled inputs | Sends raw inputs to network |

The generic library profile is 200 million timesteps and 8,192 environments.
It is not the profile selected by the public training script.

For example, `(512, 256, 64)` means three hidden layers containing 512, 256,
and 64 neurons in that order. These values are network dimensions, not three
alternative settings and not a minimum/maximum range.
