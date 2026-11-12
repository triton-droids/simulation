"""Zero-training reset-only audit of normalization-induced policy KL.

This is not the first PPO rollout batch and cannot explain its epoch mean KL.
"""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def compare_normalizers(network, normalizer, policy, observations):
    """Keep weights fixed; compare identity and one running-statistics update."""
    import jax.numpy as jp
    from brax.training.acme import running_statistics

    updated = running_statistics.update(normalizer, observations)
    dist = network.parametric_action_distribution
    old_logits = network.policy_network.apply(normalizer, policy, observations)
    identity_logits = network.policy_network.apply(normalizer, policy, observations)
    new_logits = network.policy_network.apply(updated, policy, observations)
    old = dist.create_dist(old_logits)
    identity = dist.create_dist(identity_logits)
    new = dist.create_dist(new_logits)
    return {
        'identity_kl': jp.mean(identity.kl_divergence(old)),
        'identity_expected_floor': old.loc.shape[-1] * jp.log(jp.asarray(1.0 + 1e-5, dtype=old.loc.dtype)),
        'identity_logits_max_difference': jp.max(jp.abs(identity_logits - old_logits)),
        'normalizer_only_kl': jp.mean(new.kl_divergence(old)),
        'mean_location_change_rms': jp.sqrt(jp.mean((new.loc - old.loc) ** 2)),
        'mean_scale_change_rms': jp.sqrt(jp.mean((new.scale - old.scale) ** 2)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import jax
    import numpy as np
    from brax.io import model
    from omegaconf import OmegaConf
    from robocub.scripts import evaluate_g1 as ev

    cfg = OmegaConf.load(args.run_dir / 'resolved_config.json')
    assert cfg.agent.normalize_observations
    assert cfg.sim.reset.randomize
    assert cfg.sim.playground.recovery_reset_candidates == 1
    assert cfg.sim.playground.reset_disturbance_scale == 1.0
    cfg.robot.fetch_model = False
    env = ev.get_env_class(cfg.env.name)(cfg.robot.name, ev.make_robot(cfg.robot), cfg.env.terrain, cfg.sim)
    factory = functools.partial(ev.ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=cfg.agent.value_hidden_layer_sizes,
        policy_obs_key='state', value_obs_key='privileged_state')
    net = ev._make_evaluation_network(factory, env.observation_size, env.action_size,
                                      normalize_observations=True)
    path = args.run_dir / 'logs/checkpoints/0/policy'
    normalizer, policy = model.load_params(path)
    if int(normalizer.count.hi) != 0 or int(normalizer.count.lo) != 0:
        raise ValueError('Requires untouched checkpoint0 normalizer')
    # Match the single-process trainer's initial reset key derivation.
    _, local_key = jax.random.split(jax.random.PRNGKey(args.seed))
    local_key = jax.random.fold_in(local_key, 0)
    _, key_env, _ = jax.random.split(local_key, 3)
    keys = jax.random.split(key_env, 2048)
    observations = jax.jit(jax.vmap(lambda key: env.reset(key).obs))(keys)
    metrics = jax.jit(functools.partial(compare_normalizers, net))(normalizer, policy, observations)
    metrics = {k: float(v) for k, v in metrics.items()}
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError('Nonfinite diagnostic')
    # Brax adds 1e-5 inside log(scale ratio), so identical distributions
    # have a small positive floor, not exactly zero.
    if (metrics['identity_logits_max_difference'] != 0 or
            abs(metrics['identity_kl'] - metrics['identity_expected_floor']) > 1e-5):
        raise ValueError('Identity control failed')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / 'observations.npz',
                        **{k: np.asarray(v) for k, v in observations.items()})
    report = {'kind': 'reset_only_normalizer_shift', 'training_steps': 0,
        'episodes': 0, 'reset_count': 2048, 'seed': args.seed,
        'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'metrics': metrics, 'effective_config': OmegaConf.to_container(cfg, resolve=True),
        'full_validation': False,
        'limitation': 'Reset observations only, not a PPO rollout batch. No optimizer steps. Cannot attribute epoch KL or behavior failures to normalization. Positive result permits a separately frozen first-batch diagnostic, not training or normalization changes.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
