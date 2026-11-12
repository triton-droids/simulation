"""Frozen-policy PPO gradient compatibility diagnostic, never training/validation."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def policy_gradient(net, normalizer, actor, critic, data, key, discount, gae, clip):
    import jax
    from jax.flatten_util import ravel_pytree
    from brax.training.agents.ppo import losses

    def objective(weights):
        _, metrics = losses.compute_ppo_loss(
            losses.PPONetworkParams(policy=weights, value=critic), normalizer,
            data, key, net, entropy_cost=0., discounting=discount,
            reward_scaling=1., gae_lambda=gae, clipping_epsilon=clip,
            normalize_advantage=True)
        return metrics['policy_loss']
    loss, gradient = jax.value_and_grad(objective)(actor)
    return loss, ravel_pytree(gradient)[0]


def compatibility(a, b):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite gradient')
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if min(na, nb) <= 1e-12:
        raise ValueError('Degenerate gradient, no directional conclusion')
    return {'cosine': float(np.dot(a, b) / (na * nb)),
            'standing_norm': na, 'forward_norm': nb}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    import jax
    import jax.numpy as jp
    import numpy as np
    from brax.training import acting
    from omegaconf import OmegaConf
    from robocub.scripts import evaluate_g1 as ev
    from robocub.scripts.audit_g1_critic_resets import load_saved_critic

    args.output_dir.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(args.run_dir / 'resolved_config.json')
    cfg.robot.fetch_model = False
    cfg.sim.reset.randomize = False
    path = args.run_dir / 'logs/checkpoints/20070400/policy'
    normalizer, actor, critic = load_saved_critic(path)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    results, gradients = {}, {}
    configs = {}
    for label, command in [('stand', [0., 0., 0.]), ('forward', [.45, 0., 0.])]:
        env = ev.get_env_class(cfg.env.name)(cfg.robot.name, ev.make_robot(cfg.robot), cfg.env.terrain, cfg.sim)
        # Diagnostic command stratification only; original physics/rewards intact.
        env._env.sample_command = lambda key, cmd=command: jp.asarray(cmd)
        configs[label] = {'command_override': command, 'effective': env.effective_config}
        factory = functools.partial(ev.ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=cfg.agent.value_hidden_layer_sizes,
            policy_obs_key='state', value_obs_key='privileged_state')
        net = ev._make_evaluation_network(factory, env.observation_size, env.action_size,
            normalize_observations=bool(cfg.agent.normalize_observations))
        policy = ev.ppo_networks.make_inference_fn(net)((normalizer, actor, critic), deterministic=False)
        wrapped = env.brax_training_wrapper(env, episode_length=500, action_repeat=1)

        @jax.jit
        def collect(key):
            reset_key, burn_key, rollout_key = jax.random.split(key, 3)
            state = wrapped.reset(jax.random.split(reset_key, 64))
            state, burn = acting.generate_unroll(wrapped, state, policy, burn_key, 100,
                extra_fields=('truncation',))
            _, batch = acting.generate_unroll(wrapped, state, policy, rollout_key, 20,
                extra_fields=('truncation',))
            return jax.tree.map(lambda x: jp.swapaxes(x, 0, 1), batch), jp.sum(1 - burn.discount)

        grad_fn = jax.jit(functools.partial(policy_gradient, net, normalizer, actor, critic,
            discount=float(cfg.agent.discounting), gae=float(cfg.agent.gae_lambda),
            clip=float(cfg.agent.clipping_epsilon)))
        for draw in [17, 29]:
            batch, burn_terminals = collect(jax.random.PRNGKey(draw))
            if not np.allclose(np.asarray(batch.observation['state'])[..., 9:12], command, rtol=0, atol=1e-7):
                raise ValueError('Command stratification failed')
            loss, grad = grad_fn(data=batch, key=jax.random.PRNGKey(draw + 1000))
            grad = np.asarray(grad)
            tag = f'{label}_{draw}'
            gradients[tag] = grad
            np.savez_compressed(args.output_dir / (tag + '_batch.npz'),
                state=np.asarray(batch.observation['state']),
                privileged_state=np.asarray(batch.observation['privileged_state']),
                next_state=np.asarray(batch.next_observation['state']),
                next_privileged_state=np.asarray(batch.next_observation['privileged_state']),
                action=np.asarray(batch.action), reward=np.asarray(batch.reward),
                discount=np.asarray(batch.discount),
                **{k: np.asarray(v) for k, v in batch.extras['policy_extras'].items()},
                truncation=np.asarray(batch.extras['state_extras']['truncation']))
            results[tag] = {'policy_loss': float(loss),
                'burnin_terminations': int(burn_terminals),
                'batch_terminations': int(np.asarray(1 - batch.discount).sum()),
                'transitions': 1280}
            if not np.isfinite(float(loss)):
                raise ValueError('Nonfinite loss')
    pairs = {str(draw): compatibility(gradients[f'stand_{draw}'], gradients[f'forward_{draw}']) for draw in [17, 29]}
    if hashlib.sha256(path.read_bytes()).hexdigest() != original_hash:
        raise ValueError('Source checkpoint changed')
    np.savez_compressed(args.output_dir / 'gradients.npz', **gradients)
    summary = {'kind': 'frozen_actor_gradient_compatibility', 'optimizer_steps': 0,
        'full_validation': False, 'checkpoint_sha256': original_hash,
        'full_checkpoint_actor_normalizer_exact': True,
        'pairs': pairs, 'batches': results, 'effective_configs': configs,
        'limitation': 'Stochastic nominal command-stratified diagnostic batches, frozen critic/normalizer, policy loss only. Gradient cosine is a local noisy estimate, not proof of an architectural defect or inability to learn jointly. Burn-in does not guarantee settled gait, especially after autoresets.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
