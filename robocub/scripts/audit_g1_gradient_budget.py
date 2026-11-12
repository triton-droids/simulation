"""Replay saved diagnostic batches to measure actor/critic gradient clipping budget."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def gradient_budget(network, params, normalizer, batch, key, config):
    import jax
    import jax.numpy as jp
    from brax.training.agents.ppo import losses

    def objective(weights):
        return losses.compute_ppo_loss(weights, normalizer, batch, key, network,
            entropy_cost=config['entropy_cost'], discounting=config['discounting'],
            reward_scaling=config['reward_scaling'], gae_lambda=config['gae_lambda'],
            clipping_epsilon=config['clipping_epsilon'], normalize_advantage=True)
    (loss, metrics), gradient = jax.value_and_grad(objective, has_aux=True)(params)
    squared = lambda tree: sum(jp.sum(x*x) for x in jax.tree.leaves(tree))
    actor_sq, critic_sq = squared(gradient.policy), squared(gradient.value)
    total_norm = jp.sqrt(actor_sq + critic_sq)
    return {'loss': loss, 'actor_norm': jp.sqrt(actor_sq),
        'critic_norm': jp.sqrt(critic_sq), 'total_norm': total_norm,
        'critic_actor_ratio': jp.sqrt(critic_sq) / jp.maximum(jp.sqrt(actor_sq), 1e-12),
        'shared_clip_multiplier': jp.minimum(1., config['max_grad_norm'] / jp.maximum(total_norm, 1e-12)),
        'policy_loss': metrics['policy_loss'], 'value_loss': metrics['v_loss'],
        'entropy_loss': metrics['entropy_loss']}


def load_batch(path):
    import numpy as np
    import jax.numpy as jp
    from brax.training.types import Transition
    with np.load(path) as saved:
        d = {k: jp.asarray(saved[k]) for k in saved.files}
    return Transition(observation={'state':d['state'],'privileged_state':d['privileged_state']},
        action=d['action'], reward=d['reward'], discount=d['discount'],
        next_observation={'state':d['next_state'],'privileged_state':d['next_privileged_state']},
        extras={'state_extras':{'truncation':d['truncation']},
                'policy_extras':{k:d[k] for k in ['raw_action','log_prob','distribution_params']}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import functools
    import jax
    import numpy as np
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks, losses
    from robocub.scripts.audit_g1_critic_resets import load_saved_critic
    cfg = json.loads((args.run_dir / 'resolved_config.json').read_text())['agent']
    assert cfg['normalize_observations'] and cfg['max_grad_norm'] == 1.0
    path = args.run_dir / 'logs/checkpoints/20070400/policy'
    norm, actor, critic = load_saved_critic(path)
    source = json.loads((args.batch_dir / 'summary.json').read_text())
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != source['checkpoint_sha256']:
        raise ValueError('Batch and checkpoint mismatch')
    first = load_batch(args.batch_dir / 'stand_17_batch.npz')
    shapes = jax.tree.map(lambda x:x.shape[2:], first.observation)
    net = networks.make_ppo_networks(shapes, first.action.shape[-1],
        policy_hidden_layer_sizes=cfg['policy_hidden_layer_sizes'],
        value_hidden_layer_sizes=cfg['value_hidden_layer_sizes'],
        policy_obs_key='state',value_obs_key='privileged_state',
        preprocess_observations_fn=running_statistics.normalize)
    fn = jax.jit(functools.partial(gradient_budget,net,losses.PPONetworkParams(actor,critic),norm,config=cfg))
    rows = {}
    for cmd in ['stand','forward']:
        for draw in [17,29]:
            tag=f'{cmd}_{draw}';batch_path=args.batch_dir/(tag+'_batch.npz')
            batch=load_batch(batch_path)
            values={k:float(v) for k,v in fn(batch=batch,key=jax.random.PRNGKey(draw+1000)).items()}
            if not all(np.isfinite(v) for v in values.values()) or values['actor_norm']<=1e-12:
                raise ValueError('Nonfinite/degenerate budget')
            if not np.isclose(values['policy_loss'],source['batches'][tag]['policy_loss'],atol=1e-6,rtol=1e-5):
                raise ValueError('Saved policy loss replay mismatch')
            rows[tag]={'metrics':values,'batch_sha256':hashlib.sha256(batch_path.read_bytes()).hexdigest()}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report={'full_validation':False,'optimizer_steps':0,'new_rollouts':0,
        'checkpoint_sha256':checksum,'rows':rows,
        'limitation':'Frozen-normalizer diagnostic batches, not exact training minibatches or Adam history. Shared clipping rescales both networks, and Adam can compensate for common scale. Norm dominance alone does not establish effective update starvation or justify a training change.'}
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
