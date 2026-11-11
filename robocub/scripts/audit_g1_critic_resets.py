"""Read saved critic predictions at original reset states; no trajectories or training."""
import argparse
import csv
import functools
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def rank_failures(rows):
    result=[]
    for row in rows:
        if row['episode_steps'] >= 500:
            continue
        peers=[x['value'] for x in rows if x['command']==row['command'] and x['episode_steps']==500]
        result.append({'command':row['command'],'seed':row['seed'],
                       'failed_value':row['value'],'survivor_values':peers,
                       'above_all_survivors':bool(peers and row['value'] >= max(peers))})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--episodes-csv',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    from robocub.scripts import evaluate_g1 as ev
    import jax
    import jax.numpy as jp
    import numpy as np
    from omegaconf import OmegaConf
    args.output_dir.mkdir(parents=True,exist_ok=False)
    cfg=OmegaConf.load(args.run_dir/'resolved_config.json')
    cfg.sim.playground.recovery_reset_candidates=1
    cfg.sim.playground.reset_disturbance_scale=1.0
    cfg.robot.fetch_model=False
    cfg.sim.noise.add_noise=False
    cfg.sim.push.add_push=False
    cfg.sim.domain_rand.add_domain_rand=False
    cfg.sim.commands.resample_time=501*(cfg.sim.sim.timestep*cfg.sim.action.n_frames)
    cfg.sim.reset.randomize=True
    env=ev.get_env_class(cfg.env.name)(cfg.robot.name,ev.make_robot(cfg.robot),cfg.env.terrain,cfg.sim)
    factory=functools.partial(ev.ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=cfg.agent.value_hidden_layer_sizes,
        policy_obs_key='state',value_obs_key='privileged_state')
    net=ev._make_evaluation_network(factory,env.observation_size,env.action_size,
        normalize_observations=bool(cfg.agent.normalize_observations))
    path=args.run_dir/'logs/checkpoints/1146880/policy'
    params=ev.model.load_params(path)

    if len(params)!=3:
        raise ValueError('Saved critic required; no initialized substitute')
    def predict(seed,command):
        key,_=jax.random.split(jax.random.PRNGKey(seed))
        state=ev._replace_command(env.reset(key),command)
        value=net.value_network.apply(params[0],params[2],state.obs)
        data=ev._state_data(env,state)
        return value,data.qpos,data.qvel
    predict=jax.jit(predict)
    labels=[r for r in csv.DictReader(args.episodes_csv.open()) if r['controller']=='trained']
    expected={(c,s) for c in ['stand','forward'] for s in [6000,6001,6002,6014]}
    assert len(labels)==8 and {(r['command_name'],int(r['reset_seed'])) for r in labels}==expected
    rows=[]
    for r in labels:
        cmd=jp.asarray([float(r[k]) for k in ['command_vx','command_vy','command_yaw_rate']])
        value,qpos,qvel=map(np.asarray,predict(int(r['reset_seed']),cmd))
        assert np.isfinite(value).all() and np.isfinite(qpos).all() and np.isfinite(qvel).all()
        rows.append({'command':r['command_name'],'seed':int(r['reset_seed']),
                     'episode_steps':int(r['episode_steps']),'episode_return':float(r['episode_return']),
                     'value':float(value),'qpos':qpos.tolist(),'qvel':qvel.tolist()})
    summary={'full_validation':False,'kind':'initial_critic_rank_audit',
             'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'labels_sha256':hashlib.sha256(args.episodes_csv.read_bytes()).hexdigest(),
             'rows':rows,'failed_state_ranks':rank_failures(rows),
             'limitation':'Critic predicts discounted stochastic-policy reward, not deterministic survival. Rank inversion is not proof of a critic defect or causal failure.',
             'effective_config':OmegaConf.to_container(cfg,resolve=True)}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
