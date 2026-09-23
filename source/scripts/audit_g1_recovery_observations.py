"""Read-only audit of actor observation scaling at nominal/randomized starts."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from source.scripts import evaluate_g1 as ev
from brax.io import model
import jax
import jax.numpy as jp
import numpy as np
from omegaconf import OmegaConf

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--checkpoint',type=int,required=True);p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument("--seeds",default="6000,6001,6002")
 a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
 cfg=OmegaConf.load(a.run_dir/'resolved_config.json');cfg.robot.fetch_model=False;cfg.sim.playground.fetch_source=False;cfg.sim.noise.add_noise=False;cfg.sim.push.add_push=False;cfg.sim.domain_rand.add_domain_rand=False
 if 'recovery_reset_candidates' in cfg.sim.playground: cfg.sim.playground.recovery_reset_candidates=1
 params=model.load_params(a.run_dir/'logs/checkpoints'/str(a.checkpoint)/'policy');stats=params[0]
 mean=np.asarray(stats.mean['state']);std=np.asarray(stats.std['state']);rows=[]
 for randomized in [False,True]:
  cfg.sim.reset.randomize=randomized
  env=ev.get_env_class(cfg.env.name)(cfg.robot.name,ev.make_robot(cfg.robot),cfg.env.terrain,cfg.sim);reset=jax.jit(env.reset)
  for seed in [int(v) for v in a.seeds.split(",")]:
   state=ev._replace_command(reset(jax.random.split(jax.random.PRNGKey(seed))[0]),jp.array([.45,0.,0.]))
   obs=np.asarray(state.obs['state']);z=(obs-mean)/np.maximum(std,1e-8)
   assert np.isfinite(z).all()
   indices=np.argsort(np.abs(z))[-12:][::-1]
   rows.append(dict(randomized=randomized,seed=seed,initial_world_qvel=np.asarray(state.data.qvel[:6]).tolist(),initial_qpos=np.asarray(state.data.qpos).tolist(),initial_actor_velocity=obs[:3].tolist(),max_abs_standardized=float(np.max(np.abs(z))),channels_over_10=int(np.sum(np.abs(z)>10)),top=[dict(index=int(i),raw=float(obs[i]),mean=float(mean[i]),std=float(std[i]),standardized=float(z[i])) for i in indices]))
 report=dict(kind='observation_scaling_diagnostic',note='Raw standardized deviations with std floor1e-8, not a gait gate or training change.',run=str(a.run_dir),normalizer_count_leaves=[np.asarray(x).tolist() for x in jax.tree_util.tree_leaves(stats.count)],rows=rows,git=ev._git_record())
 (a.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
