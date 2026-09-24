"""Verify reset-scale endpoints and state/observation coherence on real MJX."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from source.scripts import evaluate_g1 as ev
from omegaconf import OmegaConf
import jax,numpy as np
cfg=OmegaConf.load('results/post_f1_c10/train/resolved_config.json')
cfg.robot.fetch_model=False;cfg.sim.playground.fetch_source=False
cfg.sim.noise.add_noise=False;cfg.sim.push.add_push=False;cfg.sim.reset.randomize=True
cfg.sim.playground.reset_disturbance_scale=1.0
env=ev.get_env_class(cfg.env.name)(cfg.robot.name,ev.make_robot(cfg.robot),cfg.env.terrain,cfg.sim)
key=jax.random.PRNGKey(734)
original=jax.jit(env._env.reset)(key)
rows=[]
for scale in [1.0,0.0,0.5]:
 env._reset_disturbance_scale=scale
 state=jax.jit(env.reset)(key);jax.block_until_ready(state)
 np.testing.assert_allclose(state.data.qpos[:7],original.data.qpos[:7],atol=1e-6,rtol=0)
 np.testing.assert_allclose(state.data.qvel,original.data.qvel*scale,atol=1e-6,rtol=0)
 np.testing.assert_allclose(state.data.qpos[7:],env._env._init_q[7:]+scale*(original.data.qpos[7:]-env._env._init_q[7:]),atol=1e-6,rtol=0)
 assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(state.obs))
 if scale==1:
  np.testing.assert_array_equal(state.obs['state'],original.obs['state'])
 else:
  info=dict(state.info);obs=env._env._get_obs(state.data,info,env._contact(state.data))
  np.testing.assert_allclose(state.obs['state'],obs['state'],atol=1e-6,rtol=0)
 rows.append(dict(scale=scale,passed=True))
print(json.dumps(rows))
