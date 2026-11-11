"""Bounded same-executable repeatability diagnostic; never validation/training."""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robocub.utils.trace_reproducibility import compare_traces


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--capture-compiler-ir',action='store_true')
    p.add_argument('--compile-only',action='store_true')
    args=p.parse_args()
    if args.compile_only and not args.capture_compiler_ir:
        p.error('--compile-only requires --capture-compiler-ir')
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
    f=ev._build_rollout(env,net,500)
    seed=jp.asarray(6001,dtype=jp.int32);use=jp.asarray(True)
    lowered=f.lower(params,use,seed,jp.asarray([0.,0.,0.]))
    compiled=lowered.compile()
    compiler_ir={}
    if args.capture_compiler_ir:
        for name,text in [('lowered',lowered.as_text()),('compiled',compiled.as_text())]:
            if not isinstance(text,str) or not text:
                raise RuntimeError('Compiler text unavailable: '+name)
            (args.output_dir/(name+'.txt')).write_text(text)
            compiler_ir[name]={'sha256':hashlib.sha256(text.encode()).hexdigest(),'bytes':len(text.encode())}
    traces={};initial={};steps={}
    for repeat in range(0 if args.compile_only else 2):
        for name,cmd in [('stand',[0.,0.,0.]),('forward',[.45,0.,0.])]:
            data,trace=compiled(params,use,seed,jp.asarray(cmd))
            jax.block_until_ready(trace['reward'])
            host=jax.tree.map(np.asarray,trace)
            tag=f'{name}_{repeat}'
            np.savez_compressed(args.output_dir/(tag+'.npz'),**host)
            initial[tag]={'qpos':np.asarray(data.qpos).tolist(),'qvel':np.asarray(data.qvel).tolist()}
            steps[tag]=int(host['valid'].sum());traces[tag]=host
    gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,pci.bus_id,name,driver_version','--format=csv'],capture_output=True,text=True)
    summary={'kind':'same_compiled_executable_diagnostic','full_validation':False,
        'compile_only':args.compile_only,'compiler_ir':compiler_ir,
        'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'comparisons':{name:compare_traces(traces[name+'_0'],traces[name+'_1']) for name in ([] if args.compile_only else ['stand','forward'])},
        'steps':steps,'initial_states':initial,'jax_devices':[str(x) for x in jax.devices()],
        'gpu_query':gpu.stdout,'gpu_query_returncode':gpu.returncode,
        'environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','XLA_FLAGS','JAX_COMPILATION_CACHE_DIR']},
        'effective_config':OmegaConf.to_container(cfg,resolve=True)}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
