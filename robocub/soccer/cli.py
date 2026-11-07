"""Field01 CLI. Default Isaac Lab; CPU diagnostics require explicit --backend mujoco."""
import argparse,json,sys,time,subprocess,platform,hashlib
from pathlib import Path
import numpy as np
from robocub.soccer.field.geometry import Field,ROOT
from robocub.soccer.randomization.sampler import digest
from robocub.soccer.evaluation.baseline import checkpoint_spec


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['doctor','view','smoke','train','evaluate','report','sample'])
    p.add_argument('--backend',choices=['isaaclab','mujoco'],default='isaaclab')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--robot',choices=['demo','real','g1'],default='g1')
    p.add_argument('--robot_asset');p.add_argument('--robot_metadata')
    p.add_argument('--field',default='medium_14x9',choices=['medium_14x9','large_22x14'])
    p.add_argument('--num_envs',type=int,default=1);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--randomization',choices=['disabled','nominal','mild','robust','stress'],default='mild')
    p.add_argument('--randomization_config',type=Path)
    p.add_argument('--suite',choices=['train','validation','nominal','heldout'],default='train')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--run_dir',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--view_seconds',type=float,default=0)
    p.add_argument('--validate_physics',action='store_true')
    p.add_argument('--record',action='store_true');p.add_argument('--smoke_only',action='store_true')
    p.add_argument('--camera',choices=['overhead','orbit','follow'],default='overhead')
    return p


def run(args):
    if args.command=='doctor':
        from robocub.soccer.doctor import doctor
        result=doctor();print(json.dumps(result,indent=2))
        if args.run_dir:args.run_dir.mkdir(parents=True,exist_ok=False);(args.run_dir/'doctor.json').write_text(json.dumps(result,indent=2))
        return 0
    if args.command=='report':
        if not args.run_dir:raise ValueError('--run_dir required')
        print((args.run_dir/'metrics.json').read_text());return 0
    if args.command=='train' and not args.smoke_only:raise ValueError('This package supplies a training-compatible task, not a new RL algorithm. Use train --smoke_only for scripted task validation; connect your existing trainer explicitly.')
    if args.command=='evaluate':
        if args.suite=='train':raise ValueError('Evaluation needs --suite nominal/validation/heldout')
        if args.suite=='nominal':args.randomization='nominal'
        if args.suite=='heldout':args.randomization='stress'
        if not args.checkpoint or not args.checkpoint.is_file():raise ValueError('Supply the scripted-baseline checkpoint.json from train --smoke_only. Neural policy adapter is not implemented.')
        cp=json.loads(args.checkpoint.read_text())
        if cp!=checkpoint_spec(args):raise ValueError('Unsupported checkpoint; never silently replace a learned policy')
    field=Field.load(args.field)
    import yaml
    config=yaml.safe_load(args.randomization_config.read_text()) if args.randomization_config else None
    if args.command=='sample':
        from robocub.soccer.randomization.sampler import Sampler
        sampler=Sampler(field,args.seed,args.randomization,args.backend,args.rank,args.suite,config)
        samples=[sampler.sample(i,0) for i in range(args.num_envs)]
        if args.run_dir:
            args.run_dir.mkdir(parents=True,exist_ok=False)
            (args.run_dir/'samples.json').write_text(json.dumps(samples,indent=2))
            hist={k:sum(k in s['enabled'] for s in samples) for k in sampler.sample(0,0)['values']}
            from robocub.soccer.randomization.registry import REGISTRY
            hist={k:sum(k in s['enabled'] for s in samples) for k in REGISTRY}
            (args.run_dir/'histogram.json').write_text(json.dumps(hist,indent=2))
        print(json.dumps({'sample_count':len(samples),'example':samples[0]},indent=2));return 0
    if not 1<=args.num_envs<=4096:raise ValueError('num_envs must be1..4096')
    if args.steps<=0:raise ValueError('steps must be positive')
    if args.command=='view' and args.num_envs!=1:raise ValueError('Interactive viewer requires exactly one environment')
    if args.backend=='isaaclab':
        from robocub.soccer.backends.isaaclab import execute
        return execute(args)
    from robocub.soccer.envs.task import SoccerEnv
    t0=time.perf_counter()
    env=SoccerEnv(args.num_envs,field,args.seed,args.randomization,args.suite,args.rank,args.robot,args.robot_asset,args.robot_metadata,config)
    if args.command=='view' and not args.record:
        from robocub.soccer.viewer.app import run_viewer
        run_viewer(env,args.camera,args.view_seconds);return 0
    out=args.run_dir or Path('Artifacts/Results/field01')/str(time.time_ns())
    if args.record:
        from robocub.soccer.viewer.app import record
        print(record(env,out,args.steps,args.camera));return 0
    out.mkdir(parents=True,exist_ok=False)
    (out/'initial_samples.json').write_text(json.dumps(env.records,indent=2))
    setup=time.perf_counter()-t0
    resets=0;total_reward=np.zeros(args.num_envs)
    start=time.perf_counter()
    with (out/'episodes.jsonl').open('w') as log:
        for rec in env.records:log.write(json.dumps(rec)+'\n')
        for _ in range(args.steps):
            obs,reward,term,trunc,info=env.step(env.scripted_action());total_reward+=reward
            if not np.isfinite(obs).all():raise RuntimeError('Nonfinite observations')
            ids=np.flatnonzero(term|trunc)
            if len(ids):
                env.reset(ids);resets+=len(ids)
                for i in ids:log.write(json.dumps(env.records[i])+'\n')
    elapsed=time.perf_counter()-start
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except Exception:commit='unknown'
    result={'backend':'mujoco-cpu-diagnostic','gpu_simulation':False,'num_envs':args.num_envs,'control_steps':args.steps,'physics_dt':.005,'decimation':4,'wall_seconds':elapsed,'setup_seconds':setup,'vector_steps_per_second':args.steps/elapsed,'aggregate_env_steps_per_second':args.steps*args.num_envs/elapsed,'aggregate_physics_steps_per_second':4*args.steps*args.num_envs/elapsed,'effective_fps':args.steps*args.num_envs/elapsed,'gpu_peak_vram_mib':None,'vram_note':'CPU backend; no GPU throughput claim','host':platform.node(),'device':'cpu','python':sys.version,'asset':env.info,'commit':commit,'config_hash':env.sampler.hash,'seed':args.seed,'suite':args.suite,'resets':resets,'mean_reward':float(total_reward.mean()),'observation_shape':list(obs.shape),'action_shape':[args.num_envs,env.n_actions],'baseline':'scripted joint motion, not learned soccer'}
    from robocub.soccer.evaluation.provenance import robocub_manifest
    result['source_manifest']=source_manifest()
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    if args.command=='train':(out/'checkpoint.json').write_text(json.dumps(checkpoint_spec(args)))
    print(json.dumps(result,indent=2));env.close();return 0


def main():
    try:return run(parser().parse_args())
    except (ValueError,FileNotFoundError,RuntimeError,ImportError) as e:
        print(f'FIELD01 ERROR: {e}',file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())



