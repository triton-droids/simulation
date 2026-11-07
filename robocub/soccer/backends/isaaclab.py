"""Isaac Lab3.0 Newton MJWarp backend. No silent fallback to another simulator."""
import importlib.metadata,json,time,math,platform,subprocess
from pathlib import Path
import numpy as np
from robocub.soccer.field.geometry import Field,ROOT
from robocub.soccer.field.usd import write_usd
from robocub.soccer.robots.importer import require_robot
from robocub.soccer.randomization.sampler import Sampler
from robocub.soccer.evaluation.baseline import checkpoint_spec


def execute(args):
    try:version=importlib.metadata.version('isaaclab')
    except importlib.metadata.PackageNotFoundError:raise RuntimeError('Isaac Lab is absent. Use the isolated .cache/field01_venv Python or install isaaclab==3.0.0. --backend mujoco is an explicit CPU diagnostic alternative, not Isaac validation.')
    if version!='3.0.0':raise RuntimeError(f'Isaac Lab3.0.0 required, got{version}')
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA-enabled PyTorch unavailable; install official cu128 wheels compatible with the driver, or use --device cpu for diagnostics. GPU tests have NOT passed.')
    info=require_robot(args.robot,args.robot_asset,args.robot_metadata)
    field=Field.load(args.field)
    out=args.run_dir or ROOT/'Artifacts/Results/field01'/str(time.time_ns());out.mkdir(parents=True,exist_ok=False)
    if args.robot=='g1' or (args.robot_asset and Path(args.robot_asset).suffix=='.xml'):
        import mujoco_usd_converter  # Register MJCF schemas BEFORE the first USD stage.
    field_usd,proxy_usd=write_usd(field,out/'assets')
    from isaaclab.app import launch_simulation
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg,SimulationContext
    from isaaclab.assets import AssetBaseCfg,RigidObjectCfg,ArticulationCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import instantiate
    from isaaclab.actuators import IdealPDActuatorCfg
    from isaaclab_newton.physics import NewtonCfg,MJWarpSolverCfg
    visualizers=[]
    if args.command=='view':
        from isaaclab_visualizers.viser import ViserVisualizerCfg
        visualizers=[ViserVisualizerCfg(bind_address='127.0.0.1',port=8080,share=False,open_browser=False,record_to_viser=str(out/'scene.viser') if args.record else None)]
    cfg=SimulationCfg(dt=.005,device=args.device,physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(iterations=100,integrator="implicitfast"),use_cuda_graph=args.device.startswith('cuda')),visualizer_cfgs=visualizers)
    scene_cfg=InteractiveSceneCfg(num_envs=args.num_envs,env_spacing=field.spacing,replicate_physics=True)
    scene_cfg.field=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Field',spawn=sim_utils.UsdFileCfg(usd_path=str(field_usd.resolve())))
    scene_cfg.ball=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Ball',spawn=sim_utils.SphereCfg(radius=field.ball_radius,rigid_props=sim_utils.RigidBodyPropertiesCfg(),mass_props=sim_utils.MassPropertiesCfg(mass=field.ball_mass),collision_props=sim_utils.CollisionPropertiesCfg(),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.95,.95,.95))),init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,field.ball_radius+.01)))
    meta=info.get('operator_metadata',{});asset=str(proxy_usd.resolve())
    if args.robot=='real':
        suffix=Path(args.robot_asset).suffix.lower()
        if suffix in ('.usd','.usda','.usdc'):spawn=sim_utils.UsdFileCfg(usd_path=str(Path(args.robot_asset).resolve()))
        elif suffix=='.urdf':spawn=sim_utils.UrdfFileCfg(asset_path=str(Path(args.robot_asset).resolve()),fix_base=False)
        else:spawn=sim_utils.MjcfFileCfg(asset_path=str(Path(args.robot_asset).resolve()),fix_base=False)
    else:spawn=sim_utils.UsdFileCfg(usd_path=asset)
    scene_cfg.robot=ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',spawn=spawn,init_state=ArticulationCfg.InitialStateCfg(pos=(-2,0,meta.get('initial_height',0.)),joint_pos=meta.get('nominal_joints',{'.*':0.})),actuators={'joints':IdealPDActuatorCfg(joint_names_expr=['.*'],stiffness=meta.get('kp',20.),damping=meta.get('kd',1.),effort_limit=meta.get('effort_limit',12.))})
    if args.robot=='g1':
        from robocub.soccer.robots.g1 import g1_info
        info=g1_info('g1.xml');meta=info['operator_metadata']
        names=list(meta['nominal_joints'])
        converted=mujoco_usd_converter.Converter(layer_structure=True,scene=False).convert(info['path'],str((out/'g1_import').resolve()))
        info['conversion']='mujoco-usd-converter0.6.1 native Newton schema; no PhysX importer postprocessing'
        info['converted_asset']=converted.path
        scene_cfg.robot=ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',
            spawn=sim_utils.UsdFileCfg(usd_path=converted.path),
            init_state=ArticulationCfg.InitialStateCfg(pos=(-2,0,meta['initial_height']),joint_pos=meta['nominal_joints']),
            actuators={'joints':IdealPDActuatorCfg(joint_names_expr=names,
                stiffness=dict(zip(names,info['actuator_gains'])),
                damping=dict(zip(names,info['actuator_damping'])),
                effort_limit={name:max(abs(x) for x in bounds) for name,bounds in zip(names,info['force_limits'])})})
    import yaml
    config=yaml.safe_load(args.randomization_config.read_text()) if args.randomization_config else None
    sampler=Sampler(field,args.seed,args.randomization,'isaaclab',args.rank,args.suite,config)
    episodes=np.full(args.num_envs,-1,int);records=[None]*args.num_envs
    start_setup=time.perf_counter()
    with launch_simulation(cfg):
        sim=SimulationContext(cfg);scene=instantiate(scene_cfg);sim.set_camera_view([0.,-.01,field.length*1.4],[0.,0.,0.]);sim.reset()
        robot,ball=scene['robot'],scene['ball'];n=robot.data.joint_pos.torch.shape[1]
        if args.robot=='g1' and n!=29:raise RuntimeError(f'G1 asset has {n} joints rather than required29; refusing identity mismatch')
        from robocub.soccer.randomization.isaac_runtime import IsaacRandomizer
        randomizer=IsaacRandomizer(robot,ball,scene,sampler,args.device)
        steps=np.zeros(args.num_envs,int);previous=torch.zeros((args.num_envs,n),device=args.device)
        def reset(ids):
            ids=list(map(int,ids));idx=torch.tensor(ids,device=args.device,dtype=torch.int32)
            rp=robot.data.default_root_pose.torch[ids].clone();bp=ball.data.default_root_pose.torch[ids].clone();bv=ball.data.default_root_vel.torch[ids].clone()
            for local,i in enumerate(ids):
                episodes[i]+=1;steps[i]=0;rec=sampler.sample(i,episodes[i]);records[i]=rec;v=rec['values'];rp[local,:2]=torch.tensor(v['robot_spawn_xy'],device=args.device);yaw=v['robot_spawn_yaw'];rp[local,3:]=torch.tensor([0,0,math.sin(yaw/2),math.cos(yaw/2)],device=args.device);bp[local,:2]=torch.tensor(v['ball_spawn_xy'],device=args.device);bv[local,:2]=torch.tensor(v.get('ball_velocity',[0,0]),device=args.device)
            rp[:,:3]+=scene.env_origins[ids];bp[:,:3]+=scene.env_origins[ids]
            robot.write_root_pose_to_sim_index(root_pose=rp,env_ids=idx);robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch[ids].clone(),env_ids=idx)
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch[ids].clone(),env_ids=idx);robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch[ids].clone(),env_ids=idx)
            ball.write_root_pose_to_sim_index(root_pose=bp,env_ids=idx);ball.write_root_velocity_to_sim_index(root_velocity=bv,env_ids=idx);scene.reset(idx);previous[ids]=0
            randomizer.reset(ids,records)
        reset(range(args.num_envs))
        # Compile and warm physics before throughput timing, then restore all states.
        robot.set_joint_position_target_index(target=robot.data.default_joint_pos.torch)
        for _ in range(20):scene.write_data_to_sim();sim.step();scene.update(.005)
        reset(range(args.num_envs));scene.write_data_to_sim();scene.update(.005)
        isolation={'checked':False}
        if args.num_envs>1:
            before=ball.data.root_pos_w.torch[1:].clone()
            reset([0]);scene.write_data_to_sim();scene.update(.005)
            unchanged=torch.equal(before,ball.data.root_pos_w.torch[1:])
            if not unchanged:raise RuntimeError('Reset changed another environment ball state')
            isolation={'checked':True,'reset_zero_preserves_other_balls':True}
        physics_checks=None
        if args.validate_physics:
            from robocub.soccer.evaluation.gpu_physics import check_ball_physics
            physics_checks=check_ball_physics(scene,sim,field,args.device)
            reset(range(args.num_envs))
        setup=time.perf_counter()-start_setup
        if args.device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        panel=None
        if args.command=='view':
            from robocub.soccer.viewer.isaac_panel import FieldPanel
            panel=FieldPanel(sim,field,info['name'],args.seed,args.randomization)
            panel.mode=args.camera.title();panel.camera()
            if args.record:panel.pause.value=False
        start=time.perf_counter();count=0;falls=0

        # Scripted runner; native scene exposes tensors for an external trainer.
        with (out/'episodes.jsonl').open('w') as log:
            for r in records:log.write(json.dumps(r)+'\n')
            while sim.is_running() and (args.command=='view' or count<args.steps):
                if panel:
                    if args.view_seconds and time.perf_counter()-start>=args.view_seconds:break
                    while not panel.events.empty():
                        event,value=panel.events.get()
                        if event=='camera':panel.mode=value;panel.camera(robot.data.root_pos_w.torch[0].cpu().numpy());panel.message.content=f'{value} camera applied.'
                        elif event=='reset':reset([0]);panel.message.content='Reset completed.'
                        else:
                            moving=ball if event=='ball' else robot;other=robot if event=='ball' else ball
                            if np.linalg.norm(np.array(value)-other.data.root_pos_w.torch[0,:2].cpu().numpy())<.7:
                                panel.message.content='Placement rejected: robot/ball overlap.';continue
                            pose=moving.data.root_pose_w.torch.clone();pose[0,:2]=torch.tensor(value,device=args.device)
                            ids=torch.tensor([0],device=args.device,dtype=torch.int32)
                            moving.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
                            moving.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1,6),device=args.device),env_ids=ids)
                            scene.write_data_to_sim();scene.update(.005)
                            panel.message.content=f'{event.title()} placed at {value}.'
                    if panel.mode=='Follow':panel.camera(robot.data.root_pos_w.torch[0].cpu().numpy())
                    if panel.pause.value:sim.render();time.sleep(.02);continue
                target=torch.as_tensor(.2*np.sin(steps[:,None]*.02*2+np.arange(n)[None,:]*math.pi),device=args.device,dtype=torch.float32)
                if args.robot=='g1':target.zero_()  # Nominal pose; no unvalidated walking policy.
                command=target.clone()
                for i,r in enumerate(records):
                    if r['values'].get('control_latency',0):command[i]=previous[i]
                previous=target
                robot.set_joint_position_target_index(target=command+robot.data.default_joint_pos.torch)
                for _ in range(4):
                    randomizer.forces(records,steps);scene.write_data_to_sim();sim.step();scene.update(.005)
                steps+=1;count+=1
                observation=randomizer.observe(records,steps)
                if not torch.isfinite(observation).all():raise RuntimeError("Nonfinite observation")
                pos=ball.data.root_pos_w.torch-scene.env_origins
                root=robot.data.root_pos_w.torch-scene.env_origins
                if not torch.isfinite(pos).all() or not torch.isfinite(root).all() or not torch.isfinite(robot.data.joint_pos.torch).all():raise RuntimeError('Isaac nonfinite state')
                fallen=torch.nonzero(root[:,2]<.35,as_tuple=False).flatten().cpu().tolist() if args.robot!='demo' else []
                falls+=len(fallen)
                if panel and fallen:panel.pause.value=True;panel.message.content='Robot fell under joint hold. Reset to continue; no walking policy loaded.'
                ids=torch.nonzero((pos[:,0].abs()>field.length/2+field.goal_depth+.5)|(pos[:,1].abs()>field.width/2+.5),as_tuple=False).flatten().cpu().tolist()
                ids=sorted(set(ids+np.flatnonzero(steps>=1500).tolist()+(fallen if panel is None else [])))
                if ids:
                    reset(ids)
                    for i in ids:log.write(json.dumps(records[i])+'\n')
                if panel:time.sleep(.02)
            if args.device.startswith('cuda'):torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        result={'backend':'isaaclab3-newton-mjwarp','device':args.device,'gpu_simulation':args.device.startswith('cuda'),'num_envs':args.num_envs,'control_steps':count,'wall_seconds':elapsed,'setup_seconds':setup,'vector_steps_per_second':count/elapsed,'aggregate_env_steps_per_second':args.num_envs*count/elapsed,'aggregate_physics_steps_per_second':args.num_envs*count*4/elapsed,'effective_fps':args.num_envs*count/elapsed,'torch_peak_vram_mib':torch.cuda.max_memory_allocated()/2**20 if args.device.startswith('cuda') else None,'vram_note':'torch allocations only; Warp allocations excluded','asset':info,'seed':args.seed,'config_hash':sampler.hash,'host':platform.node(),'physics_dt':.005,'decimation':4,'physics_checks':physics_checks,'isolation':isolation,'warmup_physics_steps':20,'unverified':'Reset isolation checks ball states; physical calibration and controller stability are not established'}
        from robocub.soccer.evaluation.provenance import robocub_manifest
        result['source_manifest']=source_manifest()
        result['runtime_solver_manager']=str(cfg.physics.class_type)
        result['isaaclab_version']=version
        result['torch_version']=torch.__version__
        result['torch_cuda']=torch.version.cuda
        result['joint_count']=n
        result['observation_shape']=[args.num_envs,2*n+13]
        result['action_shape']=[args.num_envs,n]
        result['fall_resets']=falls
        result['baseline']='nominal joint hold for G1; sinusoidal joints for demo; not learned soccer'
        if args.command=='train':(out/'checkpoint.json').write_text(json.dumps(checkpoint_spec(args)))
        (out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));sim.stop()
    return 0

