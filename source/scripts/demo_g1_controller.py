"""One-command T02 simulation demo; default tests moving, turning, then stopping."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MUJOCO_GL','egl' if sys.platform!='win32' else 'glfw')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from source.locomotion.unitree_g1.simulation_controller import SimulationController,DEFAULT_RUN,CHECKPOINT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--profile',choices=['smoke','fixed','transitions'],default='transitions')
    p.add_argument('--video',action='store_true')
    args=p.parse_args()
    import numpy as np
    import mujoco,cv2
    c=SimulationController(args.run_dir)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    commands=json.loads((ROOT/'research/queues/final_f1_commands.json').read_text())
    profiles=([(k,[(k,v,500)]) for k,v in commands.items()] if args.profile=='fixed' else
        [('smoke',[('stand',[0,0,0],5)])] if args.profile=='smoke' else
        [('move_turn_stop',[('forward',commands['forward'],500),('turn_left',commands['turn_left'],500),('stand',[0,0,0],500)])])
    results=[]
    renderer=mujoco.Renderer(c.env.mj_model,height=360,width=640) if args.video else None
    data=mujoco.MjData(c.env.mj_model) if args.video else None
    try:
        for episode,segments in profiles:
            c.reset(6000);rows=[]
            writer=cv2.VideoWriter(str(args.output_dir/(episode+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),25,(640,360)) if args.video else None
            if writer is not None and not writer.isOpened(): raise RuntimeError('MP4 writer unavailable')
            try:
                for label,cmd,steps in segments:
                    c.set_command(*cmd);start=len(rows)
                    for _ in range(steps):
                        s=c.step();q=np.asarray(s.data.qpos);v=np.asarray(c.env.get_local_linvel(s.data,'pelvis'));g=np.asarray(c.env.get_gyro(s.data,'pelvis'))
                        row={'segment':label,'command':cmd,'velocity':v.tolist(),'yaw_rate':float(g[2]),'height':float(q[2]),'done':bool(s.done),'qpos':q.tolist()};rows.append(row)
                        if not np.isfinite(q).all(): raise RuntimeError('Nonfinite state')
                        if writer is not None and len(rows)%2==0:
                            data.qpos[:]=q;mujoco.mj_forward(c.env.mj_model,data)
                            camera=mujoco.MjvCamera();camera.lookat[:]=q[:3];camera.distance=2.5;camera.azimuth=135;camera.elevation=-15
                            renderer.update_scene(data,camera=camera);writer.write(cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
                        if bool(s.done): break
                    part=rows[start:];lin=float(np.sqrt(np.mean([sum((r['velocity'][j]-cmd[j])**2 for j in range(2)) for r in part])));yaw=float(np.sqrt(np.mean([(r['yaw_rate']-cmd[2])**2 for r in part])))
                    result={'episode':episode,'segment':label,'requested_steps':steps,'steps':len(part),'linear_rmse':lin,'yaw_rmse':yaw,'minimum_height':min(r['height'] for r in part),'terminated':part[-1]['done']}
                    results.append(result);print(json.dumps(result),flush=True)
                    (args.output_dir/'summary.json').write_text(json.dumps({'checkpoint':CHECKPOINT,'profile':args.profile,'full_validation':False,'segments':results},indent=2))
                    if part[-1]['done']: break
            finally:
                if writer is not None: writer.release()
                (args.output_dir/(episode+'.json')).write_text(json.dumps(rows))
    finally:
        if renderer is not None: renderer.close()

if __name__=='__main__': main()
