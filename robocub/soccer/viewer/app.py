"""Native viewer and offscreen evidence. Camera starts above the entire field."""
import json,time,threading
from pathlib import Path
import numpy as np
import mujoco
from robocub.soccer.robots.importer import LABEL

def camera(field,mode='overhead',robot_pos=None):
    c=mujoco.MjvCamera();c.type=mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:]=[0,0,0];c.azimuth=90;c.elevation=-89.9;c.distance=field.length*1.4
    if mode=='orbit':c.elevation=-40;c.azimuth=135;c.distance=field.length*1.2
    elif mode=='follow':c.lookat[:]=robot_pos if robot_pos is not None else [0,0,.5];c.elevation=-25;c.azimuth=0;c.distance=4.
    return c

def record(env,out,steps=150,mode='overhead'):
    import cv2
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    renderer=mujoco.Renderer(env.models[0],height=720,width=1200)
    writer=cv2.VideoWriter(str(out/'field01.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(1200,720))
    if not writer.isOpened():raise RuntimeError('MP4 encoder unavailable')
    records=[]
    try:
        for step in range(steps):
            obs,r,term,trunc,info=env.step(env.scripted_action())
            if step%2==0:
                renderer.update_scene(env.data[0],camera= camera(env.field,mode,env.data[0].xpos[env.root_id]))
                frame=renderer.render();frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                label=env.info['name']
                cv2.putText(frame,label,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,200,255),2)
                cv2.putText(frame,f'{env.field.id} | {env.backend} CPU | seed {env.sampler.seed} | {step*env.dt:.2f}s',(12,55),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                writer.write(frame)
                if step==0:cv2.imwrite(str(out/'field01.png'),frame)
            records.append({'step':step,'ball':env.data[0].qpos[env.bq:env.bq+3].tolist(),'joints':env.data[0].qpos[env.jq].tolist(),'reward':r.tolist(),'root_position':env.data[0].xpos[env.root_id].tolist(),'terminated':bool(term[0]),'truncated':bool(trunc[0])})
            if any(term|trunc):break
    finally:writer.release();renderer.close()
    (out/'trajectory.json').write_text(json.dumps(records))
    (out/'manifest.json').write_text(json.dumps({'asset':env.info,'sample':env.records[0],'steps':len(records),'camera':mode,'backend':env.backend,'physical_fidelity':'uncalibrated diagnostic'},indent=2))
    return str(out)

def run_viewer(env,mode='overhead',seconds=0):
    import mujoco.viewer
    events=[];mutex=threading.Lock();paused=False;current=mode
    def key(k):
        with mutex:events.append(k)
    print('DEMO/PROXY label shown when applicable. O overhead, F follow, C orbit; P pause, R reset; B ball center; arrows reposition ball; I/J/K/L reposition robot; S screenshot. Close window to exit.')
    with mujoco.viewer.launch_passive(env.models[0],env.data[0],key_callback=key) as viewer:
        def set_camera():
            c=camera(env.field,current,env.data[0].xpos[env.root_id])
            with viewer.lock():
                for a in ('type','azimuth','elevation','distance'):setattr(viewer.cam,a,getattr(c,a))
                viewer.cam.lookat[:]=c.lookat
        set_camera()
        opened=time.perf_counter()
        while viewer.is_running() and (not seconds or time.perf_counter()-opened<seconds):
            start=time.perf_counter()
            with mutex:batch=list(events);events.clear()
            for k in batch:
                try:
                    if k==ord('P'):paused=not paused
                    elif k==ord('R'):env.reset()
                    elif k in (ord('O'),ord('F'),ord('C')):current={ord('O'):'overhead',ord('F'):'follow',ord('C'):'orbit'}[k];set_camera()
                    elif k==ord('B'):env.reposition(ball=[0,0])
                    elif k in (262,263,264,265):
                        xy=env.data[0].qpos[env.bq:env.bq+2].copy();xy+= {262:[.25,0],263:[-.25,0],264:[0,-.25],265:[0,.25]}[k];env.reposition(ball=xy)
                    elif k in (ord('I'),ord('J'),ord('K'),ord('L')):
                        xy=env.data[0].xpos[env.root_id,:2].copy();xy+={ord('I'):[0,.25],ord('K'):[0,-.25],ord('J'):[-.25,0],ord('L'):[.25,0]}[k];env.reposition(robot=xy)
                    elif k==ord('S'):
                        # Screenshot does not step/reset the simulation.
                        import cv2
                        with mujoco.Renderer(env.models[0],height=720,width=1200) as renderer:
                            renderer.update_scene(env.data[0],camera=viewer.cam)
                            path=Path('Artifacts/Results/field01')/f'view_{time.time_ns()}.png';path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(path),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR));print(path)
                except ValueError as exc:print(f'Reposition rejected: {exc}')
            if not paused:
                _,_,term,trunc,_=env.step(env.scripted_action())
                if any(term|trunc):paused=True;print('Episode ended. Press R then P; no hidden reset.')
            if current=='follow':
                with viewer.lock():viewer.cam.lookat[:]=env.data[0].xpos[env.root_id]
            try:
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,env.info['name'],f"{current} | {'PAUSED' if paused else 'RUNNING'} | seed={env.sampler.seed}\n{env.records[0]['values']}")])
            except TypeError:
                # Older viewer bindings still print every value on reset via CLI manifest.
                pass
            viewer.sync();time.sleep(max(0,env.dt-(time.perf_counter()-start)))

