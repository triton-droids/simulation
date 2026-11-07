"""Live T02 walking on a visual Field #1, preserving verified MuJoCo physics.

Goals and ball are visual only in this walking demo, not soccer interaction.
Arrow keys steer; End stands; Enter resumes tour; Backspace resets.
Home follows the robot; F6 shows the whole field. Close the window to quit.
"""
from pathlib import Path
import argparse
import multiprocessing as mp
import queue
import time
import json
import math
import _g1_walk_runtime as walk


def decorate(scene):
    import mujoco
    import numpy as np
    from robocub.soccer.field.geometry import Field, geometry
    shapes = geometry(Field())
    if scene.ngeom + len(shapes) + 1 > scene.maxgeom:
        raise RuntimeError('Viewer geometry capacity too small')
    for shape in shapes:
        if shape['name'] == 'turf':
            continue  # Existing physical floor is colored green; avoid coplanar flicker.
        yaw = shape['yaw']; c, s = math.cos(yaw), math.sin(yaw)
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray(shape['size']) / 2, np.asarray(shape['pos']),
            np.array([c,-s,0,s,c,0,0,0,1.]), np.asarray(shape['color']))
        scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([Field().ball_radius]*3), np.array([1.,1.,Field().ball_radius]),
        np.eye(3).ravel(), np.array([1.,1.,1.,1.]))
    scene.ngeom += 1


def main():
    import mujoco
    import mujoco.viewer
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, help='Record a bounded 12-second demo instead of opening a window')
    parser.add_argument('--view_seconds', type=float, default=0, help='Optional automatic viewer close for smoke tests')
    args = parser.parse_args()
    keys, frames = mp.Queue(), mp.Queue()
    worker = mp.Process(target=walk.simulate, args=(keys, frames), daemon=True)
    worker.start()
    print('Loading learned T02 G1 policy. First launch compiles before opening the viewer.', flush=True)
    try:
        while True:
            try: kind, model = frames.get(timeout=1); break
            except queue.Empty:
                if not worker.is_alive(): raise RuntimeError('Walking controller failed; see console output')
        if kind == 'error': raise RuntimeError(model)
        if kind == 'model_binary':
            import uuid
            cache=walk.PROJECT_ROOT/'.cache'; cache.mkdir(exist_ok=True)
            binary=cache/f'g1_field_{uuid.uuid4().hex}.mjb'
            try:
                binary.write_bytes(model)
                model=mujoco.MjModel.from_binary_path(str(binary))
            finally:
                binary.unlink(missing_ok=True)
        data = mujoco.MjData(model)
        model.vis.global_.offwidth = 1280; model.vis.global_.offheight = 720
        floor = model.geom('floor').id
        model.geom_matid[floor] = -1
        model.geom_rgba[floor] = [.12,.32,.12,1.]
        print(__doc__, flush=True)
        camera = mujoco.MjvCamera()
        def aim(cam, overview=False):
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0,0,0] if overview else data.qpos[:3]
            cam.distance = 19 if overview else 4.5
            cam.elevation = -75 if overview else -22
            cam.azimuth = 90 if overview else 135
        if args.record:
            import cv2
            args.record.mkdir(parents=True, exist_ok=False)
            writer = cv2.VideoWriter(str(args.record/'walk.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280,720))
            if not writer.isOpened(): raise RuntimeError('Video encoder unavailable')
            rows=[]; start=None; last=None
            with mujoco.Renderer(model, height=720, width=1280, max_geom=2000) as renderer:
                try:
                    while len(rows) < 600:
                        kind, q = frames.get(timeout=120)
                        if kind != 'qpos': raise RuntimeError(str(q))
                        data.qpos[:]=q; mujoco.mj_forward(model,data)
                        now=time.perf_counter()
                        if start is None: start=now
                        rows.append({'t':len(rows)*.02,'root':q[:3].tolist()})
                        if (len(rows)-1)%5==0:
                            aim(camera); renderer.update_scene(data,camera); decorate(renderer.scene)
                            rgb=renderer.render(); writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)); last=now
                            if len(rows)==1: cv2.imwrite(str(args.record/'follow.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                    aim(camera,True); renderer.update_scene(data,camera); decorate(renderer.scene)
                    cv2.imwrite(str(args.record/'field.png'),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
                finally: writer.release()
            (args.record/'trajectory.json').write_text(json.dumps(rows,indent=2))
            summary={'backend':'MuJoCo T02 learned policy; visual field overlay', 'samples':len(rows),
                'min_root_height':min(r['root'][2] for r in rows),
                'xy_displacement':float(np.linalg.norm(np.array(rows[-1]['root'][:2])-rows[0]['root'][:2])),
                'limitations':'Goals and ball are visual only; tour may automatically reset after a fall.'}
            (args.record/'summary.json').write_text(json.dumps(summary,indent=2)); print(summary)
        else:
            overview=[False]
            def key(k):
                if k==295: overview[0]=True  # F6
                elif k==walk.KEY_HOME: overview[0]=False
                else: keys.put(k)
            with mujoco.viewer.launch_passive(model,data,key_callback=key) as viewer:
                decorate(viewer.user_scn)
                opened=time.perf_counter()
                while viewer.is_running() and worker.is_alive() and (not args.view_seconds or time.perf_counter()-opened<args.view_seconds):
                    latest=None
                    try:
                        while True: latest=frames.get_nowait()[1]
                    except queue.Empty: pass
                    if latest is not None:
                        with viewer.lock():
                            data.qpos[:]=latest; mujoco.mj_forward(model,data); aim(viewer.cam,overview[0])
                        viewer.sync()
                    time.sleep(1/60)
    finally:
        keys.put(None); worker.join(timeout=5)
        if worker.is_alive(): worker.terminate(); worker.join(timeout=5)


if __name__=='__main__':
    main()
