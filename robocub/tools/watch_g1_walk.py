"""Watch the saved T02 policy walk the Unitree G1 live in the MuJoCo viewer.

Press Run. After about a minute of compilation on the first launch (about 40
seconds on later launches), a viewer window opens and the robot tours its commands (forward, turns, sidesteps, backward,
stand) in a loop. The camera follows the robot from behind and turns with it.
Click the viewer window, then use these keys:

    Up / Down      walk forward / backward
    Left / Right   turn left / right
    End            stop (stand still)
    Enter          start walking again (resume the automatic tour)
    Home           bring the camera back behind the robot
    Backspace      reset the robot to its starting pose

Avoid Space and Esc: the MuJoCo viewer uses them to pause the display and to
detach the camera. Mouse drag and scroll still orbit and zoom the camera.
If the robot falls, it resets after a short pause. Close the window to exit.
This is the simulation-only development prototype described in
Documentation/Guides/G1_CONTROLLER.md; it is not a hardware controller.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import threading
import time
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# IDE run buttons may start in scripts/; the controller uses repo-relative paths.
os.chdir(PROJECT_ROOT)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Reuse compiled policy code across runs so only the first launch is slow.
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "jax_compilation_cache")
)
# One small policy on CPU runs faster single-threaded and leaves cores for the viewer.
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
# Harmless float cast warning emitted by JAX while restoring the policy.
warnings.filterwarnings("ignore", message="overflow encountered in cast")

# (label, (vx m/s, vy m/s, yaw rad/s), seconds); values from final_f1_commands.json.
TOUR = [
    ("stand", (0.0, 0.0, 0.0), 2.0),
    ("forward", (0.45, 0.0, 0.0), 6.0),
    ("turn left", (0.0, 0.0, 0.4), 5.0),
    ("forward", (0.45, 0.0, 0.0), 4.0),
    ("turn right", (0.0, 0.0, -0.4), 5.0),
    ("step left", (0.0, 0.25, 0.0), 4.0),
    ("step right", (0.0, -0.25, 0.0), 4.0),
    ("backward", (-0.25, 0.0, 0.0), 4.0),
    ("stand", (0.0, 0.0, 0.0), 3.0),
]

# GLFW key codes. Space, Esc, Tab, [ ], + -, F1-F5, letters and digits are
# already bound by the MuJoCo viewer, so the controls avoid them.
KEY_ENTER, KEY_BACKSPACE = 257, 259
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_HOME, KEY_END = 268, 269
MANUAL_KEYS = {
    KEY_UP: ("forward", (0.45, 0.0, 0.0)),
    KEY_DOWN: ("backward", (-0.25, 0.0, 0.0)),
    KEY_LEFT: ("turn left", (0.0, 0.0, 0.4)),
    KEY_RIGHT: ("turn right", (0.0, 0.0, -0.4)),
    KEY_END: ("stand", (0.0, 0.0, 0.0)),
}
FALL_PAUSE_SECONDS = 2.0
VIEWER_FPS = 60.0
# Home camera: behind and slightly to the left of the robot, looking along its heading.
CAMERA_DISTANCE, CAMERA_ELEVATION, CAMERA_SIDE_OFFSET = 3.0, -15.0, -30.0


class Pilot:
    """Chooses the current command: the looping tour or the last arrow key."""

    def __init__(self) -> None:
        self.manual: tuple[str, tuple[float, float, float]] | None = None
        self.tour_start = time.perf_counter()

    def restart_tour(self, skip_stand: bool = False) -> None:
        self.manual = None
        self.tour_start = time.perf_counter()
        if skip_stand:  # Start at the first walking segment instead of standing.
            self.tour_start -= TOUR[0][2]

    def command(self) -> tuple[str, tuple[float, float, float]]:
        if self.manual is not None:
            return self.manual
        elapsed = (time.perf_counter() - self.tour_start) % sum(s for *_, s in TOUR)
        for label, cmd, seconds in TOUR:
            if elapsed < seconds:
                return label, cmd
            elapsed -= seconds
        return TOUR[-1][0], TOUR[-1][1]


def simulate(keys: mp.Queue, frames: mp.Queue) -> None:
    """Child process: run the policy at real time and stream qpos to the viewer.

    The policy runs in its own process because sharing one interpreter with the
    viewer's render thread slows each JAX step to slower than real time.
    """

    import numpy as np

    from robocub.locomotion.unitree_g1.simulation_controller import (
        CHECKPOINT,
        DEFAULT_RUN,
        SimulationController,
    )

    print(f"Loading T02 policy (checkpoint {CHECKPOINT}) from {DEFAULT_RUN} ...", flush=True)
    try:
        robot = SimulationController(PROJECT_ROOT / DEFAULT_RUN)
    except FileNotFoundError as error:
        frames.put(("error", str(error)))
        return
    print("Compiling the policy (first steps take a while) ...", flush=True)
    robot.reset(seed=6000)
    # The first two steps each trigger a JIT compile; finish both before the
    # viewer opens so playback starts at real time.
    for _ in range(3):
        robot.step().data.qpos.block_until_ready()
    episode = 0
    robot.reset(seed=6000)
    frames.put(("model", robot.env.mj_model))

    pilot = Pilot()
    last_label = None
    next_tick = time.perf_counter()
    while True:
        try:
            while True:
                key = keys.get_nowait()
                if key is None:
                    return
                if key in MANUAL_KEYS:
                    pilot.manual = MANUAL_KEYS[key]
                elif key == KEY_ENTER:
                    pilot.restart_tour(skip_stand=True)
                    last_label = None
                elif key == KEY_BACKSPACE:
                    episode += 1
                    robot.reset(seed=6000 + episode)
                    pilot.restart_tour()
                    last_label = None
                    print("Reset; automatic tour restarted.", flush=True)
        except queue.Empty:
            pass

        label, cmd = pilot.command()
        if label != last_label:
            mode = "manual" if pilot.manual else "tour"
            print(f"[{mode}] {label}: vx={cmd[0]} vy={cmd[1]} yaw={cmd[2]}", flush=True)
            last_label = label
        robot.set_command(*cmd)
        state = robot.step()
        frames.put(("qpos", np.asarray(state.data.qpos)))

        if bool(state.done):
            print(f"Robot fell during '{label}'; resetting in {FALL_PAUSE_SECONDS:.0f}s.", flush=True)
            time.sleep(FALL_PAUSE_SECONDS)
            episode += 1
            robot.reset(seed=6000 + episode)
            last_label = None
            next_tick = time.perf_counter()
            continue

        # Pace to real time: one policy step is robot.dt seconds of simulation.
        next_tick += robot.dt
        delay = next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.perf_counter()


def heading_degrees(qpos) -> float:
    """Yaw of the floating base from its (w, x, y, z) quaternion at qpos[3:7]."""
    w, x, y, z = qpos[3:7]
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def main() -> None:
    import mujoco
    import mujoco.viewer

    keys: mp.Queue = mp.Queue()
    frames: mp.Queue = mp.Queue()
    sim = mp.Process(target=simulate, args=(keys, frames), daemon=True)
    sim.start()

    kind, payload = frames.get()
    if kind == "error":
        sim.join()
        raise SystemExit(payload)
    model = payload
    data = mujoco.MjData(model)
    print(__doc__.split("\n\n", 1)[1], flush=True)

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    camera_home = threading.Event()
    camera_home.set()

    def on_key(key: int) -> None:
        # Runs on the viewer thread; the camera belongs to this process.
        if key == KEY_HOME:
            camera_home.set()
        else:
            keys.put(key)

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
            last_heading = None
            while viewer.is_running() and sim.is_alive():
                qpos = None
                try:
                    while True:  # Draw only the newest frame.
                        qpos = frames.get_nowait()[1]
                except queue.Empty:
                    pass
                if qpos is not None:
                    heading = heading_degrees(qpos)
                    with viewer.lock():
                        data.qpos[:] = qpos
                        mujoco.mj_forward(model, data)
                        cam = viewer.cam
                        if camera_home.is_set():
                            camera_home.clear()
                            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                            cam.trackbodyid = pelvis
                            cam.distance, cam.elevation = CAMERA_DISTANCE, CAMERA_ELEVATION
                            cam.azimuth = heading + CAMERA_SIDE_OFFSET
                        elif cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING and last_heading is not None:
                            # Turn the camera with the robot, keeping any mouse orbit offset.
                            cam.azimuth += (heading - last_heading + 180.0) % 360.0 - 180.0
                    last_heading = heading
                    viewer.sync()
                time.sleep(1.0 / VIEWER_FPS)
    finally:
        keys.put(None)
        sim.join(timeout=5)
        if sim.is_alive():
            sim.terminate()
    if sim.exitcode not in (0, None):
        raise SystemExit(f"Simulation process exited with code {sim.exitcode}.")


if __name__ == "__main__":
    main()
