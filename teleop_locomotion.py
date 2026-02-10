"""
Interactive locomotion teleop with keyboard command control.

Controls:
  W / S            vx = +0.25 / -0.25 while key is active
  A / D            vy = +0.25 / -0.25 while key is active
  Left / Right     yaw-rate = +0.25 / -0.25 while key is active
  Space            zero all commands
  R                reset environment
  H                print help

Command limits match isaaclab_env.py:
  vx in [-1.0, 1.0], vy in [-0.5, 0.5], yaw_rate in [-1.0, 1.0]


Run this like this:
mjpython teleop_locomotion.py --policy policies/phase_obs_contact_pen2.pt --key-value 0.5
"""

import argparse
import select
import sys
import termios
import time
import tty

import numpy as np
import torch
import mujoco.viewer

from envs.locomotion_env import HumanoidLocomotionEnv


CMD_LIMITS = np.array([1.0, 0.5, 1.0], dtype=float)  # [vx, vy, yaw_rate]


def _clamp_commands(cmd: np.ndarray) -> np.ndarray:
    return np.clip(cmd, -CMD_LIMITS, CMD_LIMITS)


def _format_cmd(cmd: np.ndarray) -> str:
    return f"vx={cmd[0]:+.3f}, vy={cmd[1]:+.3f}, yaw_rate={cmd[2]:+.3f}"


def _print_controls(key_value: float, key_hold: float):
    print("\n" + "=" * 60)
    print("Teleop Controls")
    print("=" * 60)
    print(f"W/S: vx = +/-{key_value} while key active")
    print(f"A/D: vy = +/-{key_value} while key active")
    print(f"Left/Right: yaw_rate = +/-{key_value} while key active")
    print(f"Key active window: {key_hold:.2f}s (supports hold via terminal key-repeat)")
    print("Space: zero commands")
    print("R: reset env")
    print("Q: quit")
    print("H: print this help")
    print("=" * 60 + "\n")


class _TerminalInput:
    """Non-blocking terminal key reader (WASD + arrow escape sequences)."""

    def __init__(self):
        self.enabled = False
        self._fd = None
        self._old_term = None

    def __enter__(self):
        if not sys.stdin.isatty():
            print("[warn] stdin is not a TTY; terminal key controls disabled.")
            return self

        self._fd = sys.stdin.fileno()
        self._old_term = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self._fd is not None and self._old_term is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
        self.enabled = False

    def _read_arrow_tail(self) -> str:
        seq = ""
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not r:
            return seq
        seq += sys.stdin.read(1)
        if seq != "[":
            return seq
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if r:
            seq += sys.stdin.read(1)
        return seq

    def read_keys(self) -> list[str]:
        if not self.enabled:
            return []

        keys: list[str] = []
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not r:
                break

            ch = sys.stdin.read(1)
            if not ch:
                continue

            if ch == "\x1b":
                tail = self._read_arrow_tail()
                if tail == "[D":
                    keys.append("LEFT")
                elif tail == "[C":
                    keys.append("RIGHT")
                continue

            keys.append(ch)

        return keys


def main():
    parser = argparse.ArgumentParser(description="Interactive keyboard teleop for locomotion_env.py")
    parser.add_argument(
        "--policy",
        type=str,
        default="policies/new_locomotion.pt",
        help="Path to TorchScript policy (.pt).",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default="robot_description/scene.xml",
        help="Path to MuJoCo XML scene.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Control loop frequency. Defaults to 50 Hz.",
    )
    parser.add_argument(
        "--key-value",
        type=float,
        default=0.25,
        help="Per-key command value for vx/vy/yaw-rate.",
    )
    parser.add_argument(
        "--key-hold",
        type=float,
        default=0.12,
        help="Seconds a key remains active after each terminal key event.",
    )
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help="Run with zero action instead of policy output.",
    )
    args = parser.parse_args()

    env = HumanoidLocomotionEnv(xml_path=args.xml_path)
    obs = env.reset()

    cmd = np.zeros(3, dtype=float)
    env._commands[:] = cmd

    policy = None
    if not args.no_policy:
        policy = torch.jit.load(args.policy, map_location="cpu")
        policy.eval()
        with torch.no_grad():
            test_action = policy(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).numpy()
        if test_action.shape != (env._nu,):
            raise ValueError(f"policy action shape {test_action.shape} does not match env action dim {(env._nu,)}")
        print(f"Loaded policy: {args.policy}")
    else:
        print("Running without policy (zero actions).")

    loop_dt = 1.0 / args.fps
    state = {
        "obs": obs,
        "cmd": cmd,
        "reset_requested": False,
        "quit_requested": False,
        "active_until": {"w": 0.0, "s": 0.0, "a": 0.0, "d": 0.0, "LEFT": 0.0, "RIGHT": 0.0},
    }

    def _zero_commands():
        for k in state["active_until"]:
            state["active_until"][k] = 0.0
        state["cmd"][:] = 0.0
        env._commands[:] = state["cmd"]
        print(f"[cmd] {_format_cmd(state['cmd'])}")

    def _request_reset():
        state["reset_requested"] = True

    def _update_cmd_from_active(now: float):
        w = state["active_until"]["w"] > now
        s = state["active_until"]["s"] > now
        a = state["active_until"]["a"] > now
        d = state["active_until"]["d"] > now
        left = state["active_until"]["LEFT"] > now
        right = state["active_until"]["RIGHT"] > now

        new_cmd = np.array(
            [
                args.key_value * (float(w) - float(s)),
                args.key_value * (float(a) - float(d)),
                args.key_value * (float(left) - float(right)),
            ],
            dtype=float,
        )
        new_cmd = _clamp_commands(new_cmd)

        if not np.array_equal(new_cmd, state["cmd"]):
            state["cmd"][:] = new_cmd
            env._commands[:] = state["cmd"]
            print(f"[cmd] {_format_cmd(state['cmd'])}")
        else:
            env._commands[:] = state["cmd"]

    def _handle_key(key: str, now: float):
        k = key.lower() if len(key) == 1 else key
        if k in state["active_until"]:
            state["active_until"][k] = now + args.key_hold
        elif k == " ":
            _zero_commands()
        elif k == "r":
            _request_reset()
        elif k == "h":
            _print_controls(args.key_value, args.key_hold)
        elif k == "q":
            state["quit_requested"] = True

    _print_controls(args.key_value, args.key_hold)
    print(f"[cmd] {_format_cmd(state['cmd'])}")
    print("Launching viewer... keep terminal focused for key controls.")

    with _TerminalInput() as term_input:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            next_tick = time.perf_counter()
            with torch.no_grad():
                while viewer.is_running() and not state["quit_requested"]:
                    now = time.perf_counter()
                    for key in term_input.read_keys():
                        _handle_key(key, now)

                    _update_cmd_from_active(now)

                    if state["reset_requested"]:
                        state["obs"] = env.reset()
                        env._commands[:] = state["cmd"]
                        state["reset_requested"] = False
                        print(f"[reset] {_format_cmd(state['cmd'])}")

                    if now < next_tick:
                        time.sleep(min(0.001, next_tick - now))
                        continue

                    obs_now = state["obs"]

                    if policy is None:
                        action = np.zeros(env._nu, dtype=float)
                    else:
                        obs_tensor = torch.from_numpy(obs_now).float().unsqueeze(0)
                        action = policy(obs_tensor).squeeze(0).numpy()
                        action = np.clip(action, -1.0, 1.0)

                    obs_next = env.step(action)
                    state["obs"] = obs_next

                    viewer.sync()
                    next_tick += loop_dt
                    if now - next_tick > loop_dt:
                        next_tick = now + loop_dt


if __name__ == "__main__":
    main()
