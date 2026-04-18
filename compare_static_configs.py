#!/usr/bin/env python3
"""Static MuJoCo vs IsaacLab environment comparison for sim2sim alignment.

This compares:
- MuJoCo teleop/task defaults from ``teleop_locomotion.py`` and ``envs/locomotion_env.py``
- MuJoCo robot/physics settings from ``robot_description/scene.xml``
- IsaacLab task defaults from ``tritonhumanoid_env.py`` and ``tritonhumanoid_env_cfg.py``
- IsaacLab robot/actuator settings from ``assets/humanoid.py``
- The source URDF used to generate the IsaacLab USD

The goal is not to declare one simulator "correct". It is to make the major
physics- and interface-level mismatches obvious before you run parity rollouts.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


def _func_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_func_name(node.value)}.{node.attr}"
    return ast.dump(node)


def _eval_node(node: ast.AST, ctx: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append(str(_eval_node(value.value, ctx)))
            else:
                raise TypeError(f"unsupported f-string part: {ast.dump(value)}")
        return "".join(parts)
    if isinstance(node, ast.List):
        return [_eval_node(elt, ctx) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt, ctx) for elt in node.elts)
    if isinstance(node, ast.Set):
        return {_eval_node(elt, ctx) for elt in node.elts}
    if isinstance(node, ast.Dict):
        return {_eval_node(k, ctx): _eval_node(v, ctx) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Name):
        if node.id in ctx:
            return ctx[node.id]
        raise KeyError(node.id)
    if isinstance(node, ast.Attribute):
        base = _eval_node(node.value, ctx)
        return getattr(base, node.attr)
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, ctx)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return +val
        raise TypeError(f"unsupported unary op: {ast.dump(node.op)}")
    if isinstance(node, ast.BinOp):
        lhs = _eval_node(node.left, ctx)
        rhs = _eval_node(node.right, ctx)
        if isinstance(node.op, ast.Add):
            return lhs + rhs
        if isinstance(node.op, ast.Sub):
            return lhs - rhs
        if isinstance(node.op, ast.Mult):
            return lhs * rhs
        if isinstance(node.op, ast.Div):
            return lhs / rhs
        if isinstance(node.op, ast.Pow):
            return lhs**rhs
        raise TypeError(f"unsupported bin op: {ast.dump(node.op)}")
    if isinstance(node, ast.Call):
        fname = _func_name(node.func)
        if fname == "np.array":
            if not node.args:
                raise ValueError("np.array call missing data arg")
            return _eval_node(node.args[0], ctx)
        try:
            func_obj = _eval_node(node.func, ctx)
        except Exception:
            func_obj = None
        if callable(func_obj):
            args = [_eval_node(arg, ctx) for arg in node.args]
            kwargs = {kw.arg: _eval_node(kw.value, ctx) for kw in node.keywords if kw.arg is not None}
            return func_obj(*args, **kwargs)
        kwargs = {kw.arg: _eval_node(kw.value, ctx) for kw in node.keywords if kw.arg is not None}
        args = [_eval_node(arg, ctx) for arg in node.args]
        if isinstance(node.func, ast.Attribute):
            try:
                self_obj = _eval_node(node.func.value, ctx)
            except Exception:
                self_obj = _func_name(node.func.value)
            return {
                "__call__": node.func.attr,
                "__self__": self_obj,
                "args": args,
                "kwargs": kwargs,
            }
        return {"__call__": fname, "args": args, "kwargs": kwargs}
    raise TypeError(f"unsupported node: {ast.dump(node)}")


def _load_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _find_class(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise ValueError(f"class {class_name!r} not found")


def _find_function(class_node: ast.ClassDef, func_name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    raise ValueError(f"function {func_name!r} not found in class {class_node.name!r}")


def _class_assignments(path: Path, class_name: str, base_ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    ctx = {"math": math}
    if base_ctx:
        ctx.update(base_ctx)
    out: dict[str, Any] = {}
    class_node = _find_class(_load_ast(path), class_name)
    for stmt in class_node.body:
        target_name = None
        value_node = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target_name = stmt.targets[0].id
            value_node = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            target_name = stmt.target.id
            value_node = stmt.value
        if target_name is None or value_node is None:
            continue
        try:
            value = _eval_node(value_node, ctx)
        except Exception:
            continue
        out[target_name] = value
        ctx[target_name] = value
    return out


def _module_assignments(path: Path, names: set[str]) -> dict[str, Any]:
    tree = _load_ast(path)
    ctx = {"math": math, "Path": Path, "__file__": str(path)}
    out: dict[str, Any] = {}
    for stmt in tree.body:
        target_name = None
        value_node = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target_name = stmt.targets[0].id
            value_node = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            target_name = stmt.target.id
            value_node = stmt.value
        if target_name is None or value_node is None:
            continue
        name = target_name
        try:
            value = _eval_node(value_node, ctx)
        except Exception:
            continue
        ctx[name] = value
        if name in names:
            out[name] = value
    return out


def _init_defaults(path: Path, class_name: str) -> dict[str, Any]:
    class_node = _find_class(_load_ast(path), class_name)
    fn = _find_function(class_node, "__init__")
    args = fn.args.args
    defaults = fn.args.defaults
    if not defaults:
        return {}
    ctx = {"math": math}
    out: dict[str, Any] = {}
    for arg, default in zip(args[-len(defaults):], defaults):
        try:
            out[arg.arg] = _eval_node(default, ctx)
        except Exception:
            continue
    return out


def _self_literal_assignments(path: Path, class_name: str, func_name: str, names: set[str]) -> dict[str, Any]:
    class_node = _find_class(_load_ast(path), class_name)
    fn = _find_function(class_node, func_name)
    ctx = {"math": math}
    out: dict[str, Any] = {}
    for stmt in ast.walk(fn):
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Attribute):
            continue
        if not isinstance(target.value, ast.Name) or target.value.id != "self":
            continue
        if target.attr not in names:
            continue
        try:
            out[target.attr] = _eval_node(stmt.value, ctx)
        except Exception:
            continue
    return out


def _parse_mujoco_scene(path: Path) -> dict[str, Any]:
    root = ET.fromstring(path.read_text())
    out: dict[str, Any] = {}

    option = root.find("option")
    if option is not None:
        out["option"] = dict(option.attrib)

    default_joint = root.find("./default/joint")
    if default_joint is not None:
        out["joint_defaults"] = dict(default_joint.attrib)

    contact_geom = root.find("./default/default[@class='contact_surface']/geom")
    if contact_geom is not None:
        out["contact_surface"] = dict(contact_geom.attrib)

    actuators: dict[str, dict[str, Any]] = {}
    actuator_order: list[str] = []
    for pos in root.findall("./actuator/position"):
        joint = pos.attrib["joint"]
        actuator_order.append(joint)
        actuators[joint] = {
            "name": pos.attrib.get("name"),
            "kp": float(pos.attrib.get("kp", 0.0)),
            "kv": float(pos.attrib.get("kv", 0.0)),
            "forcerange": pos.attrib.get("forcerange"),
        }
    out["actuators"] = actuators
    out["actuator_order"] = actuator_order

    standing = root.find("./keyframe/key[@name='locomotion_standing_pose']")
    if standing is not None:
        ctrl = [float(x) for x in standing.attrib.get("ctrl", "").split()]
        qpos = [float(x) for x in standing.attrib.get("qpos", "").split()]
        out["standing_pose"] = {
            "ctrl": ctrl,
            "qpos": qpos,
            "ctrl_by_joint": {joint: ctrl[i] for i, joint in enumerate(actuator_order) if i < len(ctrl)},
        }

    joints: dict[str, dict[str, Any]] = {}
    for joint in root.findall(".//joint"):
        if "name" not in joint.attrib:
            continue
        name = joint.attrib["name"]
        if name == "root":
            continue
        joints[name] = {
            "axis": joint.attrib.get("axis"),
            "range": joint.attrib.get("range"),
            "pos": joint.attrib.get("pos"),
        }
    out["joints"] = joints

    contact_pairs = []
    for pair in root.findall("./contact/pair"):
        contact_pairs.append(dict(pair.attrib))
    out["contact_pairs"] = contact_pairs

    excludes = []
    for exc in root.findall("./contact/exclude"):
        excludes.append(dict(exc.attrib))
    out["contact_excludes"] = excludes

    return out


def _parse_urdf_joints(path: Path) -> dict[str, Any]:
    root = ET.fromstring(path.read_text())
    joints: dict[str, Any] = {}
    for joint in root.findall("./joint"):
        name = joint.attrib.get("name")
        if not name or joint.attrib.get("type") == "fixed":
            continue
        axis = joint.find("axis")
        limit = joint.find("limit")
        dynamics = joint.find("dynamics")
        joints[name] = {
            "axis": axis.attrib.get("xyz") if axis is not None else None,
            "lower": float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
            "upper": float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
            "effort": float(limit.attrib["effort"]) if limit is not None and "effort" in limit.attrib else None,
            "velocity": float(limit.attrib["velocity"]) if limit is not None and "velocity" in limit.attrib else None,
            "damping": float(dynamics.attrib["damping"]) if dynamics is not None and "damping" in dynamics.attrib else None,
        }
    return joints


def _parse_asset_configs(path: Path) -> dict[str, Any]:
    assigns = _module_assignments(path, {"qz_minus_90", "HUMANOID_CFG", "HUMANOID_LOCOMOTION_DELAYED_PD_CFG"})
    humanoid_cfg = assigns["HUMANOID_CFG"]
    locomotion_cfg = assigns["HUMANOID_LOCOMOTION_DELAYED_PD_CFG"]

    spawn = humanoid_cfg["kwargs"]["spawn"]["kwargs"]
    init_state = humanoid_cfg["kwargs"]["init_state"]["kwargs"]
    base_actuators = humanoid_cfg["kwargs"]["actuators"]
    delayed_actuators = locomotion_cfg["kwargs"]["actuators"]

    out = {
        "usd_path": spawn["usd_path"],
        "rigid_props": spawn["rigid_props"]["kwargs"],
        "articulation_props": spawn["articulation_props"]["kwargs"],
        "init_state": init_state,
        "soft_joint_pos_limit_factor": humanoid_cfg["kwargs"].get("soft_joint_pos_limit_factor"),
        "base_actuators": base_actuators,
        "delayed_actuators": delayed_actuators,
    }
    return out


def _mujoco_summary(root: Path) -> dict[str, Any]:
    teleop_path = root / "mujoco_simulation" / "teleop_locomotion.py"
    env_path = root / "mujoco_simulation" / "envs" / "locomotion_env.py"
    scene_path = root / "mujoco_simulation" / "robot_description" / "scene.xml"

    teleop = _module_assignments(teleop_path, {"CMD_LIMITS"})
    env_consts = _module_assignments(env_path, {"DEFAULT_ACT_DELAY_RANGE_BY_NAME"})
    init_defaults = _init_defaults(env_path, "HumanoidLocomotionEnv")
    init_self = _self_literal_assignments(
        env_path,
        "HumanoidLocomotionEnv",
        "__init__",
        {"_soft_joint_limit_factor", "_ang_vel_scale", "_dof_vel_scale", "_command_yaw_offset"},
    )
    scene = _parse_mujoco_scene(scene_path)

    sim_dt = float(init_defaults["sim_dt"])
    control_dt = float(init_defaults["dt"])
    obs_max = int(init_defaults["obs_max_latency"])
    raw_obs_steps = int(init_defaults["obs_latency_steps"])

    return {
        "command_limits": list(teleop["CMD_LIMITS"]),
        "control_dt": control_dt,
        "sim_dt": sim_dt,
        "substeps_per_control": int(round(control_dt / sim_dt)),
        "frame_stack": int(init_defaults["frame_stack"]),
        "use_phase_obs": bool(init_defaults["use_phase_obs"]),
        "gait_period_s": float(init_defaults["gait_period_s"]),
        "freeze_phase_when_standing": bool(init_defaults["freeze_phase_when_standing"]),
        "stand_phase_lin_threshold": float(init_defaults["stand_phase_lin_threshold"]),
        "stand_phase_yaw_threshold": float(init_defaults["stand_phase_yaw_threshold"]),
        "stand_phase_value": float(init_defaults["stand_phase_value"]),
        "action_scale": float(init_defaults["action_scale"]),
        "action_scale_by_joint": dict(init_defaults["action_scale_by_joint"]),
        "action_noise_std": float(init_defaults["action_noise_std"]),
        "action_smoothing_alpha": float(init_defaults["action_smoothing_alpha"]),
        "act_max_latency": int(init_defaults["act_max_latency"]),
        "act_latency_steps": int(init_defaults["act_latency_steps"]),
        "act_delay_range_by_name_default": dict(env_consts.get("DEFAULT_ACT_DELAY_RANGE_BY_NAME", {})),
        "obs_max_latency": obs_max,
        "obs_latency_steps_raw": raw_obs_steps,
        "obs_latency_steps_effective": int(max(0, min(raw_obs_steps, obs_max))),
        "obs_noise_gravity_std": float(init_defaults["obs_noise_gravity_std"]),
        "obs_noise_gyro_std": float(init_defaults["obs_noise_gyro_std"]),
        "obs_noise_joint_pos_std": float(init_defaults["obs_noise_joint_pos_std"]),
        "obs_noise_joint_vel_std": float(init_defaults["obs_noise_joint_vel_std"]),
        "disturbance_force_max": float(init_defaults["disturbance_force_max"]),
        "disturbance_torque_max": float(init_defaults["disturbance_torque_max"]),
        "disturbance_prob": float(init_defaults["disturbance_prob"]),
        "soft_joint_limit_factor": float(init_self["_soft_joint_limit_factor"]),
        "ang_vel_scale": float(init_self["_ang_vel_scale"]),
        "dof_vel_scale": float(init_self["_dof_vel_scale"]),
        "command_yaw_offset": float(init_self["_command_yaw_offset"]),
        "scene": scene,
    }


def _isaac_summary(root: Path) -> dict[str, Any]:
    cfg_path = (
        root
        / "simulation"
        / "source"
        / "tritonhumanoid"
        / "tritonhumanoid"
        / "tasks"
        / "direct"
        / "tritonhumanoid"
        / "tritonhumanoid_env_cfg.py"
    )
    asset_path = (
        root
        / "simulation"
        / "source"
        / "tritonhumanoid"
        / "tritonhumanoid"
        / "assets"
        / "humanoid.py"
    )
    urdf_path = (
        root
        / "simulation"
        / "source"
        / "tritonhumanoid"
        / "tritonhumanoid"
        / "assets"
        / "human_offset_corrected.urdf"
    )

    events = _class_assignments(cfg_path, "EventCfg")
    cfg = _class_assignments(cfg_path, "HumanoidEnvCfg")
    asset = _parse_asset_configs(asset_path)
    urdf_joints = _parse_urdf_joints(urdf_path)

    sim = cfg["sim"]["kwargs"]
    terrain = cfg["terrain"]["kwargs"]

    control_dt = float(sim["dt"]) * int(cfg["decimation"])

    return {
        "control_dt": control_dt,
        "sim_dt": float(sim["dt"]),
        "substeps_per_control": int(cfg["decimation"]),
        "frame_stack": int(cfg["obs_stack_frames"]),
        "use_phase_obs": bool(cfg["use_phase_obs"]),
        "gait_period_s": float(cfg["gait_period_s"]),
        "randomize_phase": bool(cfg["randomize_phase"]),
        "freeze_phase_when_standing": bool(cfg["freeze_phase_when_standing"]),
        "stand_phase_lin_threshold": float(cfg["stand_phase_lin_threshold"]),
        "stand_phase_yaw_threshold": float(cfg["stand_phase_yaw_threshold"]),
        "stand_phase_value": float(cfg["stand_phase_value"]),
        "action_scale": float(cfg["action_scale"]),
        "action_scale_by_joint": dict(cfg["action_scale_by_joint"]),
        "command_limits": {
            "vx": tuple(cfg["lin_vel_x_range"]),
            "vy": tuple(cfg["lin_vel_y_range"]),
            "yaw_rate": tuple(cfg["ang_vel_yaw_range"]),
        },
        "command_resample_interval_s": float(cfg["command_resample_interval_s"]),
        "stand_prob": float(cfg["stand_prob"]),
        "ang_vel_scale": float(cfg["ang_vel_scale"]),
        "dof_vel_scale": float(cfg["dof_vel_scale"]),
        "joint_pos_obs_noise_std_rad": float(cfg["joint_pos_obs_noise_std_rad"]),
        "joint_vel_obs_noise_std_rad_s": float(cfg["joint_vel_obs_noise_std_rad_s"]),
        "obs_max_latency": int(cfg["obs_max_latency"]),
        "reset_joint_pos_noise": float(cfg["reset_joint_pos_noise"]),
        "reset_joint_vel_noise": float(cfg["reset_joint_vel_noise"]),
        "enable_adr": bool(cfg["enable_adr"]),
        "physics_material": sim["physics_material"]["kwargs"],
        "terrain_material": terrain["physics_material"]["kwargs"],
        "events": {k: v["kwargs"]["params"] for k, v in events.items()},
        "asset": asset,
        "urdf_joints": urdf_joints,
        "command_yaw_offset": float(cfg["command_yaw_offset"]),
    }


@dataclass
class Finding:
    severity: str
    title: str
    detail: str


def _fmt_joint_map(d: dict[str, float]) -> str:
    items = [f"{k}={v:.3f}" for k, v in d.items()]
    return ", ".join(items)


def _parse_float_vec(text: str | None) -> tuple[float, ...] | None:
    if text is None:
        return None
    return tuple(float(x) for x in text.split())


def _compare(mj: dict[str, Any], isaac: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []

    if not math.isclose(mj["control_dt"], isaac["control_dt"], rel_tol=0.0, abs_tol=1e-9):
        findings.append(
            Finding(
                "high",
                "Control dt differs",
                f"MuJoCo control_dt={mj['control_dt']:.6f}s vs Isaac control_dt={isaac['control_dt']:.6f}s.",
            )
        )
    else:
        findings.append(
            Finding(
                "match",
                "Control dt matches",
                f"Both use {mj['control_dt']:.6f}s control steps (50 Hz).",
            )
        )

    if not math.isclose(mj["sim_dt"], isaac["sim_dt"], rel_tol=0.0, abs_tol=1e-9):
        findings.append(
            Finding(
                "high",
                "Physics dt differs by 10x",
                (
                    f"MuJoCo sim_dt={mj['sim_dt']:.6f}s with {mj['substeps_per_control']} substeps/control, "
                    f"Isaac sim_dt={isaac['sim_dt']:.6f}s with {isaac['substeps_per_control']} substeps/control."
                ),
            )
        )

    if mj["action_scale"] == isaac["action_scale"] and mj["action_scale_by_joint"] == isaac["action_scale_by_joint"]:
        findings.append(
            Finding(
                "match",
                "Action scaling matches",
                f"Both use action_scale={mj['action_scale']:.3f} with the same per-joint multipliers.",
            )
        )
    else:
        findings.append(
            Finding(
                "high",
                "Action scaling differs",
                (
                    f"MuJoCo action_scale={mj['action_scale']}, per_joint={mj['action_scale_by_joint']} "
                    f"vs Isaac action_scale={isaac['action_scale']}, per_joint={isaac['action_scale_by_joint']}."
                ),
            )
        )

    if mj["soft_joint_limit_factor"] == isaac["asset"]["soft_joint_pos_limit_factor"]:
        findings.append(
            Finding(
                "match",
                "Soft joint limit factor matches",
                f"Both use 0.95 soft limits.",
            )
        )

    if math.isclose(mj["command_yaw_offset"], isaac["command_yaw_offset"], abs_tol=1e-9):
        findings.append(
            Finding(
                "match",
                "Command frame yaw offset matches",
                f"Both use command_yaw_offset={mj['command_yaw_offset']:.6f} rad.",
            )
        )

    if mj["command_limits"] == [1.0, 0.5, 1.0]:
        cmd_match = (
            isaac["command_limits"]["vx"] == (-1.0, 1.0)
            and isaac["command_limits"]["vy"] == (-0.5, 0.5)
            and isaac["command_limits"]["yaw_rate"] == (-1.0, 1.0)
        )
        if cmd_match:
            findings.append(
                Finding(
                    "match",
                    "Teleop command limits match Isaac ranges",
                    "Teleop uses vx in [-1,1], vy in [-0.5,0.5], yaw_rate in [-1,1].",
                )
            )

    mj_phase = (
        mj["freeze_phase_when_standing"],
        mj["stand_phase_lin_threshold"],
        mj["stand_phase_yaw_threshold"],
        mj["stand_phase_value"],
    )
    isaac_phase = (
        isaac["freeze_phase_when_standing"],
        isaac["stand_phase_lin_threshold"],
        isaac["stand_phase_yaw_threshold"],
        isaac["stand_phase_value"],
    )
    if mj_phase != isaac_phase:
        findings.append(
            Finding(
                "high",
                "Standing phase behavior differs",
                (
                    "MuJoCo defaults are "
                    f"(freeze={mj_phase[0]}, lin={mj_phase[1]}, yaw={mj_phase[2]}, value={mj_phase[3]:.3f}) "
                    "while Isaac uses "
                    f"(freeze={isaac_phase[0]}, lin={isaac_phase[1]}, yaw={isaac_phase[2]}, value={isaac_phase[3]:.3f})."
                ),
            )
        )

    if mj["frame_stack"] == isaac["frame_stack"]:
        findings.append(
            Finding(
                "match",
                "Observation stack depth matches",
                f"Both use {mj['frame_stack']} stacked frames.",
            )
        )

    if not math.isclose(mj["obs_noise_joint_pos_std"], isaac["joint_pos_obs_noise_std_rad"], abs_tol=1e-12) or not math.isclose(
        mj["obs_noise_joint_vel_std"], isaac["joint_vel_obs_noise_std_rad_s"], abs_tol=1e-12
    ):
        findings.append(
            Finding(
                "medium",
                "Fixed observation noise differs",
                (
                    f"MuJoCo defaults joint_pos/joint_vel noise to {mj['obs_noise_joint_pos_std']}/{mj['obs_noise_joint_vel_std']}, "
                    f"while Isaac adds {isaac['joint_pos_obs_noise_std_rad']}/{isaac['joint_vel_obs_noise_std_rad_s']}."
                ),
            )
        )

    base_delay = isaac["asset"]["delayed_actuators"]
    delay_desc = []
    isaac_delay_ranges_simple: dict[str, tuple[int, int]] = {}
    for name, cfg in base_delay.items():
        kwargs = cfg["kwargs"]
        delay_desc.append(f"{name}: {kwargs['min_delay']}..{kwargs['max_delay']} steps")
        joint_names = tuple(kwargs["joint_names_expr"])
        if joint_names == ("left_hip1_joint", "right_hip1_joint"):
            isaac_delay_ranges_simple["hip1_pair"] = (int(kwargs["min_delay"]), int(kwargs["max_delay"]))
        elif joint_names == ("left_hip2_joint", "right_hip2_joint"):
            isaac_delay_ranges_simple["hip2_pair"] = (int(kwargs["min_delay"]), int(kwargs["max_delay"]))
        elif joint_names == ("left_thigh_joint", "right_thigh_joint"):
            isaac_delay_ranges_simple["thigh_pair"] = (int(kwargs["min_delay"]), int(kwargs["max_delay"]))
        elif joint_names == ("left_knee_joint", "right_knee_joint"):
            isaac_delay_ranges_simple["knee_pair"] = (int(kwargs["min_delay"]), int(kwargs["max_delay"]))
        elif joint_names == ("left_ankle_joint", "right_ankle_joint"):
            isaac_delay_ranges_simple["ankle_pair"] = (int(kwargs["min_delay"]), int(kwargs["max_delay"]))
    findings.append(
        Finding(
            "info",
            "Isaac delayed actuator groups",
            ", ".join(delay_desc),
        )
    )

    mj_delay_ranges = mj["act_delay_range_by_name_default"]
    if not mj_delay_ranges:
        findings.append(
            Finding(
                "high",
                "MuJoCo has no actuator delay approximation",
                "Isaac locomotion uses DelayedPD actuators with pair-specific delays, but MuJoCo does not expose matching delay ranges.",
            )
        )
    elif mj_delay_ranges != isaac_delay_ranges_simple:
        findings.append(
            Finding(
                "high",
                "Actuator delay ranges do not match locomotion asset",
                f"MuJoCo delay ranges={mj_delay_ranges} vs Isaac={isaac_delay_ranges_simple}.",
            )
        )
    else:
        findings.append(
            Finding(
                "match",
                "Actuator delay ranges match locomotion asset",
                f"Both use {mj_delay_ranges}.",
            )
        )

    mj_act = mj["scene"]["actuators"]
    isaac_act: dict[str, dict[str, float]] = {}
    for group in isaac["asset"]["delayed_actuators"].values():
        kwargs = group["kwargs"]
        for joint_name in kwargs["joint_names_expr"]:
            isaac_act[joint_name] = {
                "stiffness": float(kwargs["stiffness"][joint_name]),
                "damping": float(kwargs["damping"][joint_name]),
            }

    act_mismatches = []
    for joint_name, mj_cfg in mj_act.items():
        target = isaac_act.get(joint_name)
        if target is None:
            continue
        if not math.isclose(float(mj_cfg["kp"]), target["stiffness"], abs_tol=1e-9) or not math.isclose(
            float(mj_cfg["kv"]), target["damping"], abs_tol=1e-9
        ):
            act_mismatches.append(
                f"{joint_name}: MuJoCo kp/kv={mj_cfg['kp']}/{mj_cfg['kv']} vs Isaac {target['stiffness']}/{target['damping']}"
            )
    if act_mismatches:
        findings.append(
            Finding(
                "high",
                "Actuator gains do not match locomotion asset",
                "; ".join(act_mismatches),
            )
        )

    mj_joint_defaults = mj["scene"]["joint_defaults"]
    frictionloss = float(mj_joint_defaults["frictionloss"])
    armature = float(mj_joint_defaults["armature"])
    if not math.isclose(frictionloss, 0.0, abs_tol=1e-12):
        findings.append(
            Finding(
                "medium",
                "Joint friction differs from Isaac locomotion actuator config",
                f"MuJoCo joint frictionloss={frictionloss} while delayed Isaac actuators use friction=0.0.",
            )
        )
    if not math.isclose(armature, 0.0, abs_tol=1e-12):
        findings.append(
            Finding(
                "medium",
                "Joint armature differs",
                f"MuJoCo armature={armature} while Isaac uses armature=0.0.",
            )
        )

    mj_pose = mj["scene"]["standing_pose"]["ctrl_by_joint"]
    isaac_pose = isaac["asset"]["init_state"]["joint_pos"]
    pose_deltas = {}
    for joint_name, isaac_q in isaac_pose.items():
        if joint_name not in mj_pose:
            continue
        delta = float(mj_pose[joint_name]) - float(isaac_q)
        if abs(delta) > 1e-6:
            pose_deltas[joint_name] = delta
    if pose_deltas:
        findings.append(
            Finding(
                "high",
                "Default standing pose differs",
                (
                    "MuJoCo standing target minus Isaac init pose: "
                    + _fmt_joint_map(pose_deltas)
                    + ". This matters because actions are position offsets around the default pose."
                ),
            )
        )

    mj_joint_map = mj["scene"]["joints"]
    urdf_joints = isaac["urdf_joints"]
    bad_joint_mappings = []
    for joint_name, urdf in urdf_joints.items():
        mj_joint = mj_joint_map.get(joint_name)
        if not mj_joint:
            continue
        mj_range = tuple(float(x) for x in mj_joint["range"].split())
        mj_axis = _parse_float_vec(mj_joint["axis"])
        urdf_axis = _parse_float_vec(urdf["axis"])
        if (
            mj_axis != urdf_axis
            or not math.isclose(mj_range[0], float(urdf["lower"]), abs_tol=1e-6)
            or not math.isclose(mj_range[1], float(urdf["upper"]), abs_tol=1e-6)
        ):
            bad_joint_mappings.append(joint_name)
    if bad_joint_mappings:
        findings.append(
            Finding(
                "high",
                "MuJoCo XML joint definitions differ from URDF/USD source",
                ", ".join(bad_joint_mappings),
            )
        )
    else:
        findings.append(
            Finding(
                "match",
                "MuJoCo joint axes and limits match the URDF/USD source",
                "The scene XML matches the URDF for the actuated joints.",
            )
        )

    mj_contact = mj["scene"]["contact_surface"]
    isaac_mat = isaac["terrain_material"]
    if not math.isclose(float(mj_contact["friction"].split()[0]), float(isaac_mat["static_friction"]), abs_tol=1e-9):
        findings.append(
            Finding(
                "medium",
                "Primary contact friction differs",
                (
                    f"MuJoCo contact friction={mj_contact['friction']} vs Isaac terrain static/dynamic="
                    f"{isaac_mat['static_friction']}/{isaac_mat['dynamic_friction']}."
                ),
            )
        )
    else:
        findings.append(
            Finding(
                "match",
                "Base contact friction is roughly aligned",
                (
                    f"Both nominally use 1.0 primary friction, but MuJoCo still has its own tangential/torsional "
                    f"terms and contact solver settings ({mj_contact['solref']}, {mj_contact['solimp']})."
                ),
            )
        )

    if mj["scene"]["option"].get("integrator") or mj["scene"]["option"].get("cone"):
        findings.append(
            Finding(
                "info",
                "Contact/solver models are fundamentally different",
                (
                    f"MuJoCo uses integrator={mj['scene']['option'].get('integrator')} and cone={mj['scene']['option'].get('cone')}; "
                    f"Isaac uses PhysX with solver iterations "
                    f"{isaac['asset']['articulation_props']['solver_position_iteration_count']}/"
                    f"{isaac['asset']['articulation_props']['solver_velocity_iteration_count']} and "
                    f"max_depenetration_velocity={isaac['asset']['rigid_props']['max_depenetration_velocity']}."
                ),
            )
        )

    if isaac["enable_adr"]:
        findings.append(
            Finding(
                "high",
                "Isaac resets randomize physics but MuJoCo stays deterministic",
                (
                    "Isaac EventCfg randomizes material, gravity, body mass, and joint friction/armature on every reset. "
                    "MuJoCo currently has fixed parameters unless you add equivalent sampling."
                ),
            )
        )

    if mj["disturbance_prob"] == 0.0 and isaac["enable_adr"]:
        findings.append(
            Finding(
                "medium",
                "External disturbances differ",
                "MuJoCo defaults to no pushes/micro-disturbance; Isaac enables push and disturbance curricula through ADR.",
            )
        )

    return findings


def _severity_rank(severity: str) -> int:
    return {
        "high": 0,
        "medium": 1,
        "info": 2,
        "match": 3,
    }.get(severity, 99)


def _print_report(mj: dict[str, Any], isaac: dict[str, Any], findings: list[Finding]) -> None:
    print("MuJoCo vs IsaacLab Static Comparison")
    print("=" * 72)
    print(f"MuJoCo control_dt={mj['control_dt']:.6f}s  sim_dt={mj['sim_dt']:.6f}s  substeps={mj['substeps_per_control']}")
    print(
        f"Isaac  control_dt={isaac['control_dt']:.6f}s  sim_dt={isaac['sim_dt']:.6f}s  substeps={isaac['substeps_per_control']}"
    )
    print()

    for finding in sorted(findings, key=lambda f: (_severity_rank(f.severity), f.title.lower())):
        print(f"[{finding.severity.upper():5}] {finding.title}")
        print(f"        {finding.detail}")
        print()

    print("Recommended MuJoCo Alignment Targets")
    print("-" * 72)
    print("1. Match the locomotion actuator model first:")
    print("   Use the Isaac locomotion delayed-PD gains and approximate the per-pair delays.")
    print("2. Match the default pose second:")
    print("   Make the MuJoCo standing keyframe use the same joint targets as Isaac.")
    print("3. Decide whether you want nominal parity or training-distribution parity:")
    print("   Nominal parity: disable/randomization-ignore on Isaac during evaluation.")
    print("   Training-distribution parity: add reset-time randomization to MuJoCo.")
    print("4. Only after that tune contact:")
    print("   Friction is already close; solver/contact response still needs rollout-based tuning.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MuJoCo locomotion config against IsaacLab locomotion config.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to save the parsed summary and findings.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    mj = _mujoco_summary(root)
    isaac = _isaac_summary(root)
    findings = _compare(mj, isaac)

    _print_report(mj, isaac, findings)

    if args.json_out is not None:
        payload = {
            "mujoco": mj,
            "isaac": isaac,
            "findings": [finding.__dict__ for finding in findings],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
