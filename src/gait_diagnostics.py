"""Read-only, 50 Hz gait measurements for a single-environment replay."""

import math
import statistics


class GaitDiagnostics:
    def __init__(self, env, driven_joints):
        # Keep all Kit imports after SimulationApp boot.
        import omni.usd
        import torch
        from pxr import Gf, UsdGeom

        self.env = env
        self.robot = env.scene["robot"]
        self.driven_joints = driven_joints
        self.joint_ids = [self.robot.joint_names.index(n) for n in driven_joints]
        self.points = {}
        self.body_ids = {}
        self.records = []
        self.ended = False
        stage = omni.usd.get_context().get_stage()
        for side in ("left", "right"):
            name = f"{side}_foot"
            sensor = env.scene[f"{name}_contact"]
            if list(sensor.body_names) != [name]:
                raise ValueError(f"Unexpected sensor bodies: {sensor.body_names}")
            self.body_ids[side] = self.robot.body_names.index(name)
            path = f"/World/envs/env_0/Robot/Geometry/hip/{side}_leg1/{side}_leg2/{side}_leg3/{side}_leg4/{name}/{name}_collision_box"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise ValueError(f"Missing sole collider: {path}")
            transform = UsdGeom.Xformable(prim).GetLocalTransformation()
            half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
            self.points[side] = torch.tensor([
                list(transform.Transform(Gf.Vec3d(x * half, y * half, -half)))
                for x in (-1, 1) for y in (-1, 1)
            ], device=env.device, dtype=torch.float32)

    def sample(self, t, raw_actions, done):
        import torch
        from locomotion_rewards import _tensor, lateral_sole_spacing

        # Do not mix a reset pose with the action/forces of the previous episode.
        self.ended |= bool(done[0])
        if self.ended:
            return
        data = self.robot.data
        feet = {}
        for side, points in self.points.items():
            body = self.body_ids[side]
            quat = _tensor(data.body_link_quat_w)[0, body]
            xyz = quat[:3].expand_as(points)
            uv = 2 * torch.cross(xyz, points, dim=-1)
            offsets = points + quat[3] * uv + torch.cross(xyz, uv, dim=-1)
            world = offsets + _tensor(data.body_link_pos_w)[0, body]
            center_offset = offsets.mean(dim=0)
            velocity = _tensor(data.body_link_lin_vel_w)[0, body] + torch.cross(
                _tensor(data.body_link_ang_vel_w)[0, body], center_offset, dim=-1
            )
            force = _tensor(self.env.scene[f"{side}_foot_contact"].data.net_forces_w)[0, 0]
            feet[side] = {
                "force_world_N": force.tolist(),
                "contact": bool(force.norm() > 1.0),
                "sole_corners_z_m": world[:, 2].tolist(),
                "sole_center_world_m": world.mean(dim=0).tolist(),
                "sole_center_velocity_world_mps": velocity.tolist(),
            }
        term = self.env.action_manager.get_term("joint_pos")
        self.records.append({
            "t_s": t, "feet": feet,
            "lateral_sole_spacing_m": float(lateral_sole_spacing(self.env)[0]),
            "raw_actions": raw_actions[0].tolist(),
            "applied_actions": self.env.action_manager.action[0].tolist(),
            "joint_targets_rad": term.processed_actions[0].tolist(),
            "joint_positions_rad": data.joint_pos[0, self.joint_ids].tolist(),
        })

    def report(self):
        return {
            "sample_rate_hz": 1 / self.env.step_dt,
            "driven_joint_names": self.driven_joints,
            "scope": "First episode only; 50 Hz end-of-control-step net foot forces, not physics-rate impulses. Flat ground z=0. Clearance is lowest sole collider corner.",
            "summary": summarize(self.records, self.driven_joints, self.env.step_dt),
            "records": self.records,
        }


def summarize(records, names, dt, warmup=2.0):
    rows = [r for r in records if r["t_s"] > warmup]
    if not rows:
        return {"samples": 0}
    mean = statistics.mean
    summary = {"samples": len(rows), "start_s": rows[0]["t_s"], "end_s": rows[-1]["t_s"], "feet": {}, "joints": {}}
    if all("lateral_sole_spacing_m" in r for r in rows):
        widths = [r["lateral_sole_spacing_m"] for r in rows]
        summary["lateral_sole_spacing_m"] = {"min": min(widths), "mean": mean(widths), "max": max(widths)}
    loads = {}
    for side in ("left", "right"):
        feet = [r["feet"][side] for r in rows]
        contact = [f["contact"] for f in feet]
        loads[side] = sum(max(0, f["force_world_N"][2]) for f in feet)
        # Only complete, internally bounded swing intervals; reject window edges.
        swings = []
        i = 1
        while i < len(feet):
            if contact[i - 1] and not contact[i]:
                start = i
                while i < len(feet) and not contact[i]:
                    i += 1
                if i < len(feet):
                    a, b = feet[start - 1]["sole_center_world_m"], feet[i]["sole_center_world_m"]
                    swings.append({
                        "duration_s": (i - start) * dt,
                        "peak_min_corner_clearance_m": max(min(f["sole_corners_z_m"]) for f in feet[start:i]),
                        "horizontal_advance_m": math.hypot(b[0] - a[0], b[1] - a[1]),
                    })
            i += 1
        stance_speed = [math.hypot(*f["sole_center_velocity_world_mps"][:2]) for f in feet if f["contact"]]
        summary["feet"][side] = {
            "contact_fraction": mean(contact),
            "mean_vertical_force_N": loads[side] / len(feet),
            "max_min_corner_clearance_m": max(min(f["sole_corners_z_m"]) for f in feet),
            "mean_stance_sole_speed_mps": mean(stance_speed) if stance_speed else None,
            "complete_swings": swings,
        }
    total = sum(loads.values())
    for side in loads:
        summary["feet"][side]["sampled_vertical_load_share"] = loads[side] / total if total else None
    summary["double_support_fraction"] = mean(r["feet"]["left"]["contact"] and r["feet"]["right"]["contact"] for r in rows)
    for i, name in enumerate(names):
        raw = [r["raw_actions"][i] for r in rows]
        applied = [r["applied_actions"][i] for r in rows]
        targets = [r["joint_targets_rad"][i] for r in rows]
        actual = [r["joint_positions_rad"][i] for r in rows]
        summary["joints"][name] = {
            "raw_action_min_max": [min(raw), max(raw)],
            "upper_clip_fraction": mean(a > 1 for a in raw),
            "lower_clip_fraction": mean(a < -1 for a in raw),
            "applied_action_min_max": [min(applied), max(applied)],
            "target_min_max_rad": [min(targets), max(targets)],
            "actual_min_max_rad": [min(actual), max(actual)],
        }
    return summary
