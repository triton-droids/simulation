#!/usr/bin/env python3
"""Export td_perception BallState or Triton PointStamped from a ROS2 bag.

Run in a sourced ROS2 environment. The evaluator itself does not require ROS.
"""
import argparse
import csv
import json
import math
from pathlib import Path

STATUS = {0: "tentative", 1: "tracking", 2: "coasting", 3: "lost"}


def convert_message(msg, kind, matrix=None):
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    pos = msg.position if kind == "td" else msg.point
    p = [pos.x, pos.y, pos.z]
    if not all(math.isfinite(v) for v in p):
        raise ValueError("nonfinite message position")
    if matrix is not None:
        p = [sum(matrix[i][j] * p[j] for j in range(3)) + matrix[i][3] for i in range(3)]
    row = {"timestamp_s": t, "status": STATUS[msg.status] if kind == "td" else "tracking",
           "x_m": p[0], "y_m": p[1], "vx_mps": "", "vy_mps": ""}
    if kind == "td":
        v = [msg.velocity.x, msg.velocity.y, msg.velocity.z]
        if matrix is not None:
            v = [sum(matrix[i][j] * v[j] for j in range(3)) for i in range(3)]
        row["vx_mps"], row["vy_mps"] = v[:2]
    return row


def validate_transform(cfg):
    m = cfg["matrix"]
    if len(m) != 4 or any(len(r) != 4 for r in m):
        raise ValueError("transform must be a 4x4 matrix")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for r in m for v in r):
        raise ValueError("transform contains invalid numbers")
    if any(abs(a - b) > 1e-6 for a, b in zip(m[3], [0, 0, 0, 1])):
        raise ValueError("invalid homogeneous last row")
    for i in range(3):
        for j in range(3):
            dot = sum(m[k][i] * m[k][j] for k in range(3))
            if abs(dot - int(i == j)) > 1e-5:
                raise ValueError("rotation must be orthonormal (no scale)")
    det = (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
           - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
           + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    if abs(det - 1) > 1e-5:
        raise ValueError("rotation must have determinant +1")
    if not cfg.get("source_frame") or not cfg.get("target_frame"):
        raise ValueError("transform needs source_frame and target_frame")
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag")
    parser.add_argument("--kind", choices=["td", "triton"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--topic")
    parser.add_argument("--storage-id", default="", help="Optional sqlite3 or mcap; otherwise autodetect")
    parser.add_argument("--transform", type=Path, help="Optional fixed source→ground-frame rigid transform JSON")
    args = parser.parse_args()
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        parser.exit(2, "Run this exporter inside a sourced ROS2 environment. Source td_msgs for --kind td.\n")
    try:
        cfg = json.loads(args.transform.read_text()) if args.transform else None
        matrix = validate_transform(cfg) if cfg else None
        topic = args.topic or ("/ball/state" if args.kind == "td" else "/vision/ball_3d_position")
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id=args.storage_id), rosbag2_py.ConverterOptions("", ""))
        topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
        expected = "td_msgs/msg/BallState" if args.kind == "td" else "geometry_msgs/msg/PointStamped"
        if topics.get(topic) != expected:
            raise ValueError(f"expected {topic} with type {expected}; found {topics.get(topic)}")
        cls = get_message(expected)
        rows, frames = [], set()
        while reader.has_next():
            name, data, _ = reader.read_next()
            if name != topic:
                continue
            msg = deserialize_message(data, cls)
            frame = msg.header.frame_id
            if not frame or cfg and frame != cfg["source_frame"]:
                raise ValueError(f"missing/unexpected source frame: {frame!r}")
            frames.add(frame)
            rows.append(convert_message(msg, args.kind, matrix))
        if len(frames) != 1 or not rows:
            raise ValueError("need messages in exactly one frame")
        # Keep header timestamps: never silently sort/drop duplicate or out-of-order states.
        if any(b["timestamp_s"] <= a["timestamp_s"] for a, b in zip(rows, rows[1:])):
            raise ValueError("non-increasing header stamps; select a monotonic segment before export")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        target = cfg["target_frame"] if cfg else next(iter(frames))
        metadata = {"frame": target, "source_topic": topic, "kind": args.kind,
                    "ground_plane": bool(cfg) or target in {"odom", "base_footprint", "map"},
                    "note": "No publication latency inferred from bag receipt times. Fixed transforms only; moving-head TF must be applied separately."}
        args.out.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Exported {len(rows)} states in {target} to {args.out}")
        if not metadata["ground_plane"]:
            print("Optical/non-ground frame: apply a calibrated ground-frame transform before XY scoring.")
    except (ValueError, KeyError, OSError, RuntimeError) as e:
        parser.exit(2, f"Export failed: {e}\n")


if __name__ == "__main__":
    main()
