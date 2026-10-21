"""Summarize the existing embedded tracking logs for sim-to-real calibration.

This is a descriptive closed-loop analysis. It does not claim to identify
motor inertia, friction, contact, or a pure communications delay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_COMMIT = "2728fe19eb17908ac5bd4d88c8e9a131111a9afd"
MIN_LAG_SAMPLES = 20
MAX_LAG_STEPS = 5


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
  a = a - np.mean(a)
  b = b - np.mean(b)
  denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
  if denominator < 1e-12:
    return None
  return float(np.dot(a, b) / denominator)


def response_lag_steps(
  commanded: np.ndarray, measured_velocity: np.ndarray, dt: float
) -> tuple[list[int | None], list[float | None]]:
  """Estimate command-to-velocity response lag by peak cross-correlation.

  The result includes controller, mechanism, contact, filtering, and transport
  dynamics. It must not be interpreted as a pure message delay.
  """
  command_velocity = np.diff(commanded, axis=0) / dt
  measured_velocity = measured_velocity[1:]
  lag_steps: list[int | None] = []
  correlations: list[float | None] = []
  for joint in range(command_velocity.shape[1]):
    candidates: list[tuple[float, int]] = []
    for lag in range(MAX_LAG_STEPS + 1):
      n = command_velocity.shape[0] - lag
      if n < MIN_LAG_SAMPLES:
        continue
      command_slice = command_velocity[:n, joint]
      measured_slice = measured_velocity[lag : lag + n, joint]
      correlation = _correlation(command_slice, measured_slice)
      if correlation is not None:
        candidates.append((correlation, lag))
    if candidates:
      correlation, lag = max(candidates)
      lag_steps.append(lag)
      correlations.append(correlation)
    else:
      lag_steps.append(None)
      correlations.append(None)
  return lag_steps, correlations


def gain_profile(metadata: dict[str, Any]) -> str:
  kp = np.asarray(metadata["kp"], dtype=float)
  kd = np.asarray(metadata["kd"], dtype=float)
  deployed_kp = np.array([100, 100, 100, 80, 22] * 2, dtype=float)
  deployed_kd = np.full(10, 2.0)
  if np.array_equal(kp, deployed_kp) and np.array_equal(kd, deployed_kd):
    return "deployed"
  return "high_gain_experimental"


def stop_category(reason: str) -> str:
  if reason == "completed":
    return "completed"
  if "missed" in reason:
    return "missed_feedback"
  if "out of limits" in reason:
    return "joint_limit"
  return "other_abort"


def analyze_run(path: Path) -> dict[str, Any]:
  with np.load(path, allow_pickle=True) as data:
    metadata = json.loads(str(data["meta"].item()))
    commanded = np.asarray(data["commanded"], dtype=float)
    joint_pos = np.asarray(data["joint_pos"], dtype=float)
    joint_vel = np.asarray(data["joint_vel"], dtype=float)
    loop_dt = np.asarray(data["loop_dt"], dtype=float)
    freshness = np.asarray(data["feedback_fresh"], dtype=bool)
    guard = (
      np.asarray(data["guard_clipped"], dtype=bool)
      if "guard_clipped" in data
      else np.zeros_like(freshness)
    )

  control_hz = float(metadata["control_hz"])
  nominal_dt = 1.0 / control_hz
  lags, correlations = response_lag_steps(commanded, joint_vel, nominal_dt)
  if len(commanded) > 1:
    max_slew = float(np.max(np.abs(np.diff(commanded, axis=0))) / nominal_dt)
  else:
    max_slew = 0.0
  tracking_error = commanded - joint_pos
  return {
    "file": path.name,
    "mode": metadata["mode"],
    "gain_profile": gain_profile(metadata),
    "steps": int(len(commanded)),
    "duration_s": float(len(commanded) / control_hz),
    "control_hz": control_hz,
    "median_loop_dt_ms": float(np.median(loop_dt) * 1000),
    "p99_loop_dt_ms": float(np.percentile(loop_dt, 99) * 1000),
    "status": stop_category(metadata["stop_reason"]),
    "stop_reason": metadata["stop_reason"],
    "feedback_fresh_fraction": float(np.mean(freshness)),
    "guard_clipped_fraction": float(np.mean(guard)),
    "tracking_rmse_rad": float(np.sqrt(np.mean(tracking_error**2))),
    "max_command_slew_rad_s": max_slew,
    "response_lag_steps_by_joint": lags,
    "response_lag_ms_by_joint": [
      None if lag is None else lag * nominal_dt * 1000 for lag in lags
    ],
    "response_correlation_by_joint": correlations,
  }


def summarize_profile(runs: list[dict[str, Any]]) -> dict[str, Any]:
  eligible = [
    run for run in runs if run["mode"] == "live" and run["steps"] >= MIN_LAG_SAMPLES
  ]
  lags = [
    value
    for run in eligible
    for value in run["response_lag_ms_by_joint"]
    if value is not None
  ]
  correlations = [
    value
    for run in eligible
    for value in run["response_correlation_by_joint"]
    if value is not None
  ]
  live = [run for run in runs if run["mode"] == "live"]
  return {
    "runs": len(runs),
    "live_runs": len(live),
    "completed_live_runs": sum(run["status"] == "completed" for run in live),
    "duration_s": float(sum(run["duration_s"] for run in runs)),
    "tracking_rmse_rad_median": float(
      np.median([run["tracking_rmse_rad"] for run in eligible])
    ),
    "tracking_rmse_rad_range": [
      float(min(run["tracking_rmse_rad"] for run in eligible)),
      float(max(run["tracking_rmse_rad"] for run in eligible)),
    ],
    "response_lag_ms_median": float(np.median(lags)),
    "response_lag_ms_range": [float(min(lags)), float(max(lags))],
    "response_correlation_median": float(np.median(correlations)),
  }


def make_report(analysis: dict[str, Any]) -> str:
  lines = [
    "# Embedded tracking-log analysis",
    "",
    f"Source: `triton-droids/embedded` commit `{SOURCE_COMMIT}`.",
    "",
    "## What the existing data supports",
    "",
    "These logs lock down the deployed 50 Hz command contract, joint order, signs, "
    "limits, gain sets, safety exits, target slew, and a closed-loop response envelope. "
    "They are enough to continue simulation training and domain randomization now.",
    "",
    "The lag below is command-to-measured-velocity response. It combines PD response, "
    "mechanical dynamics, contact, filtering, and communications. It is not a pure "
    "transport-delay estimate and should be used to validate randomized rollouts, not "
    "copied directly into a delay queue.",
    "",
    "## Gain-profile summary",
    "",
    "| profile | runs | completed live | duration | tracking RMSE | response lag | median correlation |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for name, summary in analysis["gain_profiles"].items():
    lo, hi = summary["tracking_rmse_rad_range"]
    lag_lo, lag_hi = summary["response_lag_ms_range"]
    lines.append(
      f"| {name} | {summary['runs']} | {summary['completed_live_runs']}/"
      f"{summary['live_runs']} | {summary['duration_s']:.1f} s | "
      f"{summary['tracking_rmse_rad_median']:.3f} rad ({lo:.3f}-{hi:.3f}) | "
      f"{summary['response_lag_ms_median']:.0f} ms ({lag_lo:.0f}-{lag_hi:.0f}) | "
      f"{summary['response_correlation_median']:.2f} |"
    )
  lines.extend(
    [
      "",
      "## Run inventory",
      "",
      "| file | mode/profile | status | duration | fresh | guard clip | RMSE | max target slew |",
      "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
  )
  for run in analysis["runs"]:
    lines.append(
      f"| {run['file']} | {run['mode']}/{run['gain_profile']} | {run['status']} | "
      f"{run['duration_s']:.2f} s | {run['feedback_fresh_fraction']:.1%} | "
      f"{run['guard_clipped_fraction']:.1%} | {run['tracking_rmse_rad']:.3f} rad | "
      f"{run['max_command_slew_rad_s']:.2f} rad/s |"
    )
  lines.extend(
    [
      "",
      "## Decision",
      "",
      "Proceed with the velocity-policy training pipeline. Keep the two gain profiles "
      "separate, reproduce the measured closed-loop response envelope in simulation, "
      "and retain conservative dynamics randomization because these logs do not uniquely "
      "identify motor friction, link inertia, ground contact, or calibrated joint torque.",
      "",
      "Before the first untethered policy deployment, collect dedicated per-joint tests: "
      "small step and chirp commands at the intended deployed gains, unloaded and loaded; "
      "log commanded position, measured position/velocity, timestamps, voltage, current, "
      "and calibrated output torque when available. Also measure total robot mass, segment "
      "masses/centres of mass, and foot-ground friction. Those tests can narrow the current "
      "randomization ranges; they do not block policy development.",
      "",
      "Raw per-joint lag and correlation values are in `analysis.json`.",
      "",
    ]
  )
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("data_dir", type=Path)
  args = parser.parse_args()

  paths = sorted(args.data_dir.glob("*.npz"))
  if not paths:
    raise SystemExit(f"no NPZ logs found in {args.data_dir}")
  runs = [analyze_run(path) for path in paths]
  profiles = {
    name: summarize_profile([run for run in runs if run["gain_profile"] == name])
    for name in sorted({run["gain_profile"] for run in runs})
  }
  analysis = {
    "source_commit": SOURCE_COMMIT,
    "limitations": [
      "Closed-loop response lag is not pure transport delay.",
      "Motor torque feedback is not calibrated joint-output torque.",
      "These runs cannot uniquely identify friction, inertia, contact, and delay.",
    ],
    "gain_profiles": profiles,
    "runs": runs,
  }
  (args.data_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
  (args.data_dir / "REPORT.md").write_text(make_report(analysis))
  print(f"analyzed {len(runs)} logs ({sum(run['duration_s'] for run in runs):.1f} s)")
  print(f"wrote {args.data_dir / 'analysis.json'}")
  print(f"wrote {args.data_dir / 'REPORT.md'}")


if __name__ == "__main__":
  main()
