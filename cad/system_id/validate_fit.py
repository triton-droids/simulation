#!/usr/bin/env python3
"""Replay held-out recorded commands; retain predictions and measured data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sysid_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motor", type=int, required=True)
    parser.add_argument("--sim-repo", type=Path, default=sysid_fit.SIM_REPO)
    args = parser.parse_args()
    sysid_fit.SIM_REPO = args.sim_repo.resolve()
    fit = json.loads(args.params.read_text())
    model_info = fit["mjcf"]
    # Pin to the fit's recorded source commit, not a moving branch.
    sysid_fit.MJCF_REF = model_info["commit"] + ":" + model_info["ref"].split(":", 1)[1]
    sysid_fit.BASE_MODE = model_info["base_mode"]
    sysid_fit.SIM_DT = model_info["sim_dt"]
    joint = next(j for j in fit["joints"] if j["motor_id"] == args.motor)
    assert set(joint["trials"]["fit"]).isdisjoint(joint["trials"]["holdout"])
    args.output.mkdir(parents=True, exist_ok=False)
    sim = sysid_fit.Simulator()
    results = []
    for name in joint["trials"]["holdout"]:
        path = args.session / name
        trial = sysid_fit.Trial(path)
        predicted = sim.run(trial, joint["fitted"])
        baseline = sim.run(trial, fit["nominal_baseline"])
        rms = lambda q: float(np.degrees(np.sqrt(np.mean((q-trial.meas_q)**2))))
        row = {"trial": name, "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "fitted_rms_deg": rms(predicted), "baseline_rms_deg": rms(baseline),
               "samples": len(predicted)}
        np.savez_compressed(args.output / name, measured_t=trial.meas_t,
                            measured_q=trial.meas_q, fitted_q=predicted, baseline_q=baseline,
                            command_t=trial.cmd_t, post_guard_command=trial.cmd,
                            metadata=json.dumps(row))
        results.append(row)
    report = {"fit_sha256": hashlib.sha256(args.params.read_bytes()).hexdigest(),
              "motor": args.motor, "mjcf": model_info, "results": results,
              "scope": "Recorded gantry holdout replay; no contact, peak torque or hardware qualification"}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
