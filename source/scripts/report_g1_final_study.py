"""Offline report for the predeclared three-seed F1 study; no model calls."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from source.scripts.run_g1_queue import assess_gate


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(rows, key):
    return statistics.mean(float(r[key]) for r in rows)


def validate_evaluation(summary, rows, protocol, entry, reference):
    assert summary["commands"] == protocol["commands"], "Changed held-out commands"
    assert summary["reset_seeds"] == protocol["reset_seeds"], "Changed held-out reset seeds"
    assert summary["steps_per_episode"] == 500 and summary["trained_checkpoint"] == 1003520
    assert summary["untrained_checkpoint"] == 0 and summary["reference_controller_label"] == "untrained"
    assert Path(summary["reference_run_dir"]).resolve() == reference.resolve(), "Wrong untrained reference"
    assert summary["reset_randomized"] == (entry["mode"] == "randomized")
    assert summary["observation_noise"] is False
    for row in rows:
        assert [float(row[k]) for k in ("command_vx", "command_vy", "command_yaw_rate")] == protocol["commands"][row["command_name"]]


def build_report(protocol, root=ROOT):
    plan = read(root / protocol["queue_plan"])
    jobs = {j["id"]: j for j in plan["jobs"]}
    results, initial_hashes, training_steps = [], [], 0
    for seed_run in protocol["seed_runs"]:
        seed = seed_run["seed"]
        assert read(root / seed_run["untrained_audit"])["normalizer_count"] == 0
        initial = root / seed_run["untrained_run"] / "logs/checkpoints/0/policy"
        initial_hashes.append(hashlib.sha256(initial.read_bytes()).hexdigest())
        for index, training in enumerate(seed_run["training"]):
            directory = root / training["run_dir"]
            manifest = read(directory / "run_manifest.json")
            config = read(directory / "resolved_config.json")
            assert config["agent"]["seed"] == seed
            assert config["agent"]["num_timesteps"] == training["steps"]
            assert manifest["jax_backend"] == "gpu", "Training did not use GPU"
            assert manifest["command"] == [a.replace("{root}", str(root)) for a in training["argv"]]
            assert manifest.get("ended_at_utc"), "Unfinished training stage"
            assert (directory / "logs/checkpoints" / str(training["steps"]) / "policy").exists()
            if index == 0:
                assert "--resume" not in manifest["command"] and "--checkpoint" not in manifest["command"]
            else:
                audit = read(directory / "restore_parity.json")
                assert audit["exact_actor_normalizer_match"] and audit["leaves"] == 17
            training_steps += training["steps"]
        prior = read(root / f"results/final_f1/seed{seed}/command_prior/audit.json")
        assert prior["max_old_subspace_action_difference"] == 0
        assert prior["max_full_checkpoint_round_trip_difference"] == 0
        for entry in seed_run["evaluations"]:
            directory = root / entry["output_dir"]
            summary = read(directory / "summary.json")
            with (directory / "episodes.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            validate_evaluation(summary, rows, protocol, entry, root / seed_run["untrained_run"])
            gate = assess_gate(directory / "episodes.csv", jobs[entry["job_id"]]["gate"])
            grouped = {c: [r for r in rows if r["controller"] == c] for c in ("trained", "untrained", "standing")}
            metrics = {c: {"mean_steps": mean(r, "episode_steps"),
                           "linear_rmse": mean(r, "linear_velocity_vector_rmse"),
                           "yaw_rmse": mean(r, "yaw_rate_rmse")} for c, r in grouped.items()}
            # Material superiority is required on both matched evaluation suites;
            # standing's zero-yaw bias is already covered by absolute yaw gates.
            superiority = all(metrics["trained"]["linear_rmse"] <= .8 * metrics[c]["linear_rmse"]
                              and metrics["trained"]["mean_steps"] > metrics[c]["mean_steps"]
                              for c in ("untrained", "standing"))
            results.append(dict(seed=seed, mode=entry["mode"], metrics=metrics,
                                gates=gate, controls_outperformed=superiority,
                                numeric_pass=gate["outcomes"]["pass"]["passed"] and superiority,
                                videos=[str(directory / v) for v in summary["videos"] if v.startswith("trained_")]))
    assert len(set(initial_hashes)) == len(protocol["training_seeds"]) == 3, "Initializations not distinct"
    assert training_steps == protocol["total_training_steps"]
    stats = {}
    for mode in ("nominal", "randomized"):
        stats[mode] = {}
        for key in ("linear_rmse", "yaw_rmse", "mean_steps"):
            values = [r["metrics"]["trained"][key] for r in results if r["mode"] == mode]
            stats[mode][key] = {"seed_values": values, "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}
    return dict(status="needs_final_visual_review", all_numeric_pass=all(r["numeric_pass"] for r in results),
                training_steps=training_steps, training_seeds=protocol["training_seeds"],
                initial_policy_sha256=initial_hashes, results=results, seed_statistics=stats,
                note="No retuning, best-seed selection or final PASS until representative videos are reviewed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    report = build_report(protocol)
    out = ROOT / protocol["output_dir"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# F1 three-seed validation", "", f"All numeric gates and control comparisons pass: {report['all_numeric_pass']}",
             "Final video review remains required. Training seeds are independent; no seed is omitted.", "",
             "| Seed | Regime | Mean steps | Linear RMSE | Yaw RMSE | Numeric pass |", "|---|---|---|---|---|---|"]
    for r in report["results"]:
        m = r["metrics"]["trained"]
        lines.append(f"| {r['seed']} | {r['mode']} | {m['mean_steps']:.1f} | {m['linear_rmse']:.4f} | {m['yaw_rmse']:.4f} | {r['numeric_pass']} |")
    lines += ["", "See summary.json for seed-level statistics, every gate result, control metrics and video paths."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_numeric_pass": report["all_numeric_pass"], "report": str(out)}))


if __name__ == "__main__":
    main()
