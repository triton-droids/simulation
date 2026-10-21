"""Wall-clock training budget and periodic checkpoint selection."""

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner


class TrainingBudgetReached(Exception):
  pass


class BudgetedTrackingRunner(MotionTrackingOnPolicyRunner):
  def configure_budget(self, hours, evaluation_interval, run_dir):
    self.budget_started = time.monotonic()
    self.deadline = self.budget_started + hours * 3600 if hours else float("inf")
    self.evaluation_interval = evaluation_interval
    self.run_dir = Path(run_dir)
    self.best_score = None
    self.last_evaluated = None
    self.last_saved = None

  def evaluate_candidate(self, checkpoint):
    checkpoint = Path(checkpoint)
    if self.last_evaluated == str(checkpoint):
      return
    results = []
    for condition in ("nominal", "latency", "extreme"):
      output = self.run_dir / f"selection_{checkpoint.stem}_{condition}.json"
      with output.with_suffix(".log").open("w") as stream:
        subprocess.run(
          [
            sys.executable,
            str(Path(__file__).with_name("evaluate_hardware_tracking.py")),
            str(checkpoint),
            "--condition",
            condition,
            "--num-envs",
            "64",
            "--steps",
            "500",
            "--output",
            str(output),
          ],
          check=True,
          timeout=600,
          stdout=stream,
          stderr=subprocess.STDOUT,
        )
      results.append(json.loads(output.read_text()))
    # First enforce torque feasibility; then minimize failures across conditions,
    # tracking error and finally maximize reward. Selection does not prove deployment safety.
    score = (
      sum(not r["software_cap_pass"] for r in results),
      sum(not r["mean_stalled_rating_pass"] for r in results),
      sum(r["failure_resets"] for r in results),
      sum(r["mean_tracking_errors"].get("error_joint_pos", 1e6) for r in results),
      -sum(r["mean_reward_per_step"] for r in results),
    )
    if not all(math.isfinite(float(value)) for value in score):
      raise RuntimeError(f"Non-finite evaluation score for {checkpoint}")
    self.last_evaluated = str(checkpoint)
    if self.best_score is None or score < self.best_score:
      self.best_score = score
      shutil.copy2(checkpoint, self.run_dir / "best.pt")
      self.export_policy_to_onnx(str(self.run_dir), "best.onnx")
      for condition, result in zip(
        ("nominal", "latency", "extreme"), results, strict=True
      ):
        (self.run_dir / f"best_{condition}.json").write_text(
          json.dumps(result, indent=2)
        )
      (self.run_dir / "best.json").write_text(
        json.dumps(
          {
            "checkpoint": str(checkpoint),
            "score": score,
            "selection": "64 environments, 10 seconds in nominal, latency and one extreme corner",
            "candidates_are_periodic": True,
          },
          indent=2,
        )
      )
    print(f"Checkpoint evaluation: {checkpoint.name}, score={score}", flush=True)

  def save(self, path, infos=None):
    super().save(path, infos)
    self.last_saved = path
    iteration = self.current_learning_iteration
    expired = time.monotonic() >= self.deadline
    if iteration > 0 and (expired or iteration % self.evaluation_interval == 0):
      self.evaluate_candidate(path)
    (self.run_dir / "status.json").write_text(
      json.dumps(
        {
          "iteration": iteration,
          "elapsed_seconds": time.monotonic() - self.budget_started,
          "latest_checkpoint": str(path),
          "best_score": self.best_score,
          "state": "budget_reached" if expired else "training",
        },
        indent=2,
      )
    )
    if expired:
      raise TrainingBudgetReached()
