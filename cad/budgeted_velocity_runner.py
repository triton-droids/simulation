"""Wall-clock training budget and periodic checkpoint selection for velocity RL."""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class TrainingBudgetReached(Exception):
  pass


class BudgetedVelocityRunner(VelocityOnPolicyRunner):
  evaluate_script = "evaluate_velocity.py"

  def score(self, result):
    # Survive first, then track the command.  Reward is the tie-breaker.
    return (
      result["fall_rate"],
      -result["mean_linear_tracking_reward"],
      -result["mean_forward_speed"],
    )

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
    output = self.run_dir / f"selection_{checkpoint.stem}.json"
    try:
      with output.with_suffix(".log").open("w") as stream:
        subprocess.run(
          [
            sys.executable,
            str(Path(__file__).with_name(self.evaluate_script)),
            "--checkpoint",
            str(checkpoint.resolve()),
            "--num-envs",
            "64",
            "--steps",
            "400",
            "--output",
            str(output.resolve()),
          ],
          check=True,
          timeout=1800,
          stdout=stream,
          stderr=subprocess.STDOUT,
        )
    except (subprocess.SubprocessError, OSError) as error:
      # A failed evaluation must not throw away a training run.
      print(f"[WARN] checkpoint evaluation failed for {checkpoint}: {error}", flush=True)
      return
    result = json.loads(output.read_text())
    score = self.score(result)
    self.last_evaluated = str(checkpoint)
    if self.best_score is None or score < self.best_score:
      self.best_score = score
      shutil.copy2(checkpoint, self.run_dir / "best.pt")
      self.export_policy_to_onnx(str(self.run_dir), "best.onnx")
      shutil.copy2(output, self.run_dir / "best_evaluation.json")
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
