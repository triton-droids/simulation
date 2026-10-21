"""Budgeted runner for the soccer task: selects checkpoints by ball progress."""

from budgeted_velocity_runner import BudgetedVelocityRunner, TrainingBudgetReached

__all__ = ["BudgetedSoccerRunner", "TrainingBudgetReached"]


class BudgetedSoccerRunner(BudgetedVelocityRunner):
  evaluate_script = "evaluate_soccer.py"

  def score(self, result):
    # Fewest falls, then most goals, then hardest kicks.
    return (
      result["fall_rate"],
      -result["goals_per_episode"],
      -result["peak_ball_speed"],
      -result["mean_ball_travel"],
    )
