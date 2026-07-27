"""Gait helper functions for phase-based locomotion experiments."""

import jax
import jax.numpy as jp
from typing import Union

def get_rz(
    phi: Union[jax.Array, float], swing_height: Union[jax.Array, float] = 0.08
) -> jax.Array:
  """Compute target foot height from gait phase.

  Args:
      phi: Phase value in radians.
      swing_height: Maximum desired swing-foot height.

  Returns:
      Desired z-height for the current phase.
  """

  def cubic_bezier_interpolation(y_start, y_end, x):
    """Interpolate smoothly between two foot-height targets.

    Args:
        y_start: Starting height.
        y_end: Ending height.
        x: Normalized phase progress in [0, 1].

    Returns:
        Interpolated height.
    """

    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier

  x = (phi + jp.pi) / (2 * jp.pi)
  stance = cubic_bezier_interpolation(0, swing_height, 2 * x)
  swing = cubic_bezier_interpolation(swing_height, 0, 2 * x - 1)
  return jp.where(x <= 0.5, stance, swing)
