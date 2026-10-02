"""Verify the deployed ONNX contract and its baked reference motion."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

EXPECTED_OUTPUTS = (
  "actions",
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)
REFERENCE_OUTPUTS = EXPECTED_OUTPUTS[1:]


def verify(model_path: Path, reference_path: Path):
  session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
  assert [(i.name, i.shape) for i in session.get_inputs()] == [
    ("obs", [1, 56]),
    ("time_step", [1, 1]),
  ]
  assert tuple(o.name for o in session.get_outputs()) == EXPECTED_OUTPUTS
  assert session.get_outputs()[0].shape == [1, 10]
  ref = np.load(reference_path)
  assert ref["joint_pos"].shape == (299, 10)
  max_error = 0.0
  for frame in (0, 1, 149, 298, 320):
    result = session.run(
      None,
      {
        "obs": np.zeros((1, 56), dtype=np.float32),
        "time_step": np.array([[frame]], dtype=np.float32),
      },
    )
    assert result[0].shape == (1, 10)
    for name, got in zip(REFERENCE_OUTPUTS, result[1:], strict=True):
      expected = ref[name][min(frame, 298)]
      if name.startswith("body_"):
        # Tracking task excludes the torso from the 13-body NPZ.
        expected = expected[[0, *range(2, 13)]]
      assert got.shape == (1, *expected.shape), (name, got.shape, expected.shape)
      max_error = max(max_error, float(np.max(np.abs(got[0] - expected))))
  assert max_error < 1e-5, max_error
  print(f"ONNX contract and 299-frame reference verified; max error {max_error:.3g}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("onnx", type=Path)
  parser.add_argument("reference", type=Path)
  args = parser.parse_args()
  verify(args.onnx, args.reference)
