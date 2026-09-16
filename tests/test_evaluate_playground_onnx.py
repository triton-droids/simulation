"""Regression tests for the shipped-policy behavioral oracle."""

import numpy as np

from source.scripts.evaluate_playground_onnx import _is_terminal


def test_oracle_termination_uses_positive_upvector_as_upright() -> None:
    finite = np.zeros(3)

    assert not _is_terminal(1.0, False, finite, finite)
    assert _is_terminal(-1e-6, False, finite, finite)
    assert _is_terminal(1.0, True, finite, finite)
    assert _is_terminal(1.0, False, np.array([np.inf]), finite)
