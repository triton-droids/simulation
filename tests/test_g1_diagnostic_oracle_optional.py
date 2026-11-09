"""Exercise optional oracle setup without requiring the GPU simulation stack."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def setup_code():
    path = Path(__file__).resolve().parents[1] / 'robocub/scripts/compare_g1_gait_rewards.py'
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    branch = next(n for n in main.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'not args.local_only')
    return compile(ast.Module(body=[branch], type_ignores=[]), str(path), 'exec')


def test_local_only_never_accesses_oracle():
    oracle = Mock()
    oracle._default_policy_path.side_effect = AssertionError('unused asset accessed')
    scope = dict(args=SimpleNamespace(local_only=True), oracle=oracle,
                 session=None, oracle_sha256=None)
    exec(setup_code(), scope)
    assert scope['session'] is None
    assert scope['oracle_sha256'] is None
    oracle._default_policy_path.assert_not_called()


def test_reference_mode_rejects_wrong_checksum_before_loading_runtime():
    oracle = Mock(EXPECTED_ONNX_SHA256='expected')
    oracle._sha256.return_value = 'wrong'
    with pytest.raises(AssertionError):
        exec(setup_code(), dict(args=SimpleNamespace(local_only=False), oracle=oracle))
    oracle._sha256.assert_called_once()
