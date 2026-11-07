import json

import pytest

from robocub.user_cli.sweep import _load_trials, _override


def test_sweep_config_requires_nonempty_trial_list(tmp_path):
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps([{}]))
    assert _load_trials(path) == [{}]

    path.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="non-empty JSON list"):
        _load_trials(path)


def test_hydra_override_rendering():
    assert _override("agent.learning_rate", 0.0003) == "agent.learning_rate=0.0003"
    assert _override("sim.push.add_push", True) == "sim.push.add_push=true"
    assert _override("agent.policy_hidden_layer_sizes", [32, 16]) == "agent.policy_hidden_layer_sizes=[32,16]"
