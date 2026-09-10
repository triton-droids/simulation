"""Tests for Brax checkpoint layout compatibility."""

import pytest

from source.utils.checkpoints import inference_params_from_training_params


def test_current_brax_three_item_layout():
    params = ("normalizer", {"params": "policy"}, {"params": "value"})

    assert inference_params_from_training_params(params) == (
        "normalizer",
        {"params": "policy"},
    )


def test_historical_nested_mapping_layout():
    params = ("normalizer", {"policy": "policy", "value": "value"})

    assert inference_params_from_training_params(params) == (
        "normalizer",
        "policy",
    )


def test_already_inference_shaped_layout():
    assert inference_params_from_training_params(("normalizer", "policy")) == (
        "normalizer",
        "policy",
    )


def test_invalid_layout_is_rejected():
    with pytest.raises(ValueError, match="at least two"):
        inference_params_from_training_params(("normalizer",))
