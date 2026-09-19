"""Guard explicit normalization preparation for new command dimensions."""
import jax.numpy as jp
import numpy as np
import pytest
from brax.training.acme import running_statistics as rs
from source.utils.checkpoints import set_unseen_g1_command_std


def _params():
    stats = rs.init_state({"state": jp.zeros(103), "privileged_state": jp.zeros(216)})
    stats = stats.replace(count=jp.asarray(100.),
        std={k: v.at[10:12].set(1e-6) for k,v in stats.std.items()},
        summed_variance={k: jp.ones_like(v).at[10:12].set(0) * 100 for k,v in stats.std.items()})
    return stats, {"actor": jp.array([1., 2.])}, {"critic": jp.array([3., 4.])}


def test_command_prior_preserves_old_subspace_weights_and_other_statistics():
    original = _params()
    prepared = set_unseen_g1_command_std(original, .15, .25)
    assert prepared[1] is original[1] and prepared[2] is original[2]
    samples = {k: jp.arange(len(v), dtype=float).at[10:12].set(0) for k,v in original[0].mean.items()}
    before, after = rs.normalize(samples, original[0]), rs.normalize(samples, prepared[0])
    for key in samples:
        np.testing.assert_array_equal(before[key], after[key])
        np.testing.assert_array_equal(prepared[0].mean[key], original[0].mean[key])
        np.testing.assert_array_equal(prepared[0].std[key][:10], original[0].std[key][:10])
        np.testing.assert_array_equal(prepared[0].std[key][12:], original[0].std[key][12:])
        np.testing.assert_allclose(prepared[0].summed_variance[key][10:12], [2.25, 6.25])
        np.testing.assert_allclose(original[0].std[key][10:12], [1e-6, 1e-6])
    batch = {k: jp.zeros((1,len(v))) for k,v in prepared[0].mean.items()}
    updated = rs.update(prepared[0], batch)
    np.testing.assert_allclose(updated.std["state"][10:12], np.array([.15,.25])*np.sqrt(100/101), rtol=1e-6)


def test_command_prior_rejects_trained_channels_and_invalid_priors():
    params = _params()
    with pytest.raises(ValueError, match="finite"):
        set_unseen_g1_command_std(params, float("nan"), .2)
    std = {k: v.at[10].set(.1) for k,v in params[0].std.items()}
    with pytest.raises(ValueError, match="already-varying"):
        set_unseen_g1_command_std((params[0].replace(std=std), *params[1:]), .15, .25)
