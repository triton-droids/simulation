import functools
import numpy as np
import pytest


def test_normalizer_shift_keeps_weights_and_detects_change():
    jax = pytest.importorskip("jax")
    import jax.numpy as jp
    from brax.training.acme import running_statistics as rs
    from brax.training.agents.ppo import networks
    from robocub.scripts.audit_g1_normalizer_shift import compare_normalizers
    net = networks.make_ppo_networks({"state": (4,), "privileged_state": (4,)}, 2,
        policy_hidden_layer_sizes=(8,), value_hidden_layer_sizes=(8,),
        policy_obs_key="state", value_obs_key="privileged_state",
        preprocess_observations_fn=rs.normalize)
    params = net.policy_network.init(jax.random.PRNGKey(1))
    before = [np.asarray(x).copy() for x in jax.tree_util.tree_leaves(params)]
    norm = rs.init_state({"state": jp.zeros(4), "privileged_state": jp.zeros(4)})
    x = jp.arange(128, dtype=jp.float32).reshape(32,4) / 10 + 5
    obs = {"state": x, "privileged_state": x}
    metrics = jax.jit(functools.partial(compare_normalizers, net))(norm, params, obs)
    assert all(np.isfinite(float(v)) for v in metrics.values())
    assert float(metrics["identity_logits_max_difference"]) == 0
    assert abs(float(metrics["identity_kl"] - metrics["identity_expected_floor"])) < 1e-5
    assert float(metrics["normalizer_only_kl"]) > 0
    assert int(norm.count.lo) == 0
    assert all(np.array_equal(a,np.asarray(b)) for a,b in zip(before,jax.tree_util.tree_leaves(params)))
