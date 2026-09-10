"""Regression test for the pinned Brax/modern JAX pmap adapter."""

import jax
import jax.numpy as jp
import numpy as np

from source.utils.jax_compat import install_brax_pmap_compatibility


def test_brax_replication_helper_is_available_and_preserves_leading_device_axis():
    install_brax_pmap_compatibility()
    value = {"x": jp.array([1.0, 2.0]), "y": jp.array(3.0)}
    replicated = jax.device_put_replicated(value, jax.local_devices())
    assert replicated["x"].shape == (jax.local_device_count(), 2)
    assert replicated["y"].shape == (jax.local_device_count(),)
    np.testing.assert_allclose(np.asarray(replicated["x"][0]), [1.0, 2.0])
