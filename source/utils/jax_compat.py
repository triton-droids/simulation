# Copyright 2026 Triton Droids
# Drop-in compatibility approach adapted from the JAX pmap migration guide
# (Apache-2.0): https://docs.jax.dev/en/latest/migrate_pmap.html
"""Narrow compatibility adapters for Brax 0.14.2 on modern JAX."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _api_available(name: str) -> bool:
    try:
        getattr(jax, name)
    except AttributeError:
        return False
    return True


def _device_put_replicated(value: Any, devices: Sequence[jax.Device]):
    """Temporary public-API replacement for removed JAX replication helper."""

    if not devices:
        raise ValueError("devices must contain at least one JAX device")
    mesh = Mesh(np.asarray(devices), ("_device",))
    sharding = NamedSharding(mesh, P("_device"))
    return jax.tree.map(
        lambda item: jax.device_put(jnp.stack([item] * len(devices)), sharding),
        value,
    )


def _device_put_sharded(shards: Sequence[Any], devices: Sequence[jax.Device]):
    """Temporary public-API replacement for removed JAX sharding helper."""

    if not devices or len(shards) != len(devices):
        raise ValueError("shards and devices must be nonempty and have equal length")
    mesh = Mesh(np.asarray(devices), ("_device",))
    sharding = NamedSharding(mesh, P("_device"))
    return jax.tree.map(
        lambda *items: jax.device_put(jnp.stack(items), sharding), *shards
    )


def install_brax_pmap_compatibility() -> tuple[str, ...]:
    """Install only helpers removed by JAX but still called by pinned Brax."""

    installed: list[str] = []
    if not _api_available("device_put_replicated"):
        setattr(jax, "device_put_replicated", _device_put_replicated)
        installed.append("device_put_replicated")
    if not _api_available("device_put_sharded"):
        setattr(jax, "device_put_sharded", _device_put_sharded)
        installed.append("device_put_sharded")
    return tuple(installed)
