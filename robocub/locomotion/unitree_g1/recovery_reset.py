"""Bounded training-only reset selection; one candidate preserves upstream reset."""
import jax
import jax.numpy as jp


def sample_recovery_reset(reset_fn, score_fn, rng, candidates=1):
    if not 1 <= candidates <= 8:
        raise ValueError('Reset candidates must be between 1 and 8')
    original = reset_fn(rng)
    if candidates == 1:
        return original
    selected = original
    for i in range(1, candidates):
        candidate = reset_fn(jax.random.fold_in(rng, i))
        selected = jax.lax.cond(score_fn(candidate) > score_fn(selected),
                                lambda _: candidate, lambda _: selected, None)
    # Retain ordinary resets in half the training episodes.
    focused = jax.random.bernoulli(jax.random.fold_in(rng, 10001), .5)
    return jax.lax.cond(focused, lambda _: selected, lambda _: original, None)


def backward_lateral_score(velocity):
    return jp.maximum(-velocity[0], 0) * jp.abs(velocity[1])
