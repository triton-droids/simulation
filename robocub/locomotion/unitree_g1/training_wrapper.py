"""Preserve Brax terminal metadata through Playground's full autoreset.

Uses the preserve-info extension in Playground 8a4b464 (Apache-2.0).
Environment history resets, but transition metadata must reach PPO unchanged.
"""

from brax.envs.base import Wrapper


_PRESERVE = "AutoResetWrapper_preserve_info"
_EPISODE_KEYS = ("steps", "truncation", "episode_done", "episode_metrics")
_BOOKKEEPING = "g1_episode_bookkeeping"


class _CaptureEpisodeInfo(Wrapper):
    @staticmethod
    def _capture(state):
        info = dict(state.info)
        preserved = dict(info.get(_PRESERVE, {}))
        preserved[_BOOKKEEPING] = {key: info[key] for key in _EPISODE_KEYS}
        info[_PRESERVE] = preserved
        return state.replace(info=info)

    def reset(self, rng):
        return self._capture(self.env.reset(rng))

    def step(self, state, action):
        # EpisodeWrapper updates nested accumulators in place.
        info = dict(state.info)
        info["episode_metrics"] = dict(info["episode_metrics"])
        return self._capture(self.env.step(state.replace(info=info), action))


class _RestoreEpisodeInfo(Wrapper):
    def step(self, state, action):
        result = self.env.step(state.replace(info=dict(state.info)), action)
        info = dict(result.info)
        info.update(info[_PRESERVE][_BOOKKEEPING])
        return result.replace(info=info)


def wrap_for_brax_training(
    env, episode_length=1000, action_repeat=1, randomization_fn=None,
    *, wrapper_module, full_reset=True,
):
    """Retain the pinned wrapper stack and its supported full-reset mechanism."""
    if not full_reset:
        raise ValueError("G1 requires coherent full-state reset")
    wrapped = wrapper_module.wrap_for_brax_training(
        env, episode_length=episode_length, action_repeat=action_repeat,
        randomization_fn=randomization_fn, full_reset=True,
    )
    wrapped.env = _CaptureEpisodeInfo(wrapped.env)
    return _RestoreEpisodeInfo(wrapped)
