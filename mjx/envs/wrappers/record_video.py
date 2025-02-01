from mujoco_playground._src.wrapper import Wrapper
from typing import Any, Callable, Dict, Optional, Tuple, Union
import os
from absl import logging
from mujoco_playground._src import mjx_env
import jax
import mediapy as media
from utils.rollout import mjx_rollout
import mujoco
import multiprocessing


def default_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 4, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0

class RecordVideo(Wrapper):
    """ This wrapper records videos of rollouts

    To do this, you can specify **either** ``episode_trigger`` **or** ``step_trigger`` (not both).
    They should be functions returning a boolean that indicates whether a recording should be started at the
    current episode or step, respectively.
    If neither :attr:`episode_trigger` nor ``step_trigger`` is passed, a default ``episode_trigger`` will be employed.
    By default, the recording will be stopped once a `terminated` or `truncated` signal has been emitted by the environment. However, you can
    also create recordings of fixed length (possibly spanning several episodes) by passing a strictly positive value for
    ``video_length``.

    NOT FULLY TrUE JUST YET
    """

    def __init__(self,
                 env : mjx_env.MjxEnv,
                 video_folder: str,
                 episode_trigger: Callable[[int], bool] = None,
                 video_length: int = 0,
                 name_prefix: str = "rl-video",      
    ):
        super().__init__(env)
        if episode_trigger is None:
            episode_trigger = default_video_schedule

        self.episode_trigger = episode_trigger
        self.video_folder = os.path.abspath(video_folder)

        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logging.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.video_length = video_length

        self.record = False
        self.recorded_frames = 0
        self.episode_id = 0
    
    def reset(self, rng):
        """Reset the environment and save trajectory if episode trigger is enabled """
        state = self.env.reset(rng)

        if self._video_enabled():
            self.save_trajectory()

        return state

    def _video_enabled(self):
        return self.episode_trigger(self.episode_id)

    def save_trajectory(self):
        """Save the trajectory of the current episode"""
        video_name = f"{self.name_prefix}-episode-{self.episode_id}"
        rollout = mjx_rollout(self.env, self.inference_fn, episode_length=self.video_length)
        base_path = os.path.join(self.video_folder, video_name)
        self.save_video(rollout, base_path)
    
    def save_video(self, rollout: list[mjx_env.State], path: str):
        """ Saves the video of the rollout """
        render_every = 2
        fps = 1.0 / self.env.dt / render_every

        traj = rollout[::render_every]

        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

        frames = self.env.render(
            traj, height=480, width=640, scene_option=scene_option
        )
        media.write_video(f"{self.name_prefix}.mp4", frames, fps=fps)


   