from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import engineai_t800_flat_tracking_env_cfg
from .rl_cfg import engineai_t800_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-T800",
  env_cfg=engineai_t800_flat_tracking_env_cfg(),
  play_env_cfg=engineai_t800_flat_tracking_env_cfg(play=True),
  rl_cfg=engineai_t800_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# SDK-deployable variant: actor limited to the six observation terms the
# EngineAI Native SDK runner provides (134-dim input). Use this task for any
# policy that will be exported to the robot/evaluation runner.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-T800-Deploy",
  env_cfg=engineai_t800_flat_tracking_env_cfg(deploy=True),
  play_env_cfg=engineai_t800_flat_tracking_env_cfg(play=True, deploy=True),
  rl_cfg=engineai_t800_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
