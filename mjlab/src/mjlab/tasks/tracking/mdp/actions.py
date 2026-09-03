"""Tracking-task action terms.

The EngineAI Native SDK's whole-body-tracking runner computes

  q_des = ref_joint_pos(t) + action * action_scale

(see submission_tools/SDK_FINDINGS.md §2), i.e. the action offset is the
time-varying reference pose from the trajectory, not a constant default pose.
Policies that will be deployed through that runner must be trained with the
same semantics; this module provides the matching action term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MotionReferenceJointPositionActionCfg(BaseActionCfg):
  """Joint position control offset by the motion command's reference pose.

  target = ref_joint_pos(t) + action * scale

  This mirrors the EngineAI SDK runner exactly: the reference used as offset is
  the same one the policy sees in its ``command`` observation for the current
  step. The ``offset`` field inherited from ``BaseActionCfg`` must remain 0.
  """

  command_name: str = "motion"

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT
    if self.offset != 0.0:
      raise ValueError(
        "MotionReferenceJointPositionActionCfg does not support a constant "
        "'offset'; the offset is the motion reference pose."
      )

  def build(self, env: ManagerBasedRlEnv) -> MotionReferenceJointPositionAction:
    return MotionReferenceJointPositionAction(self, env)


class MotionReferenceJointPositionAction(BaseAction):
  """Joint position action with the motion reference pose as dynamic offset."""

  cfg: MotionReferenceJointPositionActionCfg

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    command = cast(
      MotionCommand, self._env.command_manager.get_term(self.cfg.command_name)
    )
    # Reference at the step the policy observed; columns are entity joint
    # indices, matching command.joint_pos / robot.data.joint_pos ordering.
    ref_joint_pos = command.joint_pos[:, self._target_ids]
    self._processed_actions = self._raw_actions * self._scale + ref_joint_pos
    if self.cfg.clip is not None:
      self._processed_actions = torch.clamp(
        self._processed_actions,
        min=self._clip[:, :, 0],
        max=self._clip[:, :, 1],
      )

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
