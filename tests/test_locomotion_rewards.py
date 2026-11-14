"""Behavior checks runnable without booting Isaac Sim: python -m unittest discover -s tests."""

import sys
import math
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from locomotion_rewards import (
    FOOT_NAMES, SOLE_CENTERS, alternating_contact, feet_slide, gait_phase_observation,
    lateral_sole_spacing, stance_width_tracking, heading_error_l2, heading_tracking, heading_error_observation,
)


def example_env():
    commands = torch.tensor([[0., .35, 0.], [0., .35, 0.], [0., 0., 0.], [0., .35, 0.]])
    data = NS(
        projected_gravity_b=torch.tensor([[0., 0., -1.]] * 3 + [[0., 0., 1.]]),
        body_link_quat_w=torch.tensor([[[0., 0., 0., 1.]] * 2] * 4),
        body_link_lin_vel_w=torch.zeros(4, 2, 3),
        body_link_ang_vel_w=torch.zeros(4, 2, 3),
    )
    scene = {"robot": NS(data=data, body_names=list(FOOT_NAMES))}
    for name, contact in zip(FOOT_NAMES, ([1, 0, 1, 1], [0, 1, 1, 0])):
        forces = torch.zeros(4, 1, 3)
        forces[:, 0, 2] = torch.tensor(contact) * 78.
        scene[name + "_contact"] = NS(body_names=[name], data=NS(net_forces_w=forces))
    return NS(scene=scene, episode_length_buf=torch.tensor([10, 30, 10, 10]), step_dt=.02,
              command_manager=NS(get_command=lambda name: commands))


class RewardBehavior(unittest.TestCase):
    def heading_env(self, angles):
        env = example_env()
        yaw = torch.tensor(angles)
        env.scene["robot"].data.root_quat_w = torch.stack(
            (torch.zeros_like(yaw), torch.zeros_like(yaw), (yaw / 2).sin(), (yaw / 2).cos()), dim=-1
        )
        env.scene["robot"].data.projected_gravity_b[:, 2] = -1
        return env

    def test_heading_reward_and_penalty_distinguish_stopped_but_misaligned_robot(self):
        env = self.heading_env([0., .2, -.2, .5])
        torch.testing.assert_close(heading_error_l2(env), torch.tensor([0., .04, .04, .25]))
        scores = heading_tracking(env)
        torch.testing.assert_close(scores, torch.exp(-torch.tensor([0., .04, .04, .25]) / .25**2))
        self.assertGreater(float(scores[0]), float(scores[1]))
        self.assertGreater(float(scores[1]), float(scores[3]))
        # The third environment is commanded to stand; it must still hold heading.
        self.assertGreater(float(heading_error_l2(env)[2]), 0.)

    def test_heading_wraps_at_pi_and_accepts_nonzero_target(self):
        env = self.heading_env([math.radians(-179)] * 4)
        expected = torch.full((4,), math.radians(2)**2)
        torch.testing.assert_close(heading_error_l2(env, desired_yaw=math.radians(179)), expected)
        torch.testing.assert_close(heading_tracking(env, desired_yaw=math.radians(-179)), torch.ones(4))

    def test_heading_observation_encodes_error_direction_and_standing_alignment(self):
        env = self.heading_env([0., math.pi / 2, -math.pi / 2, math.pi])
        obs = heading_error_observation(env)
        self.assertEqual(tuple(obs.shape), (4, 2))
        torch.testing.assert_close(obs, torch.tensor([[0., 1.], [-1., 0.], [1., 0.], [0., -1.]]), atol=1e-6, rtol=0)
        # A desired yaw equal to the current yaw must encode perfect alignment.
        env = self.heading_env([.4] * 4)
        torch.testing.assert_close(heading_error_observation(env, desired_yaw=.4), torch.tensor([[0., 1.]] * 4))

    def test_heading_observation_is_continuous_at_wrap_and_disabled_only_for_turns(self):
        env = self.heading_env([math.pi - 1e-5, -math.pi + 1e-5, 0., 0.])
        obs = heading_error_observation(env)
        self.assertLess(float(torch.linalg.vector_norm(obs[0] - obs[1])), 3e-5)
        env.command_manager.get_command = lambda name: torch.tensor([[0., .35, .5], [0., .35, -.5], [0., 0., 0.], [0., .35, 0.]])
        torch.testing.assert_close(heading_error_observation(env), torch.tensor([[0., 0.], [0., 0.], [0., 1.], [0., 1.]]))

    def test_heading_terms_allow_commanded_turns_and_alignment_reward_requires_upright(self):
        env = self.heading_env([.2] * 4)
        env.command_manager.get_command = lambda name: torch.tensor([[0., .35, .5]] * 4)
        torch.testing.assert_close(heading_error_l2(env), torch.zeros(4))
        torch.testing.assert_close(heading_tracking(env), torch.zeros(4))
        env.command_manager.get_command = lambda name: torch.tensor([[0., .35, 0.]] * 4)
        env.scene["robot"].data.projected_gravity_b[:, 2] = 1
        torch.testing.assert_close(heading_tracking(env), torch.zeros(4))
        with self.assertRaises(ValueError):
            heading_tracking(env, std=0.)

    def test_alternating_foot_lifts_score_better_than_standing_or_hopping(self):
        env = example_env()
        torch.testing.assert_close(alternating_contact(env), torch.tensor([1., 1., 0., 0.]))
        env.scene["left_foot_contact"].data.net_forces_w[:, 0, 2] = 78
        env.scene["right_foot_contact"].data.net_forces_w[:, 0, 2] = 78
        torch.testing.assert_close(alternating_contact(env), torch.tensor([.5, .5, 0., 0.]))
        for name in FOOT_NAMES:
            env.scene[name + "_contact"].data.net_forces_w.zero_()
        torch.testing.assert_close(alternating_contact(env), torch.tensor([.5, .5, 0., 0.]))

    def test_wrong_leg_phase_is_not_rewarded(self):
        env = example_env()
        for name in FOOT_NAMES:
            f = env.scene[name + "_contact"].data.net_forces_w
            f[:, 0, 2] = 78 - f[:, 0, 2]
        torch.testing.assert_close(alternating_contact(env), torch.zeros(4))

    def test_phase_is_observable_periodic_and_disabled_for_standing(self):
        env = example_env()
        obs = gait_phase_observation(env)
        torch.testing.assert_close(obs[:3], torch.tensor([[1., 0.], [-1., 0.], [0., 0.]]), atol=1e-6, rtol=0)
        env.episode_length_buf += 40
        torch.testing.assert_close(gait_phase_observation(env), obs, atol=1e-6, rtol=0)

    def test_slip_counts_only_loaded_feet_including_standing_commands(self):
        env = example_env()
        env.scene["robot"].data.body_link_lin_vel_w[:, 0, 0] = .2
        env.scene["robot"].data.body_link_lin_vel_w[:, 1, 0] = .7
        torch.testing.assert_close(feet_slide(env), torch.tensor([.2, .7, .9, .2]))

    def test_pivoting_with_stationary_sole_is_not_misread_as_slip(self):
        env = example_env()
        data = env.scene["robot"].data
        data.body_link_ang_vel_w[..., 2] = 1
        offsets = torch.tensor(SOLE_CENTERS).expand(4, -1, -1)
        data.body_link_lin_vel_w = -torch.cross(data.body_link_ang_vel_w, offsets, dim=-1)
        torch.testing.assert_close(feet_slide(env), torch.zeros(4))

    def test_sensor_mapping_fails_loudly(self):
        env = example_env()
        env.scene["left_foot_contact"].body_names = ["hip"]
        with self.assertRaises(ValueError):
            alternating_contact(env)

    def test_width_uses_sole_centers_not_cad_origins_or_fore_aft_distance(self):
        env = example_env()
        data = env.scene["robot"].data
        data.root_quat_w = torch.tensor([[0., 0., 0., 1.]] * 4)
        centers = torch.zeros(4, 2, 3)
        centers[:, 0, 0] = torch.tensor([.18, .06, -.18, .30])
        centers[:, 0, 1] = .50  # A long stride must not inflate lateral width.
        data.body_link_pos_w = centers - torch.tensor(SOLE_CENTERS)
        torch.testing.assert_close(lateral_sole_spacing(env), torch.tensor([.18, .06, -.18, .30]))
        data.projected_gravity_b[:, 2] = -1
        env.command_manager.get_command = lambda name: torch.tensor([[0., .35, 0.]] * 4)
        scores = stance_width_tracking(env)
        self.assertEqual(float(scores[0]), 1.)
        self.assertTrue(bool((scores[1:] < scores[0]).all()))
        self.assertLess(float(scores[2]), float(scores[1]))

    def test_spacing_is_invariant_to_world_yaw_and_translation(self):
        env = example_env()
        data = env.scene["robot"].data
        # At +90 degrees heading, local X points along world +Y.
        a = 2 ** -.5
        data.root_quat_w = torch.tensor([[0., 0., a, a]] * 4)
        data.body_link_quat_w = data.root_quat_w[:, None].expand(-1, 2, -1)
        offsets = torch.tensor(SOLE_CENTERS)
        rotated = torch.stack((-offsets[:, 1], offsets[:, 0], offsets[:, 2]), dim=-1)
        centers = torch.tensor([[[3., 2.18, 0.], [3.5, 2., 0.]]] * 4)
        data.body_link_pos_w = centers - rotated
        torch.testing.assert_close(lateral_sole_spacing(env), torch.full((4,), .18), atol=1e-6, rtol=0)

    def test_width_reward_is_disabled_for_standing_and_inverted_robot(self):
        env = example_env()
        data = env.scene["robot"].data
        data.root_quat_w = torch.tensor([[0., 0., 0., 1.]] * 4)
        centers = torch.tensor([[[.18, 0., 0.], [0., 0., 0.]]] * 4)
        data.body_link_pos_w = centers - torch.tensor(SOLE_CENTERS)
        torch.testing.assert_close(stance_width_tracking(env), torch.tensor([1., 1., 0., 0.]))


if __name__ == "__main__":
    unittest.main()
