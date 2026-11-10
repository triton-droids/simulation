"""Behavior checks runnable without booting Isaac Sim: python -m unittest discover -s tests."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from locomotion_rewards import (
    FOOT_NAMES, SOLE_CENTERS, alternating_contact, feet_slide, gait_phase_observation,
    lateral_sole_spacing, stance_width_tracking,
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
