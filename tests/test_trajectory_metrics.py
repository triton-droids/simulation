import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trajectory_metrics import TrajectoryMetrics
from locomotion_rewards import yaw_rate_tracking, yaw_rate_error_l2


def rotation(angle):
    return torch.tensor([[0., 0., math.sin(angle / 2), math.cos(angle / 2)]])


class StraightWalkingMetrics(unittest.TestCase):
    def test_world_motion_projects_to_initial_body_y(self):
        m = TrajectoryMetrics(torch.zeros(1, 3), rotation(math.pi / 2))
        m.update(torch.tensor([[-2., 0., 0.]]), rotation(math.pi / 2), torch.tensor([False]), 1.)
        self.assertAlmostEqual(m.summary()["mean_forward_displacement_m"], 2.)
        self.assertLess(m.summary()["mean_abs_lateral_displacement_m"], 1e-6)

    def test_unwrap_full_circle_is_not_zero_drift(self):
        m = TrajectoryMetrics(torch.zeros(1, 3), rotation(0))
        for i in range(1, 41):
            m.update(torch.zeros(1, 3), rotation(i * 2 * math.pi / 40), torch.tensor([False]), .02)
        self.assertAlmostEqual(m.summary()["mean_abs_final_heading_drift_deg"], 360., places=3)

    def test_reset_jump_and_later_episode_are_excluded(self):
        m = TrajectoryMetrics(torch.zeros(1, 3), rotation(0))
        m.update(torch.tensor([[0., 1., 0.]]), rotation(0), torch.tensor([False]), .02)
        m.update(torch.tensor([[99., 99., 0.]]), rotation(2), torch.tensor([True]), .02)
        m.update(torch.tensor([[100., 100., 0.]]), rotation(3), torch.tensor([False]), .02)
        self.assertEqual(m.summary()["mean_forward_displacement_m"], 1.)
        self.assertEqual(m.summary()["survived_full_trial"], 0)
        self.assertEqual(m.summary()["mean_abs_final_heading_drift_deg"], 0.)

    def test_yaw_reward_tracks_command_and_penalizes_both_directions(self):
        command = torch.tensor([[0., .35, 0.], [0., .35, 0.], [0., .35, 0.], [0., 0., .4]])
        data = NS(root_ang_vel_w=torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., -1.], [0., 0., .4]]))
        env = NS(scene={"robot": NS(data=data)}, command_manager=NS(get_command=lambda name: command))
        torch.testing.assert_close(yaw_rate_error_l2(env), torch.tensor([0., 1., 1., 0.]))
        scores = yaw_rate_tracking(env)
        self.assertEqual(float(scores[0]), 1.)
        self.assertEqual(float(scores[3]), 1.)
        self.assertEqual(float(scores[1]), float(scores[2]))
        self.assertLess(float(scores[1]), .1)


if __name__ == "__main__":
    unittest.main()
