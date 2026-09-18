from __future__ import annotations

import unittest

import numpy as np

from deployment.eef_math import (
    GRIPPER_CLOSED_RAW,
    GRIPPER_OPEN_RAW,
    GuardLimits,
    flange_to_tcp_state,
    tcp_action_to_flange,
)


class DeploymentGripperBinaryTests(unittest.TestCase):
    def test_feedback_is_binarized_with_adapter_threshold(self) -> None:
        open_state = flange_to_tcp_state(np.zeros(6), -2.61)
        closed_state = flange_to_tcp_state(np.zeros(6), -2.6)
        self.assertEqual(open_state[7], 0.0)
        self.assertEqual(closed_state[7], 1.0)

    def test_policy_action_commands_a_gripper_endpoint(self) -> None:
        current = flange_to_tcp_state(np.zeros(6), -3.4)
        target = np.asarray([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5001])
        _, closed = tcp_action_to_flange(target, current, GuardLimits())
        target[-1] = 0.5
        _, open_ = tcp_action_to_flange(target, current, GuardLimits())
        self.assertEqual(closed, GRIPPER_CLOSED_RAW)
        self.assertEqual(open_, GRIPPER_OPEN_RAW)

    def test_out_of_range_policy_gripper_action_is_safely_saturated(self) -> None:
        current = flange_to_tcp_state(np.zeros(6), -3.4)
        target = np.asarray([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, -0.01])
        _, open_ = tcp_action_to_flange(target, current, GuardLimits())
        target[-1] = 1.01
        _, closed = tcp_action_to_flange(target, current, GuardLimits())
        self.assertEqual(open_, GRIPPER_OPEN_RAW)
        self.assertEqual(closed, GRIPPER_CLOSED_RAW)


if __name__ == "__main__":
    unittest.main()
