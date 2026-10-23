"""No hardware: exercise paired IK rejection and subsequent action execution."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from deployment import robot_runner as runner


class BimanualIKSkipTest(unittest.TestCase):
    def setUp(self):
        self.sdk = Mock()
        self.sdk.forward_kinematics.return_value = np.zeros(6)
        self.feedback = {s: np.zeros(7) for s in ("left", "right")}
        self.tcp = {s: np.zeros(8) for s in ("left", "right")}
        self.action = np.zeros(16)
        self.arms = {s: Mock(fault=None) for s in ("left", "right")}
        for name, kwargs in (
            ("tcp_action_to_flange", {"return_value": (np.zeros(6), 0.1)}),
            ("validate_ik_solution", {}),
        ):
            p = patch.object(runner, name, **kwargs)
            p.start()
            self.addCleanup(p.stop)

    def step(self, index):
        return runner.execute_bimanual_step(
            self.arms, self.sdk, self.action, self.feedback, self.tcp,
            runner.GuardLimits(), np.array([.15, 0, 0]), .8, index,
        )

    def test_one_failed_step_in_fifty(self):
        for failed_side, failed_index in (("left", 0), ("right", 0), ("left", 17), ("right", 17)):
            with self.subTest(side=failed_side, index=failed_index):
                for arm in self.arms.values():
                    arm.reset_mock()
                calls = []
                for step in range(50):
                    if step == failed_index:
                        if failed_side == "right":
                            calls.append(np.zeros(6))
                        calls.append(ValueError("IK 无解: 残差 2.03e-04"))
                    else:
                        calls.extend([np.zeros(6), np.zeros(6)])
                self.sdk.inverse_kinematics.side_effect = calls
                results = [self.step(i) for i in range(50)]
                self.assertEqual(sum(results), 49)
                self.assertFalse(results[failed_index])
                self.assertTrue(results[failed_index + 1])
                for arm in self.arms.values():
                    self.assertEqual(arm.set_joint_positions.call_count, 49)
                    self.assertEqual(arm.set_gripper_pos.call_count, 49)

    def test_unrelated_solver_errors_propagate(self):
        for exc in (ValueError("invalid input"), RuntimeError("SDK fault")):
            self.sdk.inverse_kinematics.side_effect = exc
            with self.assertRaises(type(exc)):
                self.step(0)
        for arm in self.arms.values():
            arm.set_joint_positions.assert_not_called()

    def test_safety_failure_is_not_skipped(self):
        self.sdk.inverse_kinematics.return_value = np.zeros(6)
        with patch.object(runner, "validate_ik_solution", side_effect=ValueError("IK target outside joint limits")):
            with self.assertRaises(ValueError):
                self.step(0)
        for arm in self.arms.values():
            arm.set_joint_positions.assert_not_called()

    def test_command_failure_is_not_skipped(self):
        self.sdk.inverse_kinematics.return_value = np.zeros(6)
        self.arms["left"].set_joint_positions.return_value = False
        with self.assertRaises(RuntimeError):
            self.step(0)


if __name__ == "__main__":
    unittest.main()
