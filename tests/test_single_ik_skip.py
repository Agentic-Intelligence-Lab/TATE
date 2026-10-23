"""Single-arm IK skipping without opening hardware."""
import ast
import inspect
import unittest
from unittest.mock import Mock, patch

import numpy as np
from deployment import robot_runner as r


class SingleIKSkipTests(unittest.TestCase):
    def setUp(self):
        self.arm = Mock(fault=None)
        self.sdk = Mock()
        self.sdk.forward_kinematics.return_value = np.zeros(6)
        self.sdk.inverse_kinematics.return_value = np.zeros(6)
        for name, kwargs in (("tcp_action_to_flange", {"return_value": (np.zeros(6), .1)}),
                             ("validate_ik_solution", {})):
            p = patch.object(r, name, **kwargs)
            p.start()
            self.addCleanup(p.stop)

    def step(self, index=0, cancel=False):
        return r.execute_single_step(self.arm, self.sdk, np.zeros(16), np.zeros(7),
            np.zeros(8), r.GuardLimits(), np.array([.15, 0, 0]), .8, index,
            should_cancel=lambda: cancel)

    def test_first_or_middle_step_is_skipped(self):
        for index in (0, 17):
            with self.subTest(index=index):
                self.arm.reset_mock()
                self.sdk.inverse_kinematics.side_effect = [
                    ValueError("IK 无解: test") if i == index else np.zeros(6) for i in range(50)]
                results = [self.step(i) for i in range(50)]
                self.assertFalse(results[index])
                self.assertTrue(results[index + 1])
                self.assertEqual(sum(results), 49)
                self.assertEqual(self.arm.set_joint_positions.call_count, 49)
                self.assertEqual(self.arm.set_gripper_pos.call_count, 49)

    def test_all_failed_no_commands(self):
        self.sdk.inverse_kinematics.side_effect = ValueError("IK 无解: test")
        with patch("builtins.print"):
            self.assertFalse(any(self.step(i) for i in range(50)))
        self.arm.set_joint_positions.assert_not_called()
        self.arm.set_gripper_pos.assert_not_called()

    def test_other_errors_are_not_skipped(self):
        for error in (ValueError("invalid SDK input"), RuntimeError("SDK fault")):
            self.sdk.inverse_kinematics.side_effect = error
            with self.assertRaises(type(error)):
                self.step()
        self.arm.set_joint_positions.assert_not_called()

    def test_safety_rejection_not_skipped(self):
        with patch.object(r, "validate_ik_solution", side_effect=ValueError("joint limits")):
            with self.assertRaises(ValueError):
                self.step()
        self.arm.set_joint_positions.assert_not_called()

    def test_stop_during_ik_cancels_send(self):
        self.assertFalse(self.step(cancel=True))
        self.arm.set_joint_positions.assert_not_called()

    def test_main_no_longer_calls_first_step_preflight(self):
        tree = ast.parse(inspect.getsource(r.main))
        calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertNotIn("preflight_target", calls)
        self.assertIn("execute_single_step", calls)


if __name__ == "__main__":
    unittest.main()
