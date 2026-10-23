import unittest
from unittest.mock import Mock, patch

import numpy as np

from deployment import robot_runner as r


class WorkspaceTests(unittest.TestCase):
    def target(self, left_y, right_y=0):
        sdk = Mock()
        sdk.inverse_kinematics.return_value = np.zeros(6)
        sdk.forward_kinematics.return_value = np.zeros(6)
        action = np.array([.2, left_y, 0, 0, 0, 0, 1, 0,
                           .2, right_y, 0, 0, 0, 0, 1, 0])
        feedback = {s: np.zeros(7) for s in ("left", "right")}
        tcp = {"left": action[:8], "right": action[8:]}
        with patch.object(r, "validate_ik_solution"):
            return r.guarded_bimanual_targets(sdk, action, feedback, tcp,
                r.GuardLimits(), np.array([.15, 0, 0]), .8)

    def test_left_expanded_boundary(self):
        for y in (-.30, -.25, -.20):
            self.assertEqual(set(self.target(y)), {"left", "right"})

    def test_outside_still_rejected(self):
        for left, right in ((-.3001, 0), (.4001, 0), (0, -.2001)):
            with self.assertRaisesRegex(ValueError, "y outside workspace"):
                self.target(left, right)

    def test_single_arm_default_unchanged(self):
        self.assertEqual(r.GuardLimits().y_range, (-.20, .40))


if __name__ == "__main__":
    unittest.main()
