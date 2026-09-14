from __future__ import annotations

import argparse
import unittest

import numpy as np

from real2sim.move_arx_tcp_safe import (
    pose_error,
    resolve_target,
    rotation_distance_deg,
    validate_ik,
    validate_target_change,
)


class SafeTcpMathTests(unittest.TestCase):
    def test_rotation_distance_wraps_rpy(self) -> None:
        self.assertAlmostEqual(
            rotation_distance_deg(
                np.asarray([0.0, 0.0, np.pi - 0.01]),
                np.asarray([0.0, 0.0, -np.pi + 0.01]),
            ),
            np.degrees(0.02),
            places=8,
        )

    def test_pose_error(self) -> None:
        pos, rot = pose_error(
            np.zeros(6), np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, np.pi / 180.0])
        )
        self.assertAlmostEqual(pos, 0.01)
        self.assertAlmostEqual(rot, 1.0)

    def test_delta_requires_current_pose(self) -> None:
        args = argparse.Namespace(target=None, delta=[0.01, 0, 0, 0, 0, 0])
        with self.assertRaisesRegex(RuntimeError, "current-pose"):
            resolve_target(args, None)

    def test_target_change_limits(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "translation"):
            validate_target_change(
                np.zeros(6),
                np.asarray([0.021, 0, 0, 0, 0, 0]),
                max_translation_m=0.02,
                max_rotation_deg=5.0,
            )

    def test_ik_checks_joint_step(self) -> None:
        args = argparse.Namespace(
            max_joint_step_rad=0.1,
            ik_position_tolerance_m=0.002,
            ik_rotation_tolerance_deg=1.0,
        )
        with self.assertRaisesRegex(RuntimeError, "joint step"):
            validate_ik(
                np.asarray([0.2, 1.0, 1.0, 0, 0, 0]),
                np.asarray([0.0, 1.0, 1.0, 0, 0, 0]),
                np.zeros(6),
                np.zeros(6),
                args,
            )


if __name__ == "__main__":
    unittest.main()
