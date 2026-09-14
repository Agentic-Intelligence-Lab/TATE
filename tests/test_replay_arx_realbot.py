from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from real2sim.replay_arx_realbot import (
    JointTrajectory,
    SideTrajectory,
    as_abs,
    execute_trajectory,
    update_tracking_feedback,
)


class FakeArm:
    def __init__(self, q: np.ndarray, gripper: float) -> None:
        self.q = np.asarray(q, dtype=np.float64).copy()
        self.gripper = float(gripper)
        self.fault = None
        self.home_calls = 0
        self.protect_calls = 0
        self.close_calls = 0

    def set_joint_positions(self, positions, duration=0.0):
        self.q = np.asarray(positions, dtype=np.float64).copy()
        return True

    def set_gripper_pos(self, value):
        self.gripper = float(value)
        return True

    def get_joint_positions(self):
        return np.concatenate([self.q, [self.gripper]])

    def go_home(self, duration, wait=True):
        self.home_calls += 1
        self.q[:] = 0.0
        return True

    def protect_mode(self):
        self.protect_calls += 1
        return True

    def close(self):
        self.close_calls += 1


class RealbotReplayTests(unittest.TestCase):
    def test_relative_data_path_prefers_current_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.parquet"
            path.touch()
            with patch("pathlib.Path.cwd", return_value=Path(directory)):
                self.assertEqual(as_abs("episode.parquet"), path.resolve())

    def test_tracking_feedback_requires_consecutive_bad_frames(self) -> None:
        arm = FakeArm(np.full(6, 0.3), 0.0)
        target = {"right": (np.zeros(6), 0.0)}
        args = SimpleNamespace(
            max_tracking_joint_error_rad=0.25,
            max_tracking_gripper_error=0.75,
            max_tracking_error_frames=2,
        )
        consecutive = {"right": 0}
        maxima = {"right": (0.0, 0.0)}
        update_tracking_feedback(
            {"right": arm}, target, args, consecutive, maxima
        )
        with self.assertRaisesRegex(RuntimeError, "2 consecutive frames"):
            update_tracking_feedback(
                {"right": arm}, target, args, consecutive, maxima
            )

    def test_normal_completion_holds_then_homes_without_protect(self) -> None:
        q = np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        gripper = -1.0
        arm = FakeArm(q, gripper)
        frames = [{"right": (q.copy(), gripper)} for _ in range(2)]
        trajectory = JointTrajectory(
            time_s=np.asarray([0.0, 0.1]),
            sides={
                "right": SideTrajectory(
                    arm_qpos=np.stack([q, q]),
                    gripper_pos=np.asarray([gripper, gripper]),
                )
            },
            source="test",
        )
        args = SimpleNamespace(
            execute=True,
            yes_i_understand_risk=True,
            rate=10.0,
            ramp_time=0.01,
            max_joint_speed=0.6,
            max_gripper_speed=1.0,
            progress_every=0,
            max_tracking_joint_error_rad=0.25,
            max_tracking_gripper_error=0.75,
            max_tracking_error_frames=10,
            final_joint_tolerance_rad=0.08,
            final_gripper_tolerance=0.20,
            final_feedback_timeout=1.0,
            feedback_poll_period=0.01,
            home_duration=1.0,
        )

        class FakeSingleArm:
            @staticmethod
            def hold(_duration):
                raise AssertionError("hold/protect cleanup must not run normally")

        with patch(
            "real2sim.replay_arx_realbot.build_real_arms",
            return_value=(FakeSingleArm, {"right": arm}),
        ), patch("real2sim.replay_arx_realbot.time.sleep"), patch(
            "builtins.input", return_value=""
        ):
            execute_trajectory(trajectory, frames, args)

        self.assertEqual(arm.home_calls, 1)
        self.assertEqual(arm.protect_calls, 0)
        self.assertEqual(arm.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
