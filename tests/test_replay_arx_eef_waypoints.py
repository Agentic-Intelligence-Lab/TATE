from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from real2sim.move_arx_tcp_safe import rotation_from_rpy
from real2sim.replay_arx_eef_waypoints import (
    _execute_multi_locked,
    Waypoint,
    build_joint_approach,
    command_approach_joint_target,
    execute,
    load_corrected_eef,
    matrix_to_rpy,
    select_from_first_grasp_events,
    select_waypoints,
    select_synchronized_waypoints,
    solve_chain,
    tcp_target_to_sdk_flange,
    wait_for_approach_target,
)


class EefWaypointTests(unittest.TestCase):
    def test_matrix_to_rpy_round_trip(self) -> None:
        for rpy in (
            np.asarray([0.2, -0.3, 0.4]),
            np.asarray([-1.0, 0.5, 2.0]),
        ):
            recovered = matrix_to_rpy(rotation_from_rpy(rpy))
            self.assertTrue(np.allclose(rotation_from_rpy(recovered), rotation_from_rpy(rpy)))

    def test_tcp_target_is_converted_to_sdk_flange_target(self) -> None:
        tcp = np.eye(4)
        tcp[:3, :3] = rotation_from_rpy(np.asarray([0.0, 0.0, np.pi / 2.0]))
        tcp[:3, 3] = np.asarray([0.2, 0.3, 0.4])
        flange = tcp_target_to_sdk_flange(tcp, np.asarray([0.15, 0.0, 0.0]))
        self.assertTrue(np.allclose(flange[:3, 3], [0.2, 0.15, 0.4]))
        self.assertTrue(np.allclose(flange[:3, :3], tcp[:3, :3]))

    def test_sampling_keeps_endpoints(self) -> None:
        available = [
            Waypoint(i, i / 30.0, np.eye(4), np.zeros(6)) for i in range(20)
        ]
        selected = select_waypoints(available, 5, None, None)
        self.assertEqual([point.frame_index for point in selected], [0, 5, 10, 14, 19])

    def test_synchronized_sampling_keeps_grasp_event_edges(self) -> None:
        right = [
            Waypoint(i, float(i), np.eye(4), np.zeros(6), grasp_state=int(i >= 7))
            for i in range(20)
        ]
        left = [
            Waypoint(i, float(i), np.eye(4), np.zeros(6), grasp_state=int(i >= 13))
            for i in range(20)
        ]
        selected = select_synchronized_waypoints(
            {"left": left, "right": right}, 3, None, None
        )
        expected_required = {0, 6, 7, 10, 12, 13, 19}
        self.assertTrue(expected_required.issubset({w.frame_index for w in selected["left"]}))
        self.assertEqual(
            [w.frame_index for w in selected["left"]],
            [w.frame_index for w in selected["right"]],
        )

    def test_each_arm_starts_at_its_own_first_grasp_event(self) -> None:
        right = [
            Waypoint(
                i,
                float(i),
                np.eye(4),
                np.zeros(6),
                grasp_state=int(3 <= i < 8),
            )
            for i in range(12)
        ]
        left = [
            Waypoint(
                i,
                float(i),
                np.eye(4),
                np.zeros(6),
                grasp_state=int(5 <= i < 10),
            )
            for i in range(12)
        ]
        selected = select_from_first_grasp_events(
            {"left": left, "right": right}, 4, None, None
        )
        self.assertEqual(selected["left"][0].frame_index, 5)
        self.assertEqual(selected["right"][0].frame_index, 3)
        self.assertEqual(
            [point.frame_index for point in selected["left"][1:]],
            [point.frame_index for point in selected["right"][1:]],
        )
        # Later open events remain present after the independent entry poses.
        self.assertIn(8, [point.frame_index for point in selected["right"]])
        self.assertIn(10, [point.frame_index for point in selected["left"]])

    def test_joint_approach_is_split_into_bounded_steps(self) -> None:
        args = SimpleNamespace(
            max_approach_total_joint_rad=2.0,
            allow_long_approach=False,
            approach_joint_step_rad=0.2,
            max_approach_steps=256,
            approach_translation_step_m=0.02,
            approach_rotation_step_deg=5.0,
            ik_position_tolerance_m=0.002,
            ik_rotation_tolerance_deg=1.0,
        )

        class FakeKinematics:
            def forward_kinematics(self, q):
                return np.asarray([q[0] * 0.05, 0.0, 0.0, 0.0, 0.0, q[1] * 0.1])

        start_q = np.zeros(6)
        target_q = np.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        approach = build_joint_approach(
            start_q, target_q, np.zeros(6), FakeKinematics(), args
        )
        self.assertEqual(len(approach), 5)
        previous_q = start_q
        for q, _pose in approach:
            self.assertLessEqual(np.max(np.abs(q - previous_q)), 0.2 + 1e-12)
            previous_q = q
        self.assertTrue(np.allclose(approach[-1][0], target_q))

    def test_approach_timeout_retries_the_same_target(self) -> None:
        target = np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])

        class FakeArm:
            fault = None

            def __init__(self):
                self.commands = []

            def set_joint_positions(self, positions, duration):
                self.commands.append((np.asarray(positions), duration))
                return True

            def get_joint_positions(self):
                return target - 0.01

        arm = FakeArm()
        args = SimpleNamespace(approach_retries=2)
        with patch(
            "real2sim.replay_arx_eef_waypoints.wait_for_approach_target",
            side_effect=[RuntimeError("max_joint_error=0.04635rad"), None],
        ):
            command_approach_joint_target(
                arm,
                target,
                np.zeros(6),
                3.0,
                args,
                label="right approach step 4/8",
            )
        self.assertEqual(len(arm.commands), 2)
        self.assertTrue(np.allclose(arm.commands[0][0], target))
        self.assertTrue(np.allclose(arm.commands[1][0], target))

    def test_approach_accepts_fk_pose_when_joint_residual_is_larger(self) -> None:
        target_q = np.zeros(6)
        target_pose = np.asarray([0.2, 0.1, 0.3, 0.1, -0.2, 0.3])

        class FakeArm:
            fault = None

            def get_joint_positions(self):
                return np.full(6, 0.05)

            def get_ee_pose_xyzrpy(self):
                return target_pose

        args = SimpleNamespace(
            timeout_padding=0.01,
            poll_period=0.001,
            approach_feedback_position_tolerance_m=0.006,
            approach_feedback_rotation_tolerance_deg=5.0,
            feedback_joint_tolerance_rad=0.04,
        )
        wait_for_approach_target(FakeArm(), target_q, target_pose, 0.01, args)

    def test_long_approach_bypasses_only_total_joint_span(self) -> None:
        args = SimpleNamespace(
            max_approach_total_joint_rad=0.2,
            allow_long_approach=True,
            approach_joint_step_rad=0.1,
            max_approach_steps=256,
            approach_translation_step_m=0.02,
            approach_rotation_step_deg=5.0,
            ik_position_tolerance_m=0.002,
            ik_rotation_tolerance_deg=1.0,
        )

        class FakeKinematics:
            def forward_kinematics(self, q):
                return np.asarray([q[0] * 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])

        approach = build_joint_approach(
            np.zeros(6),
            np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.zeros(6),
            FakeKinematics(),
            args,
        )
        self.assertEqual(len(approach), 10)

    def test_uncorrected_eef_is_rejected(self) -> None:
        payload = {
            "schema": "tate.dual_arm_eef",
            "schema_version": 2,
            "eef_coordinate_convention": {
                "pose_semantics": "arx_tcp",
                "tcp_orientation_applied": True,
                "per_side_frames": {"right": "right_flange_zero"},
            },
            "frames": [
                {
                    "idx": 0,
                    "ts": 0.0,
                    "hand_r": {"tcp_pose_eef_frame": np.eye(4).tolist()},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eef.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "real_anchor_correction"):
                load_corrected_eef(path, "right", allow_uncorrected=False)

    def test_offline_seed_to_first_ik_is_not_a_motion_step(self) -> None:
        first_q = np.asarray([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        second_q = np.asarray([0.0, 1.1, 1.0, 0.0, 0.0, 0.0])
        poses = [
            np.asarray([0.10, 0.0, 0.20, 0.0, 0.0, 0.0]),
            np.asarray([0.11, 0.0, 0.20, 0.0, 0.0, 0.0]),
        ]
        waypoints = [Waypoint(i, float(i), np.eye(4), pose) for i, pose in enumerate(poses)]

        class FakeKinematics:
            def __init__(self):
                self.index = 0

            def inverse_kinematics(self, pose, q_init=None):
                q = (first_q, second_q)[self.index]
                self.index += 1
                return q

            def forward_kinematics(self, _q):
                return poses[self.index - 1]

        args = SimpleNamespace(
            max_first_translation_m=0.10,
            max_first_rotation_deg=30.0,
            max_segment_translation_m=0.10,
            max_segment_rotation_deg=35.0,
            max_first_joint_step_rad=0.20,
            max_segment_joint_step_rad=0.20,
            ik_position_tolerance_m=0.002,
            ik_rotation_tolerance_deg=1.0,
        )
        solutions = solve_chain(
            waypoints,
            np.zeros(6),
            FakeKinematics(),
            args,
            current_pose=None,
        )
        self.assertTrue(np.allclose(solutions[0], first_q))
        self.assertTrue(np.allclose(solutions[1], second_q))

    def test_normal_execution_never_calls_protect(self) -> None:
        target = np.asarray([0.1, 0.0, 0.2, 0.0, 0.0, 0.0])
        waypoints = [Waypoint(i, float(i), np.eye(4), target) for i in range(2)]

        class FakeArm:
            def __init__(self):
                self.fault = None
                self.protect_calls = 0
                self.close_calls = 0
                self.home_calls = 0
                self.set_calls = 0
                self.last_target = target

            def get_ee_pose_xyzrpy(self):
                return self.last_target

            def get_joint_positions(self):
                return np.asarray([0.0, 1.2, 1.5, 0.0, 0.0, 0.0])

            def inverse_kinematics(self, pose, q_init=None):
                self.last_target = np.asarray(pose)
                return self.get_joint_positions()

            def forward_kinematics(self, _q):
                return self.last_target

            def set_ee_pose_xyzrpy(self, pose, duration=0.0):
                self.last_target = np.asarray(pose)
                self.set_calls += 1
                return True

            def go_home(self, duration, wait):
                self.home_calls += 1
                return True

            def protect_mode(self):
                self.protect_calls += 1
                return True

            def close(self):
                self.close_calls += 1

        arm = FakeArm()

        class Loading:
            def __init__(self, _message):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        sdk = SimpleNamespace(Loading=Loading, SingleArm=lambda _config: arm)
        args = SimpleNamespace(
            yes_i_understand_risk=True,
            can_port=None,
            side="right",
            arm_type=2,
            max_first_translation_m=0.10,
            max_first_rotation_deg=30.0,
            max_segment_translation_m=0.10,
            max_segment_rotation_deg=35.0,
            max_first_joint_step_rad=0.70,
            max_segment_joint_step_rad=0.50,
            ik_position_tolerance_m=0.002,
            ik_rotation_tolerance_deg=1.0,
            segment_duration=3.0,
            approach_duration=1.5,
            approach_control_mode="segmented",
            approach_retries=2,
            approach_joint_step_rad=0.2,
            approach_translation_step_m=0.02,
            approach_rotation_step_deg=5.0,
            max_approach_total_joint_rad=2.0,
            allow_long_approach=False,
            max_approach_total_translation_m=0.30,
            max_approach_total_rotation_deg=90.0,
            max_approach_steps=256,
            timeout_padding=1.0,
            feedback_position_tolerance_m=0.006,
            feedback_rotation_tolerance_deg=3.0,
            approach_feedback_position_tolerance_m=0.006,
            approach_feedback_rotation_tolerance_deg=5.0,
            feedback_joint_tolerance_rad=0.04,
            poll_period=0.01,
            home_duration=5.0,
        )
        with patch("real2sim.replay_arx_eef_waypoints.controller_processes", return_value=[]), patch(
            "builtins.input", return_value=""
        ):
            execute(args, sdk, waypoints)
        self.assertEqual(arm.set_calls, 2)
        self.assertEqual(arm.home_calls, 1)
        self.assertEqual(arm.protect_calls, 0)
        self.assertEqual(arm.close_calls, 1)

    def test_dual_execution_replays_grasp_edges_without_protect(self) -> None:
        target = np.asarray([0.1, 0.0, 0.2, 0.0, 0.0, 0.0])
        matrix = np.eye(4)
        waypoints = {
            side: [
                Waypoint(0, 0.0, matrix, target, grasp_state=0),
                Waypoint(1, 1.0, matrix, target, grasp_state=1),
            ]
            for side in ("left", "right")
        }

        class FakeArm:
            def __init__(self):
                self.fault = None
                self.pose = target.copy()
                self.q = np.asarray([0.0, 1.2, 1.5, 0.0, 0.0, 0.0])
                self.gripper = []
                self.set_calls = self.home_calls = self.protect_calls = self.close_calls = 0

            def get_ee_pose_xyzrpy(self): return self.pose
            def get_joint_positions(self): return self.q
            def inverse_kinematics(self, pose, q_init=None): return self.q
            def forward_kinematics(self, _q): return self.pose
            def set_ee_pose_xyzrpy(self, pose, duration=0.0):
                self.pose = np.asarray(pose); self.set_calls += 1; return True
            def set_gripper_pos(self, value): self.gripper.append(value); return True
            def go_home(self, duration, wait): self.home_calls += 1; return True
            def protect_mode(self): self.protect_calls += 1; return True
            def close(self): self.close_calls += 1

        arms = {"left": FakeArm(), "right": FakeArm()}

        class Loading:
            def __init__(self, _message): pass
            def __enter__(self): return self
            def __exit__(self, *_args): return False

        sdk = SimpleNamespace(
            Loading=Loading,
            SingleArm=lambda config: arms["left" if config["can_port"] == "can1" else "right"],
        )
        args = SimpleNamespace(
            left_can="can1", right_can="can3", can_port=None, arm_type=2,
            seed_q=[0.0, 1.2, 1.5, 0.0, 0.0, 0.0],
            max_first_translation_m=0.10, max_first_rotation_deg=30.0,
            max_segment_translation_m=0.10, max_segment_rotation_deg=35.0,
            max_first_joint_step_rad=0.70, max_segment_joint_step_rad=0.50,
            ik_position_tolerance_m=0.002, ik_rotation_tolerance_deg=1.0,
            max_approach_total_translation_m=0.30, max_approach_total_rotation_deg=90.0,
            max_approach_total_joint_rad=2.0, approach_joint_step_rad=0.2,
            approach_translation_step_m=0.02, approach_rotation_step_deg=5.0,
            max_approach_steps=256,
            approach_duration=1.5,
            approach_control_mode="segmented",
            approach_retries=2,
            allow_long_approach=False,
            feedback_joint_tolerance_rad=0.04,
            feedback_position_tolerance_m=0.006, feedback_rotation_tolerance_deg=3.0,
            approach_feedback_position_tolerance_m=0.006,
            approach_feedback_rotation_tolerance_deg=5.0,
            timeout_padding=1.0, poll_period=0.001, segment_duration=0.01,
            replay_grasp=True, gripper_open=-3.0, gripper_closed=0.0,
            gripper_settle_s=0.001, home_duration=5.0,
        )
        with patch("real2sim.replay_arx_eef_waypoints.controller_processes", return_value=[]), patch(
            "builtins.input", return_value=""
        ):
            _execute_multi_locked(args, sdk, waypoints)
        for arm in arms.values():
            self.assertEqual(arm.set_calls, 1)
            self.assertEqual(arm.gripper, [-3.0, 0.0])
            self.assertEqual(arm.home_calls, 1)
            self.assertEqual(arm.protect_calls, 0)
            self.assertEqual(arm.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
