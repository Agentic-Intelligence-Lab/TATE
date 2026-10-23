"""Lifecycle regressions; all controllers, cameras and inference are mocked."""
import io
import json
import signal
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from deployment import robot_runner as r


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.control = r.BimanualControl()
        self.arms = {s: Mock(fault=None) for s in ("left", "right")}
        for arm in self.arms.values():
            arm.get_joint_positions.return_value = np.zeros(7)
        self.args = SimpleNamespace(
            fps=30, n_action_steps=50, gripper_threshold=.8, prompt="test",
            start_pose=False, tcp_offset_m=None, max_runtime_seconds=5,
            max_policy_latency=5, policy_url="unused", left_can="can1", right_can="can3",
        )

    def test_hold_and_shutdown_are_different(self):
        self.control.on_signal(signal.SIGUSR2)
        self.assertTrue(self.control.hold)
        self.assertFalse(self.control.shutdown)
        self.control.on_signal(signal.SIGHUP)
        self.assertTrue(self.control.reset)
        self.control.on_signal(signal.SIGTERM)
        self.assertTrue(self.control.shutdown)

    def resume_line(self):
        return "RESUME " + json.dumps(dict(fps=20, n_action_steps=25,
            gripper_threshold=.7, tcp_offset_m=[.15, 0, 0], start_pose=True, prompt="new"))

    def test_hold_keeps_fixed_snapshot_then_resume(self):
        # A drifting encoder must not turn the held goal into a drifting goal.
        for arm in self.arms.values():
            arm.get_joint_positions.side_effect = [np.zeros(7), np.ones(7)]
        with patch.object(r.select, "select", side_effect=[([], [], []), ([1], [], [])]), \
             patch.object(r.sys, "stdin", io.StringIO(self.resume_line())):
            self.assertTrue(r.wait_bimanual_resume(self.arms, self.args, self.control))
        self.assertTrue(self.args.start_pose)
        self.assertEqual(self.args.prompt, "new")
        for arm in self.arms.values():
            self.assertEqual(arm.get_joint_positions.call_count, 1)
            self.assertEqual(arm.set_joint_positions.call_count, 2)
            arm.close.assert_not_called()

    def test_reset_returns_home_and_keeps_session(self):
        self.control.reset = True
        events = []
        for side, arm in self.arms.items():
            feedback = np.zeros(7)
            arm.get_joint_positions.side_effect = lambda f=feedback: f.copy()
            def grip(value, duration=0, f=feedback, s=side):
                f[6] = value
                events.append((s, "grip", value))
                return True
            arm.set_gripper_pos.side_effect = grip
            arm.go_home.side_effect = lambda *a, s=side, **kw: events.append((s, "home")) or True
        with patch.object(r.select, "select", return_value=([1], [], [])), \
             patch.object(r.sys, "stdin", io.StringIO(self.resume_line())):
            self.assertTrue(r.wait_bimanual_resume(self.arms, self.args, self.control))
        for arm in self.arms.values():
            arm.go_home.assert_called_once_with(5.0, wait=False, gripper=False)
            arm.close.assert_not_called()
            arm.protect_mode.assert_not_called()
        self.assertEqual(events[:4], [("left", "grip", r.GRIPPER_OPEN_RAW),
            ("right", "grip", r.GRIPPER_OPEN_RAW), ("left", "home"), ("right", "home")])
        self.assertEqual(events[4:6], [("left", "grip", r.GRIPPER_CLOSED_RAW),
            ("right", "grip", r.GRIPPER_CLOSED_RAW)])

    def test_prepare_starts_without_preflight_or_run(self):
        with patch.object(r, "infer") as infer, patch.object(r, "wait_for_run") as wait:
            self.assertTrue(r.prepare_bimanual(self.arms, None, None, self.args,
                r.GuardLimits(), np.array([.15, 0, 0]), self.control))
            infer.assert_not_called()
            wait.assert_not_called()

    def test_open_failure_does_not_home(self):
        self.arms["left"].set_gripper_pos.return_value = False
        with self.assertRaises(RuntimeError):
            r.reset_bimanual_home(self.arms, self.control)
        for arm in self.arms.values():
            arm.go_home.assert_not_called()

    def test_recorded_start_still_requires_move_and_run(self):
        self.args.start_pose = True
        targets = {s: (np.zeros(6), .1) for s in self.arms}
        with patch.object(r, "bimanual_start_pose", return_value=(targets, {"frame_index": 120})), \
             patch.object(r, "wait_for_start_pose", return_value=True) as move, \
             patch.object(r, "move_bimanual_to_start_pose"), \
             patch.object(r, "wait_for_run", return_value=True) as run:
            self.assertTrue(r.prepare_bimanual(self.arms, None, None, self.args,
                r.GuardLimits(), np.array([.15, 0, 0]), self.control))
            move.assert_called_once()
            run.assert_called_once()

    def test_reset_interruption_never_closes_grippers(self):
        def interrupt(*args):
            self.control.hold = True
        with patch.object(r, "wait_bimanual_grippers", side_effect=interrupt):
            with self.assertRaises(RuntimeError):
                r.reset_bimanual_home(self.arms, self.control)
        for arm in self.arms.values():
            arm.go_home.assert_not_called()
            arm.set_gripper_pos.assert_called_once_with(r.GRIPPER_OPEN_RAW, duration=0.0)

    def test_hold_failure_still_attempts_other_arm(self):
        self.arms["left"].set_joint_positions.return_value = False
        with self.assertRaises(RuntimeError):
            r.hold_bimanual_position(self.arms)
        self.arms["right"].set_joint_positions.assert_called_once()

    def test_shutdown_exits_hold(self):
        self.control.shutdown = True
        self.assertFalse(r.wait_bimanual_resume(self.arms, self.args, self.control))

    def test_stop_during_ik_cancels_send(self):
        with patch.object(r, "guarded_bimanual_targets", return_value={}), \
             patch.object(r, "send_bimanual_targets") as send:
            self.assertFalse(r.execute_bimanual_step(
                {}, None, None, None, None, None, None, .8, 0, should_cancel=lambda: True))
            send.assert_not_called()

    def exercise_session(self, error=False):
        rig = Mock()
        rig.status.return_value = {s: {"online": True} for s in r.CAMERA_SERIALS}
        def infer(*_):
            self.control.on_signal(signal.SIGUSR2)
            return np.zeros((50, 16)), .01
        hold_calls = []
        def hold(*_):
            # No cleanup on either operator stop or a recoverable exception.
            for arm in self.arms.values():
                arm.close.assert_not_called()
                self.assertEqual(arm.protect_mode.call_count, 1)  # startup only
            hold_calls.append(True)
            self.control.hold = False
            if len(hold_calls) == 1:
                return True
            self.control.shutdown = True
            return False
        with patch.object(r, "prepare_bimanual", return_value=True) as prepare, \
             patch.object(r, "read_bimanual_observation", return_value=({}, {}, {}),
                          side_effect=OSError("network down") if error else None), \
             patch.object(r, "infer", side_effect=infer), \
             patch.object(r, "wait_bimanual_resume", side_effect=hold), \
             patch.object(r, "send_bimanual_targets") as send:
            self.assertEqual(r.run_bimanual(self.args, Mock(),
                Mock(side_effect=list(self.arms.values())), Mock(return_value=rig), self.control), 0)
            self.assertEqual(prepare.call_count, 2)
            self.assertEqual(len(hold_calls), 2)
            send.assert_not_called()
        for arm in self.arms.values():
            arm.close.assert_called_once()  # true shutdown only

    def test_stop_holds_and_prepares_again(self):
        self.exercise_session()

    def test_network_failure_holds_and_prepares_again(self):
        self.exercise_session(error=True)


if __name__ == "__main__":
    unittest.main()
