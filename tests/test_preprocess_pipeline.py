import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from preprocess.Hand2Gripper import FingerCenter, HumanEgo, Qwen, make_hand2gripper
from preprocess.PipelineIO import (
    enforce_cache_kinematic_limits,
    export_eef_from_cache,
    load_arm_camera_transforms,
    load_yaml,
    select_intrinsics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = REPO_ROOT / "cfg" / "preprocess" / "base" / "RealSenseD405.yaml"
EEF_CONFIG_PATH = REPO_ROOT / "cfg" / "preprocess" / "base" / "EEFExport.yaml"


def synthetic_humanego_keypoints(tip_ratio: float = 0.4) -> np.ndarray:
    """Non-degenerate HumanEgo/Aria-order hand with a 0.1 m palm size."""
    keypoints = np.zeros((21, 3), dtype=np.float64)
    keypoints[:] = [0.0, 0.05, 0.6]
    keypoints[5] = [0.0, 0.0, 0.6]       # wrist
    keypoints[6] = [-0.035, 0.045, 0.6]  # thumb MCP
    keypoints[8] = [0.03, 0.075, 0.6]    # index MCP
    keypoints[11] = [0.0, 0.1, 0.6]      # middle MCP
    keypoints[14] = [-0.01, 0.09, 0.6]   # ring MCP
    keypoints[17] = [-0.03, 0.075, 0.6]  # pinky MCP
    keypoints[20] = [0.0, 0.06, 0.6]     # palm center
    tip_distance = 0.1 * float(tip_ratio)
    keypoints[0] = [-0.5 * tip_distance, 0.13, 0.6]
    keypoints[1] = [0.5 * tip_distance, 0.13, 0.6]
    return keypoints


class PreprocessPipelineTest(unittest.TestCase):
    def setUp(self):
        self.calibration = load_yaml(CALIBRATION_PATH)

    def test_exact_intrinsic_profiles(self):
        K_640, d_640, profile_640 = select_intrinsics(self.calibration, 640, 480)
        K_720, d_720, profile_720 = select_intrinsics(self.calibration, 1280, 720)
        self.assertEqual(profile_640, "640x480")
        self.assertEqual(profile_720, "1280x720")
        self.assertAlmostEqual(K_640[0, 0], 393.030548)
        self.assertAlmostEqual(K_720[0, 0], 657.06060791)
        self.assertEqual(d_640.shape, (5,))
        self.assertEqual(d_720.shape, (5,))

    def test_unknown_resolution_fails(self):
        with self.assertRaisesRegex(ValueError, "No exact camera intrinsic profile"):
            select_intrinsics(self.calibration, 848, 480)

    def test_export_uses_independent_arm_transforms(self):
        keypoints = synthetic_humanego_keypoints()
        hand = {
            "confidence": 0.9,
            "keypoints_3d_cam": keypoints.tolist(),
        }
        cache = {
            "schema": "tate.wilor_hands",
            "schema_version": 1,
            "source_video": "test.mp4",
            "fps": 30.0,
            "width": 640,
            "height": 480,
            "K": np.eye(3).tolist(),
            "d": [0, 0, 0, 0, 0],
            "keypoint_order": "aria_humanego_21",
            "frames": [{"idx": 0, "ts": 0, "hand_r": hand, "hand_l": hand}],
        }
        expected, _ = load_arm_camera_transforms(self.calibration)
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "hands.json"
            output_path = Path(tmp) / "eef.json"
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            result = export_eef_from_cache(
                cache_path, CALIBRATION_PATH, output_path, EEF_CONFIG_PATH
            )

        right_record = result["frames"][0]["hand_r"]
        left_record = result["frames"][0]["hand_l"]
        right = np.asarray(right_record["eef_pose_world"])
        left = np.asarray(left_record["eef_pose_world"])
        np.testing.assert_allclose(
            right, expected["right"] @ np.asarray(right_record["tcp_pose_cam"])
        )
        np.testing.assert_allclose(
            left, expected["left"] @ np.asarray(left_record["tcp_pose_cam"])
        )
        self.assertFalse(np.allclose(right, left))
        self.assertEqual(result["schema_version"], 2)
        self.assertTrue(result["eef_coordinate_convention"]["tcp_orientation_applied"])

    def test_finger_center_axes(self):
        keypoints = synthetic_humanego_keypoints()
        target = FingerCenter().from_hand_record(
            {"keypoints_3d_cam": keypoints.tolist()}, is_right=True
        )
        self.assertIsNotNone(target)
        rotation = target.T_hand_in_cam[:3, :3]
        expected_jaw = keypoints[8] - keypoints[6]
        expected_jaw /= np.linalg.norm(expected_jaw)
        forward_seed = np.mean(keypoints[[8, 11, 14, 17]], axis=0) - keypoints[5]
        expected_forward = forward_seed - np.dot(forward_seed, expected_jaw) * expected_jaw
        expected_forward /= np.linalg.norm(expected_forward)
        np.testing.assert_allclose(rotation[:, 0], expected_jaw, atol=1e-10)
        np.testing.assert_allclose(rotation[:, 1], expected_forward, atol=1e-10)
        np.testing.assert_allclose(
            target.T_hand_in_cam[:3, 3], 0.5 * (keypoints[0] + keypoints[1])
        )
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_humanego_uses_upstream_thumb_index_mcp_forward_seed(self):
        keypoints = synthetic_humanego_keypoints()
        target = HumanEgo().from_hand_record(
            {"keypoints_3d_cam": keypoints.tolist()}, is_right=True
        )
        self.assertIsNotNone(target)
        jaw = keypoints[8] - keypoints[6]
        jaw /= np.linalg.norm(jaw)
        seed = 0.5 * (keypoints[6] + keypoints[8]) - keypoints[5]
        expected_forward = seed - np.dot(seed, jaw) * jaw
        expected_forward /= np.linalg.norm(expected_forward)
        np.testing.assert_allclose(target.T_hand_in_cam[:3, 0], jaw, atol=1e-10)
        np.testing.assert_allclose(target.T_hand_in_cam[:3, 1], expected_forward, atol=1e-10)
        self.assertEqual(target.mode, "humanego")

    def test_qwen_virtual_tip_and_handedness_match_lifego(self):
        keypoints = synthetic_humanego_keypoints()
        keypoints[2] = [0.08, 0.14, 0.6]
        right = Qwen().from_hand_record({"keypoints_3d_cam": keypoints.tolist()}, is_right=True)
        left = Qwen().from_hand_record({"keypoints_3d_cam": keypoints.tolist()}, is_right=False)
        self.assertIsNotNone(right)
        self.assertIsNotNone(left)
        virtual_tip = 0.7 * keypoints[1] + 0.3 * keypoints[2]
        np.testing.assert_allclose(right.T_hand_in_cam[:3, 3], 0.5 * (keypoints[0] + virtual_tip))
        np.testing.assert_allclose(right.T_hand_in_cam[:3, 0], -left.T_hand_in_cam[:3, 0])
        self.assertEqual(right.mode, "qwen")
        self.assertIsInstance(make_hand2gripper("humanego"), HumanEgo)
        self.assertIsInstance(make_hand2gripper("qwen"), Qwen)

    def test_export_accepts_mode_override_and_records_it(self):
        keypoints = synthetic_humanego_keypoints()
        hand = {"confidence": 0.9, "keypoints_3d_cam": keypoints.tolist()}
        cache = {
            "schema": "tate.wilor_hands", "schema_version": 1, "fps": 30.0,
            "width": 640, "height": 480, "K": np.eye(3).tolist(), "d": [0] * 5,
            "keypoint_order": "aria_humanego_21",
            "frames": [{"idx": 0, "ts": 0, "hand_r": hand, "hand_l": None}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache_path, output_path = Path(tmp) / "hands.json", Path(tmp) / "eef.json"
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            result = export_eef_from_cache(
                cache_path, CALIBRATION_PATH, output_path, EEF_CONFIG_PATH,
                hand2gripper_mode="qwen",
            )
        self.assertEqual(result["hand2gripper"]["mode"], "qwen")
        self.assertEqual(result["frames"][0]["hand_r"]["hand2gripper_mode"], "qwen")

    def test_single_arm_export_masks_inactive_hand(self):
        keypoints = synthetic_humanego_keypoints()
        hand = {"confidence": 0.9, "keypoints_3d_cam": keypoints.tolist()}
        cache = {
            "schema": "tate.wilor_hands", "schema_version": 1, "fps": 30.0,
            "width": 640, "height": 480, "K": np.eye(3).tolist(), "d": [0] * 5,
            "keypoint_order": "aria_humanego_21",
            "frames": [{"idx": 0, "ts": 0, "hand_r": hand, "hand_l": hand}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache_path, output_path = Path(tmp) / "hands.json", Path(tmp) / "eef.json"
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            result = export_eef_from_cache(
                cache_path,
                CALIBRATION_PATH,
                output_path,
                EEF_CONFIG_PATH,
                arm_mode="single_arm",
                active_sides=["right"],
            )
        self.assertEqual(result["arm_mode"], "single_arm")
        self.assertEqual(result["active_sides"], ["right"])
        self.assertIsNotNone(result["frames"][0]["hand_r"])
        self.assertIsNone(result["frames"][0]["hand_l"])

    def test_grasp_hysteresis_holds_state_inside_band(self):
        converter = FingerCenter(
            grasp_close_ratio=0.45, grasp_open_ratio=0.55
        )
        states = []
        measured = []
        for ratio in (0.60, 0.50, 0.40, 0.50, 0.60):
            target = converter.from_hand_record(
                {"keypoints_3d_cam": synthetic_humanego_keypoints(ratio).tolist()},
                is_right=True,
            )
            states.append(target.grasp_state)
            measured.append(target.grasp_ratio)
        self.assertEqual(states, [0, 0, 1, 1, 0])
        np.testing.assert_allclose(measured, [0.60, 0.50, 0.40, 0.50, 0.60])

    def test_tcp_local_rotation_matches_existing_arx_ik_convention(self):
        extra = self.calibration["extra_transforms"]
        local = np.asarray(extra["T_hand_to_ee"]) @ np.asarray(
            extra["T_ee_axis_correct"]
        )
        expected = np.asarray(
            [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        np.testing.assert_allclose(local[:3, :3], expected)

    def test_cache_kinematic_limits_are_enforced(self):
        frames = []
        for idx, (x, angle) in enumerate(((0.0, 0.0), (1.0, np.pi / 2))):
            pose = np.eye(4)
            pose[:3, 3] = [x, 0.0, 0.0]
            pose[:3, :3] = [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
            frames.append(
                {
                    "idx": idx,
                    "ts": idx * 100_000_000,
                    "hand_r": {
                        "midpoint_pose_cam": pose.tolist(),
                        "midpoint_translation_cam": pose[:3, 3].tolist(),
                    },
                    "hand_l": None,
                }
            )
        payload = {"fps": 10.0, "frames": frames}
        enforce_cache_kinematic_limits(payload, linear_speed_limit=0.5, angular_speed_limit=1.0)
        first = np.asarray(frames[0]["hand_r"]["midpoint_pose_cam"])
        second = np.asarray(frames[1]["hand_r"]["midpoint_pose_cam"])
        self.assertAlmostEqual(np.linalg.norm(second[:3, 3] - first[:3, 3]), 0.05)
        relative_angle = np.arccos(
            np.clip((np.trace(first[:3, :3].T @ second[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
        )
        self.assertAlmostEqual(relative_angle, 0.1)


if __name__ == "__main__":
    unittest.main()
