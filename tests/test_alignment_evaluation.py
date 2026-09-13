from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from correction.apply import _closest_depth_on_ray, _stable_seed, run as apply_correction
from correction.anchor_ids import anchor_sort_key, normalize_anchor_id, resolve_anchor_frame
from correction.methods import propagate_anchor_displacements, propagate_anchor_values
from evaluation.aggregate import METRIC_FIELDS, aggregate_records, real_to_real_noise_floors
from evaluation.align import align_pair
from evaluation.events import event_sequence, extract_events
from evaluation.loaders import load_eef_json, load_real_fk_json
from evaluation.run_eval import _pairs, _validate_split, _verify_no_leakage
from evaluation.schemas import SchemaError


def _pose(x: float) -> list[list[float]]:
    value = np.eye(4)
    value[0, 3] = x
    return value.tolist()


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _eef_payload(frame_name: str = "right_flange_zero") -> dict:
    states = [1, 1, 0, 0, 1, 1, 0]
    return {
        "schema": "tate.dual_arm_eef",
        "schema_version": 2,
        "fps": 10,
        "total_frames": len(states),
        "eef_coordinate_convention": {
            "pose_semantics": "arx_tcp",
            "per_side_frames": {"left": "left_flange_zero", "right": frame_name},
        },
        "frames": [
            {
                "idx": index,
                "ts": index * 100_000_000,
                "hand_l": None,
                "hand_r": {
                    "pose_semantics": "arx_tcp",
                    "eef_frame": frame_name,
                    "tcp_pose_eef_frame": _pose(index * 0.01),
                    "grasp_state": state,
                    "grasp_ratio": float(state),
                },
            }
            for index, state in enumerate(states)
        ],
    }


def _real_payload(frame_name: str = "right_flange_zero") -> dict:
    states = [1, 1, 0, 0, 1, 1, 0]
    frames = []
    for index, state in enumerate(states):
        arms = {}
        for side in ("left", "right"):
            arms[side] = {
                "output_frame": "left_flange_zero" if side == "left" else frame_name,
                "T_tcp_in_output_frame": _pose(index * 0.01),
                "gripper": {"binary": state if side == "right" else 0, "continuous": float(state)},
            }
        frames.append({"idx": index, "frame_index": index, "timestamp_s": index / 10, "arms": arms})
    return {
        "schema": "tate.arx_real_flange_trajectory",
        "schema_version": 1,
        "source": {"episode_index": 0},
        "gripper_convention": {"binary_open": 0, "binary_closed": 1},
        "frames": frames,
    }


def _check_loaders_events_and_alignment(tmp_path: Path) -> None:
    ego = load_eef_json(_write(tmp_path / "eef.json", _eef_payload()), episode_id=0)
    real = load_real_fk_json(_write(tmp_path / "real.json", _real_payload()), episode_id=0)
    assert not ego.sides["left"].valid.any()
    assert event_sequence(extract_events(ego, "right")) == ["open", "close", "open"]
    config = {
        "method": "position_dtw",
        "sample_count": 20,
        "max_invalid_gap_s": 0.25,
        "dtw_window_ratio": 0.5,
        "segment": {"start_event": 0, "end_event": 2},
    }
    result = align_pair(ego, real, "right", config)
    assert np.isclose(result.normalized_cost, 0.0)


def _check_alignment_rejects_frame_mismatch(tmp_path: Path) -> None:
    ego = load_eef_json(_write(tmp_path / "eef.json", _eef_payload()), episode_id=0)
    real = load_real_fk_json(_write(tmp_path / "real.json", _real_payload("wrong_frame")), episode_id=0)
    with unittest.TestCase().assertRaisesRegex(SchemaError, "coordinate-frame mismatch"):
        align_pair(ego, real, "right", {"segment": {}, "sample_count": 10})


def _check_split_leakage_is_rejected(tmp_path: Path) -> None:
    output = _write(tmp_path / "real.json", _real_payload())
    with unittest.TestCase().assertRaisesRegex(ValueError, "split leakage"):
        _validate_split(
            {"task": "task", "calibration": [0], "eval": [0]},
            {"0": output},
            {"task": "task"},
            {},
        )


def _check_split_reserve_overlap_is_rejected(tmp_path: Path) -> None:
    output = _write(tmp_path / "real.json", _real_payload())
    with unittest.TestCase().assertRaisesRegex(ValueError, "split leakage/overlap"):
        _validate_split(
            {"task": "task", "calibration": [], "eval": [0], "reserve": [0]},
            {"0": output},
            {"task": "task"},
            {},
        )


def _check_anchor_propagation_preserves_endpoints_and_anchor() -> None:
    correction, _, _ = propagate_anchor_displacements(
        11, [5], np.asarray([[0.1, -0.2, 0.3]]), method="linear"
    )
    assert np.allclose(correction[0], 0.0)
    assert np.allclose(correction[5], [0.1, -0.2, 0.3])
    assert np.allclose(correction[-1], 0.0)


def _check_scalar_anchor_propagation_for_ray_depth() -> None:
    correction, _, _ = propagate_anchor_values(
        11, [5], np.asarray([0.2]), method="min_bending"
    )
    assert correction.shape == (11, 1)
    assert np.allclose(correction[[0, 5, -1], 0], [0.0, 0.2, 0.0])


def _check_explicit_endpoint_anchors_override_zero_controls(tmp_path: Path) -> None:
    values = np.asarray(
        [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [-0.1, 0.0, 0.3]],
        dtype=np.float64,
    )
    correction, indices, controls = propagate_anchor_displacements(
        11, [0, 5, 10], values, method="min_bending"
    )
    assert np.array_equal(indices, [0, 5, 10])
    assert np.allclose(controls, values)
    assert np.allclose(correction[[0, 5, 10]], values)

    trajectory = load_eef_json(
        _write(tmp_path / "eef_endpoints.json", _eef_payload()), episode_id=0
    )
    events = extract_events(trajectory, "right")
    assert resolve_anchor_frame(trajectory, "right", events, "start") == 0
    assert resolve_anchor_frame(trajectory, "right", events, "end") == trajectory.n - 1
    assert normalize_anchor_id("01") == "1"
    assert sorted(["end", "2", "start", "0"], key=anchor_sort_key) == [
        "start",
        "0",
        "2",
        "end",
    ]


def _check_rotation_only_correction_preserves_positions(tmp_path: Path) -> None:
    source_path = _write(tmp_path / "eef_rotation_source.json", _eef_payload())
    rotation_path = _write(
        tmp_path / "rotation.json",
        {
            "schema": "tate.global_rotation_correction",
            "schema_version": 1,
            "real_calibration_episode_ids": ["0", "1"],
            "real_eval_episode_ids_not_used": ["2"],
            "sides": {
                "right": {
                    "frame_name": "right_flange_zero",
                    "R_bias_matrix": [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    "angle_deg": 90.0,
                    "axis_local": [0.0, 0.0, 1.0],
                }
            },
        },
    )
    output_path = tmp_path / "eef_rotation_only.json"
    apply_correction(
        argparse.Namespace(
            input=str(source_path),
            correction=None,
            rotation_correction=str(rotation_path),
            out=str(output_path),
            sides=None,
            target_mode="mean",
            seed=0,
            target_key="test",
            max_mahalanobis=0.0,
            space="free_xyz",
            propagation="min_bending",
            max_invalid_gap_s=0.25,
            endpoint_weight=1.0,
            anchor_weight=1.0,
            bend_weight=100.0,
            magnitude_weight=1e-3,
            overwrite=False,
        )
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    output = json.loads(output_path.read_text(encoding="utf-8"))
    for before, after in zip(source["frames"], output["frames"]):
        before_pose = np.asarray(before["hand_r"]["tcp_pose_eef_frame"])
        after_pose = np.asarray(after["hand_r"]["tcp_pose_eef_frame"])
        assert np.allclose(before_pose[:3, 3], after_pose[:3, 3])
        assert not np.allclose(before_pose[:3, :3], after_pose[:3, :3])
    metadata = output["metadata"]["real_anchor_correction"]
    assert metadata["artifact"] is None
    assert metadata["rotation_correction"] is not None


def _check_ray_depth_projects_target_onto_source_camera_ray() -> None:
    identity = np.eye(4)
    delta, projected, residual, source_depth = _closest_depth_on_ray(
        np.asarray([0.0, 0.0, 2.0]),
        np.asarray([1.0, 0.0, 4.0]),
        identity,
        identity,
    )
    assert np.isclose(source_depth, 2.0)
    assert np.isclose(delta, 2.0)
    assert np.allclose(projected, [0.0, 0.0, 4.0])
    assert np.isclose(residual, 1.0)


def _check_sample_seed_is_stable_and_identity_specific() -> None:
    assert _stable_seed(7, "episode-1", "right", 2) == _stable_seed(
        7, "episode-1", "right", 2
    )
    assert _stable_seed(7, "episode-1", "right", 2) != _stable_seed(
        7, "episode-2", "right", 2
    )


def _check_fixed_pairs_are_not_duplicated() -> None:
    config = {"pairing": {"fixed_pairs": [[0, 10], {"ego": 1, "real": 11}]}}
    assert _pairs(["0", "1"], ["10", "11"], "fixed_pairs_from_manifest", config) == [
        ("0", "10"),
        ("1", "11"),
    ]


def _check_duplicate_timestamps_are_rejected(tmp_path: Path) -> None:
    payload = _eef_payload()
    payload["frames"][2]["ts"] = payload["frames"][1]["ts"]
    with unittest.TestCase().assertRaisesRegex(SchemaError, "strictly increasing"):
        load_eef_json(_write(tmp_path / "eef.json", payload), episode_id=0)


def _check_ego_calibration_ids_are_not_treated_as_real_leakage() -> None:
    artifact = {
        "ego_calibration_episode_ids": [9],
        "real_calibration_episode_ids": [1],
    }
    assert _verify_no_leakage(artifact, [9], "rotation") == ["1"]


def _check_lifego_floors_and_rhos() -> None:
    centroids = {"0": [0.0, 0.0, 0.0], "1": [1.0, 0.0, 0.0], "2": [3.0, 0.0, 0.0]}
    pair_values = {
        ("0", "1"): (1.0, 10.0, 0.5),
        ("0", "2"): (2.0, 20.0, 1.0),
        ("1", "2"): (3.0, 30.0, 1.5),
    }
    real_records = []
    for (a, b), (d_pos, d_rot, d_shape) in pair_values.items():
        real_records.append(
            {
                "real_episode_a": a,
                "real_episode_b": b,
                "side": "right",
                "distance_components": {
                    "D_pos_m": d_pos,
                    "D_rot_deg": d_rot,
                    "D_shape_m": d_shape,
                    "ego_centroid_m": centroids[a],
                    "real_centroid_m": centroids[b],
                },
            }
        )
    floors = real_to_real_noise_floors(real_records, ["0", "1", "2"], ["right"])
    assert np.isclose(floors["right"]["D_pos_mm"]["mean"], 2000.0)
    assert np.isclose(floors["right"]["D_rot_deg"]["mean"], 20.0)
    assert np.isclose(floors["right"]["D_shape_mm"]["mean"], 1000.0)
    assert np.isclose(floors["right"]["D_offset_mm"]["mean"], 5000.0 / 3.0)

    ego_records = [
        {
            "ego_episode_id": "0",
            "real_episode_id": real_id,
            "side": "right",
            "distance_components": {
                "D_pos_m": 2.0,
                "D_rot_deg": 20.0,
                "D_shape_m": 1.0,
                "ego_centroid_m": [3.0, 0.0, 0.0],
                "real_centroid_m": centroids[real_id],
            },
        }
        for real_id in ("0", "1", "2")
    ]
    metrics = aggregate_records(
        ego_records,
        ["right"],
        floors,
        {"right": {"available": True, "value": 1.25}},
    )
    values = metrics["per_side"]["right"]["per_ego"]["0"]
    assert np.isclose(values["rho_pos"], 1.0)
    assert np.isclose(values["rho_rot"], 1.0)
    assert np.isclose(values["rho_se3"], 1.0)
    assert np.isclose(values["rho_shape"], 1.0)
    assert np.isclose(values["rho_offset"], 1.0)
    assert set(metrics["per_side"]["right"]["aggregate"]) == set(METRIC_FIELDS)


class AlignmentEvaluationTest(unittest.TestCase):
    def _with_temp(self, function) -> None:
        with tempfile.TemporaryDirectory() as directory:
            function(Path(directory))

    def test_loaders_events_and_alignment(self) -> None:
        self._with_temp(_check_loaders_events_and_alignment)

    def test_alignment_rejects_frame_mismatch(self) -> None:
        self._with_temp(_check_alignment_rejects_frame_mismatch)

    def test_split_leakage_is_rejected(self) -> None:
        self._with_temp(_check_split_leakage_is_rejected)

    def test_split_reserve_overlap_is_rejected(self) -> None:
        self._with_temp(_check_split_reserve_overlap_is_rejected)

    def test_anchor_propagation_preserves_endpoints_and_anchor(self) -> None:
        _check_anchor_propagation_preserves_endpoints_and_anchor()

    def test_scalar_anchor_propagation_for_ray_depth(self) -> None:
        _check_scalar_anchor_propagation_for_ray_depth()

    def test_explicit_endpoint_anchors_override_zero_controls(self) -> None:
        self._with_temp(_check_explicit_endpoint_anchors_override_zero_controls)

    def test_rotation_only_correction_preserves_positions(self) -> None:
        self._with_temp(_check_rotation_only_correction_preserves_positions)

    def test_ray_depth_projects_target_onto_source_camera_ray(self) -> None:
        _check_ray_depth_projects_target_onto_source_camera_ray()

    def test_sample_seed_is_stable_and_identity_specific(self) -> None:
        _check_sample_seed_is_stable_and_identity_specific()

    def test_fixed_pairs_are_not_duplicated(self) -> None:
        _check_fixed_pairs_are_not_duplicated()

    def test_duplicate_timestamps_are_rejected(self) -> None:
        self._with_temp(_check_duplicate_timestamps_are_rejected)

    def test_ego_calibration_ids_are_not_treated_as_real_leakage(self) -> None:
        _check_ego_calibration_ids_are_not_treated_as_real_leakage()

    def test_lifego_floors_and_rhos(self) -> None:
        _check_lifego_floors_and_rhos()


if __name__ == "__main__":
    unittest.main()
