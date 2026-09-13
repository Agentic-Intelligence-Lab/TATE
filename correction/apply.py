#!/usr/bin/env python3
"""Apply a pre-fitted real-anchor artifact to one EEF sidecar immutably."""

from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.events import extract_events
from evaluation.loaders import load_eef_json
from evaluation.report import write_json
from evaluation.schemas import read_json, require_schema, sha256_file

from .anchor_ids import anchor_sort_key, normalize_anchor_id, resolve_anchor_frame
from .methods import propagate_anchor_displacements, propagate_anchor_values


class CorrectionNotApplicableError(ValueError):
    """The ego trajectory cannot provide the events required by an artifact."""


def _stable_seed(base_seed: int, *identity: object) -> int:
    payload = "|".join([str(base_seed), *(str(value) for value in identity)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _sample_target(
    anchor: dict[str, Any], mode: str, *, seed: int, max_mahalanobis: float
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = np.asarray(anchor["position_mean_m"], dtype=np.float64)
    if mode == "mean":
        return mean, {"mode": "mean", "seed": None, "mahalanobis": 0.0, "attempts": 0}
    covariance = np.asarray(anchor["position_covariance_m2"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    inverse = np.linalg.pinv(covariance)
    for attempt in range(1, 1001):
        sample = rng.multivariate_normal(mean, covariance)
        delta = sample - mean
        mahalanobis = float(np.sqrt(max(0.0, delta @ inverse @ delta)))
        if max_mahalanobis <= 0.0 or mahalanobis <= max_mahalanobis:
            return sample, {
                "mode": "sample",
                "seed": seed,
                "mahalanobis": mahalanobis,
                "attempts": attempt,
            }
    raise RuntimeError(
        f"failed to sample within Mahalanobis radius {max_mahalanobis} after 1000 attempts"
    )


def _rigid_matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{label} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-4
    ):
        raise ValueError(f"{label} rotation is not rigid")
    return matrix


def _rotation_matrix(value: Any, label: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-4
    ):
        raise ValueError(f"{label} is not a valid rotation")
    return rotation


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _closest_depth_on_ray(
    source_output: np.ndarray,
    target_output: np.ndarray,
    camera_in_output: np.ndarray,
    output_in_camera: np.ndarray,
) -> tuple[float, np.ndarray, float, float]:
    source_camera = _transform_points(output_in_camera, source_output[None])[0]
    source_depth = float(source_camera[2])
    if abs(source_depth) < 1e-8:
        raise ValueError("EEF anchor has near-zero camera depth")
    ray = source_camera / source_depth
    direction_output = camera_in_output[:3, :3] @ ray
    origin_output = camera_in_output[:3, 3]
    target_depth = float(
        direction_output @ (target_output - origin_output)
        / max(float(direction_output @ direction_output), 1e-12)
    )
    projected = origin_output + direction_output * target_depth
    residual = float(np.linalg.norm(projected - target_output))
    return target_depth - source_depth, projected, residual, source_depth


def _ray_depth_correction(
    source_data: dict[str, Any],
    side: str,
    trajectory: Any,
    frames: list[int],
    targets: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    transforms = (source_data.get("arm_transforms") or {}).get("T_cam_in_output_frame") or {}
    camera_in_output = _rigid_matrix(
        transforms.get(side), f"arm_transforms.T_cam_in_output_frame.{side}"
    )
    output_in_camera = np.linalg.inv(camera_in_output)
    requested = []
    details = []
    for frame, target in zip(frames, targets):
        delta, projected, residual, source_depth = _closest_depth_on_ray(
            trajectory.sides[side].position[frame],
            target,
            camera_in_output,
            output_in_camera,
        )
        requested.append(delta)
        details.append(
            {
                "source_depth_m": source_depth,
                "requested_depth_delta_m": delta,
                "ray_projected_target_position_m": projected,
                "target_to_ray_residual_m": residual,
            }
        )
    depth_values, control_indices, control_values = propagate_anchor_values(
        trajectory.n,
        frames,
        np.asarray(requested, dtype=np.float64),
        method=args.propagation,
        endpoint_weight=args.endpoint_weight,
        anchor_weight=args.anchor_weight,
        bend_weight=args.bend_weight,
        magnitude_weight=args.magnitude_weight,
    )
    depth_delta = depth_values[:, 0]
    correction = np.zeros((trajectory.n, 3), dtype=np.float64)
    arm = trajectory.sides[side]
    for frame in np.flatnonzero(arm.valid):
        source = arm.position[frame]
        source_camera = _transform_points(output_in_camera, source[None])[0]
        if abs(float(source_camera[2])) < 1e-8:
            raise ValueError(f"{side} EEF frame {frame} has near-zero camera depth")
        ray = source_camera / source_camera[2]
        corrected_camera = ray * (source_camera[2] + depth_delta[frame])
        corrected = _transform_points(camera_in_output, corrected_camera[None])[0]
        correction[frame] = corrected - source
    return correction, control_indices, control_values, details


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.input).resolve()
    artifact_path_value = getattr(args, "correction", None)
    artifact_path = Path(artifact_path_value).resolve() if artifact_path_value else None
    rotation_artifact_path_value = getattr(args, "rotation_correction", None)
    rotation_artifact_path = (
        Path(rotation_artifact_path_value).resolve()
        if rotation_artifact_path_value
        else None
    )
    if artifact_path is None and rotation_artifact_path is None:
        raise ValueError("at least one position or rotation correction artifact is required")
    output_path = Path(args.out).resolve()
    if output_path == source_path:
        raise ValueError("correction output must differ from the immutable source EEF path")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output_path}")
    source_data = read_json(source_path)
    trajectory = load_eef_json(source_path)
    artifact: dict[str, Any] = {}
    if artifact_path is not None:
        artifact = read_json(artifact_path)
        require_schema(artifact, "tate.real_anchor_correction", 1, artifact_path)
    rotation_artifact = None
    if rotation_artifact_path is not None:
        rotation_artifact = read_json(rotation_artifact_path)
        require_schema(
            rotation_artifact,
            "tate.global_rotation_correction",
            1,
            rotation_artifact_path,
        )
    output = deepcopy(source_data)
    target_key = str(getattr(args, "target_key", None) or source_path)
    max_mahalanobis = float(getattr(args, "max_mahalanobis", 0.0))
    if max_mahalanobis < 0.0:
        raise ValueError("max_mahalanobis must be non-negative")
    space = str(getattr(args, "space", "free_xyz"))
    if space not in ("free_xyz", "ray_depth"):
        raise ValueError(f"unsupported correction space {space!r}")
    applied = {}
    for side, side_artifact in (artifact.get("sides") or {}).items():
        if args.sides and side not in args.sides:
            continue
        arm = trajectory.sides[side]
        if arm.frame_name != side_artifact.get("frame_name"):
            raise ValueError(
                f"correction frame mismatch for {side}: EEF={arm.frame_name!r}, "
                f"artifact={side_artifact.get('frame_name')!r}"
            )
        events = extract_events(trajectory, side, max_invalid_gap_s=args.max_invalid_gap_s)
        anchors = side_artifact.get("anchors") or {}
        anchor_ids = [normalize_anchor_id(value) for value in anchors]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError(f"correction artifact has duplicate anchors for {side}")
        anchor_ids.sort(key=anchor_sort_key)
        if not anchor_ids:
            raise ValueError(f"correction artifact has no anchors for {side}")
        frames = []
        for anchor_id in anchor_ids:
            try:
                frame = resolve_anchor_frame(trajectory, side, events, anchor_id)
            except ValueError as exc:
                raise CorrectionNotApplicableError(str(exc)) from exc
            frames.append(frame)
            if anchor_id in ("start", "end"):
                continue
            ordinal = int(anchor_id)
            event = events[ordinal]
            expected_transition = anchors[anchor_id].get("transition")
            if event.ambiguous:
                raise CorrectionNotApplicableError(
                    f"{side} anchor event {ordinal} crosses an excessive invalid gap"
                )
            if expected_transition and event.transition != expected_transition:
                raise CorrectionNotApplicableError(
                    f"{side} anchor {ordinal} transition is {event.transition!r}, "
                    f"artifact expects {expected_transition!r}"
                )
        sampled = []
        sampling = []
        for anchor_id in anchor_ids:
            seed = _stable_seed(args.seed, target_key, side, anchor_id)
            target, details = _sample_target(
                anchors[anchor_id],
                args.target_mode,
                seed=seed,
                max_mahalanobis=max_mahalanobis,
            )
            sampled.append(target)
            sampling.append(details)
        targets = np.asarray(sampled)
        sources = arm.position[frames]
        ray_details: list[dict[str, Any]] | None = None
        if space == "free_xyz":
            correction, control_indices, control_values = propagate_anchor_displacements(
                trajectory.n,
                frames,
                targets - sources,
                method=args.propagation,
                endpoint_weight=args.endpoint_weight,
                anchor_weight=args.anchor_weight,
                bend_weight=args.bend_weight,
                magnitude_weight=args.magnitude_weight,
            )
        else:
            correction, control_indices, control_values, ray_details = _ray_depth_correction(
                source_data, side, trajectory, frames, targets, args
            )
        key = "hand_l" if side == "left" else "hand_r"
        for frame_index, frame in enumerate(output["frames"]):
            hand = frame.get(key)
            if hand is None or not arm.valid[frame_index]:
                continue
            original = deepcopy(hand["tcp_pose_eef_frame"])
            matrix = np.asarray(original, dtype=np.float64)
            matrix[:3, 3] += correction[frame_index]
            hand["tcp_pose_eef_frame_before_correction"] = original
            hand["tcp_pose_eef_frame"] = matrix.tolist()
            hand["eef_pose_world"] = matrix.tolist()
            hand["eef_translation_world"] = matrix[:3, 3].tolist()
            hand["real_anchor_correction_m"] = correction[frame_index].tolist()
        applied[side] = {
            "space": space,
            "anchor_ids": anchor_ids,
            "anchor_ordinals": [int(value) for value in anchor_ids if value.isdigit()],
            "anchor_frames": frames,
            "source_positions_m": sources,
            "target_positions_m": targets,
            "sampling": sampling,
            "ray_depth_details": ray_details,
            "control_indices": control_indices,
            "control_values_m": control_values,
            "max_displacement_m": float(np.linalg.norm(correction, axis=1).max()),
        }
    rotation_applied = {}
    if rotation_artifact is not None:
        for side, side_artifact in (rotation_artifact.get("sides") or {}).items():
            if args.sides and side not in args.sides:
                continue
            arm = trajectory.sides[side]
            if arm.frame_name != side_artifact.get("frame_name"):
                raise ValueError(
                    f"rotation correction frame mismatch for {side}: "
                    f"EEF={arm.frame_name!r}, artifact={side_artifact.get('frame_name')!r}"
                )
            bias = _rotation_matrix(
                side_artifact.get("R_bias_matrix"),
                f"rotation correction {side}.R_bias_matrix",
            )
            key = "hand_l" if side == "left" else "hand_r"
            for frame_index, frame in enumerate(output["frames"]):
                hand = frame.get(key)
                if hand is None or not arm.valid[frame_index]:
                    continue
                before_rotation = deepcopy(hand["tcp_pose_eef_frame"])
                matrix = np.asarray(before_rotation, dtype=np.float64)
                matrix[:3, :3] = matrix[:3, :3] @ bias.T
                hand["tcp_pose_eef_frame_before_rotation_correction"] = before_rotation
                hand["tcp_pose_eef_frame"] = matrix.tolist()
                hand["eef_pose_world"] = matrix.tolist()
            rotation_applied[side] = {
                "frame_name": arm.frame_name,
                "application": "R_corrected = R_source @ R_bias.inv()",
                "bias_angle_deg": side_artifact.get("angle_deg"),
                "bias_axis_local": side_artifact.get("axis_local"),
            }
    rotation_metadata = None
    if rotation_artifact is not None and rotation_artifact_path is not None:
        rotation_metadata = {
            "artifact": str(rotation_artifact_path),
            "artifact_fingerprint": sha256_file(rotation_artifact_path),
            "real_calibration_episode_ids": rotation_artifact.get(
                "real_calibration_episode_ids"
            )
            or [],
            "real_eval_episode_ids_not_used": rotation_artifact.get(
                "real_eval_episode_ids_not_used"
            )
            or [],
            "ego_calibration_episode_ids": rotation_artifact.get(
                "ego_calibration_episode_ids"
            )
            or [],
            "applied": rotation_applied,
        }
    metadata = {
        "schema": "tate.applied_real_anchor_correction",
        "schema_version": 1,
        "artifact": str(artifact_path) if artifact_path is not None else None,
        "artifact_fingerprint": sha256_file(artifact_path) if artifact_path is not None else None,
        "source_eef": str(source_path),
        "source_eef_fingerprint": sha256_file(source_path),
        "real_calibration_episode_ids": (
            artifact.get("real_calibration_episode_ids")
            or (rotation_artifact or {}).get("real_calibration_episode_ids")
            or []
        ),
        "real_eval_episode_ids_not_used": (
            artifact.get("real_eval_episode_ids_not_used")
            or (rotation_artifact or {}).get("real_eval_episode_ids_not_used")
            or []
        ),
        "target_mode": args.target_mode,
        "seed": args.seed,
        "target_key": target_key,
        "max_mahalanobis": max_mahalanobis,
        "space": space,
        "propagation": args.propagation,
        "applied": applied,
        "rotation_correction": rotation_metadata,
    }
    output.setdefault("metadata", {})["real_anchor_correction"] = metadata
    write_json(output_path, output)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw tate.dual_arm_eef v2 JSON")
    parser.add_argument("--correction", help="Optional fitted position-anchor artifact")
    parser.add_argument(
        "--rotation-correction",
        help="Optional fitted global-rotation artifact; may be used without --correction",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--sides", nargs="*", choices=("left", "right"))
    parser.add_argument("--target-mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-key", help="Stable episode identity used for sampled targets")
    parser.add_argument(
        "--max-mahalanobis",
        type=float,
        default=0.0,
        help="Optional truncation radius for sampled targets; 0 disables truncation",
    )
    parser.add_argument("--space", choices=("free_xyz", "ray_depth"), default="free_xyz")
    parser.add_argument("--propagation", choices=("linear", "min_bending", "smooth_spline"), default="linear")
    parser.add_argument("--max-invalid-gap-s", type=float, default=0.25)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--bend-weight", type=float, default=100.0)
    parser.add_argument("--magnitude-weight", type=float, default=1e-3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
