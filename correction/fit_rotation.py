#!/usr/bin/env python3
"""Fit a task-level local SO(3) bias using only real calibration episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from evaluation.align import align_pair
from evaluation.loaders import load_real_fk_json, load_real_manifest
from evaluation.report import write_json
from evaluation.run_eval import (
    DEFAULT_CONFIG,
    _health_reason,
    _load_struct,
    _load_variant_trajectory,
    _merge,
    _resolve_cohort,
    _validate_split,
    _verify_no_leakage,
)
from evaluation.schemas import read_json, require_schema, resolve_path, sha256_file


def run(args: argparse.Namespace) -> dict[str, Any]:
    experiment_path = Path(args.experiment_manifest).resolve()
    real_manifest_path = Path(args.real_manifest).resolve()
    split_path = Path(args.real_split).resolve()
    config_path = Path(args.eval_config).resolve()
    output_path = Path(args.out).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"rotation artifact exists; pass --overwrite: {output_path}")

    experiment = read_json(experiment_path)
    require_schema(experiment, "tate.preprocess_experiment_manifest", 1, experiment_path)
    variant_id = str(args.variant)
    if variant_id not in (experiment.get("variants") or {}):
        raise KeyError(f"unknown experiment variant {variant_id!r}")
    config = _merge(DEFAULT_CONFIG, _load_struct(config_path))
    active_sides = list(args.sides or config.get("active_sides") or [])
    if not active_sides or any(side not in ("left", "right") for side in active_sides):
        raise ValueError("active sides must contain left and/or right")

    ego_ids, _, _ = _resolve_cohort(experiment, [variant_id], args.episodes)
    if len(ego_ids) < 2:
        raise ValueError("global rotation fitting needs at least two complete ego episodes")
    variant = experiment["variants"][variant_id]
    ego = {
        episode_id: _load_variant_trajectory(
            experiment_path,
            str(experiment.get("experiment_id")),
            variant_id,
            variant,
            episode_id,
        )
        for episode_id in ego_ids
    }

    real_manifest, real_outputs = load_real_manifest(real_manifest_path)
    requested_real_path = (experiment.get("source_real_dataset") or {}).get("path")
    manifest_real_path = real_manifest.get("dataset_root")
    if requested_real_path and manifest_real_path and resolve_path(
        requested_real_path, experiment_path.parent
    ) != resolve_path(manifest_real_path, real_manifest_path.parent):
        raise ValueError(
            "real FK manifest dataset_root does not match experiment source_real_dataset"
        )
    split = _load_struct(split_path)
    calibration_ids, eval_ids = _validate_split(
        split,
        real_outputs,
        config,
        experiment,
        real_manifest_fingerprint=real_manifest["_fingerprint"],
    )
    if not calibration_ids:
        raise ValueError("real split calibration cohort is empty")
    for ego_id, trajectory in ego.items():
        _verify_no_leakage(
            trajectory.source.get("correction"),
            eval_ids,
            f"rotation fit source {variant_id}/{ego_id}",
        )
    real = {
        episode_id: load_real_fk_json(real_outputs[episode_id], episode_id=episode_id)
        for episode_id in calibration_ids
    }

    exclusions: list[dict[str, Any]] = []
    side_results = {}
    for side in active_sides:
        per_ego = []
        used_ego_ids = []
        pair_count = 0
        for ego_id, ego_trajectory in ego.items():
            offsets = []
            for real_id, real_trajectory in real.items():
                ego_reason, ego_health = _health_reason(ego_trajectory, side, config)
                real_reason, real_health = _health_reason(real_trajectory, side, config)
                if ego_reason or real_reason:
                    exclusions.append(
                        {
                            "ego_episode_id": ego_id,
                            "real_episode_id": real_id,
                            "side": side,
                            "reason": ego_reason or real_reason,
                            "ego_event_health": ego_health,
                            "real_event_health": real_health,
                        }
                    )
                    continue
                alignment = align_pair(
                    ego_trajectory, real_trajectory, side, dict(config["alignment"])
                )
                ego_quat = alignment.ego.pose_xyzw[alignment.ego_indices, 3:]
                real_quat = alignment.real.pose_xyzw[alignment.real_indices, 3:]
                offsets.append(
                    (Rotation.from_quat(real_quat).inv() * Rotation.from_quat(ego_quat))
                    .as_rotvec()
                )
                pair_count += 1
            if offsets:
                per_ego.append(Rotation.from_rotvec(np.concatenate(offsets)).mean())
                used_ego_ids.append(ego_id)
        if len(per_ego) < 2:
            raise ValueError(
                f"{side} global rotation fitting needs at least two healthy ego episodes"
            )
        fitted = Rotation.from_rotvec(
            np.asarray([rotation.as_rotvec() for rotation in per_ego])
        ).mean()
        rotvec = fitted.as_rotvec()
        angle = float(np.degrees(np.linalg.norm(rotvec)))
        spread = float(
            np.mean(
                [np.degrees((fitted.inv() * rotation).magnitude()) for rotation in per_ego]
            )
        )
        frame_names = {ego[episode_id].sides[side].frame_name for episode_id in used_ego_ids}
        if len(frame_names) != 1:
            raise ValueError(f"{side} ego episodes use inconsistent frames: {sorted(frame_names)}")
        side_results[side] = {
            "frame_name": next(iter(frame_names)),
            "application": "R_corrected = R_source @ R_bias.inv()",
            "R_bias_matrix": fitted.as_matrix(),
            "R_bias_quat_xyzw": fitted.as_quat(),
            "R_bias_rotvec": rotvec,
            "angle_deg": angle,
            "axis_local": rotvec / max(float(np.linalg.norm(rotvec)), 1e-12),
            "per_ego_spread_deg": spread,
            "ego_episode_ids": used_ego_ids,
            "ego_count": len(used_ego_ids),
            "ego_real_pair_count": pair_count,
        }

    artifact = {
        "schema": "tate.global_rotation_correction",
        "schema_version": 1,
        "task": split.get("task") or config.get("task"),
        "experiment_id": experiment.get("experiment_id"),
        "variant_id": variant_id,
        "fit_correspondence": config["alignment"],
        "source_manifests": {
            "experiment": {
                "path": str(experiment_path),
                "fingerprint": sha256_file(experiment_path),
            },
            "real": {
                "path": str(real_manifest_path),
                "fingerprint": real_manifest["_fingerprint"],
            },
            "split": {"path": str(split_path), "fingerprint": sha256_file(split_path)},
            "eval_config": {
                "path": str(config_path),
                "fingerprint": sha256_file(config_path),
            },
        },
        "ego_calibration_episode_ids": ego_ids,
        "real_calibration_episode_ids": calibration_ids,
        "real_eval_episode_ids_not_used": eval_ids,
        "sides": side_results,
        "exclusions": exclusions,
    }
    write_json(output_path, artifact)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-manifest", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--real-manifest", required=True)
    parser.add_argument("--real-split", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--sides", nargs="*", choices=("left", "right"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
