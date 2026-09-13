#!/usr/bin/env python3
"""Fit held-out-safe real ARX anchor distributions from calibration episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from evaluation.events import event_health, extract_events
from evaluation.loaders import load_real_fk_json, load_real_manifest
from evaluation.report import write_json
from evaluation.schemas import read_json, sha256_file

from .anchor_ids import anchor_sort_key, normalize_anchor_id, resolve_anchor_frame
from .methods import fit_anchor_distribution


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    real_manifest_path = Path(args.real_manifest).resolve()
    split_path = Path(args.real_split).resolve()
    config_path = Path(args.eval_config).resolve()
    manifest, outputs = load_real_manifest(real_manifest_path)
    split = read_json(split_path)
    if split.get("schema") is not None and (
        split.get("schema") != "tate.real_evaluation_split" or split.get("schema_version") != 1
    ):
        raise ValueError("unsupported real evaluation split schema/version")
    config = _load_config(config_path)
    calibration_ids = [str(value) for value in split.get("calibration") or []]
    eval_ids = [str(value) for value in split.get("eval") or []]
    reserve_ids = [str(value) for value in split.get("reserve") or []]
    if not calibration_ids or not eval_ids:
        raise ValueError("real split must contain non-empty calibration and eval cohorts")
    for name, values in (
        ("calibration", calibration_ids),
        ("eval", eval_ids),
        ("reserve", reserve_ids),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"real split {name} cohort contains duplicate episode IDs")
    overlaps = {
        "calibration/eval": sorted(set(calibration_ids) & set(eval_ids)),
        "calibration/reserve": sorted(set(calibration_ids) & set(reserve_ids)),
        "eval/reserve": sorted(set(eval_ids) & set(reserve_ids)),
    }
    overlaps = {name: values for name, values in overlaps.items() if values}
    if overlaps:
        raise ValueError(f"real split leakage/overlap: {overlaps}")
    missing = sorted(
        (set(calibration_ids) | set(eval_ids) | set(reserve_ids)) - set(outputs)
    )
    if missing:
        raise ValueError(f"real split references missing real episodes: {missing}")
    expected_task = config.get("task")
    if expected_task and split.get("task") != expected_task:
        raise ValueError(f"real split task {split.get('task')!r} != config task {expected_task!r}")
    split_fingerprint = split.get("dataset_fingerprint") or split.get(
        "real_dataset_fingerprint"
    )
    manifest_fingerprint = manifest.get("dataset_fingerprint") or manifest.get(
        "source_dataset_fingerprint"
    )
    if split_fingerprint and manifest_fingerprint and split_fingerprint != manifest_fingerprint:
        raise ValueError("real split dataset fingerprint does not match the real manifest")
    requested_manifest_fingerprint = split.get("real_manifest_fingerprint")
    if requested_manifest_fingerprint and requested_manifest_fingerprint != manifest["_fingerprint"]:
        raise ValueError("real split manifest fingerprint does not match the real FK manifest")
    active_sides = args.sides or list(config.get("active_sides") or ["right"])
    if not active_sides or any(side not in ("left", "right") for side in active_sides):
        raise ValueError("active sides must contain left and/or right")
    trajectories = {
        episode_id: load_real_fk_json(outputs[episode_id], episode_id=episode_id)
        for episode_id in calibration_ids
    }
    side_results = {}
    exclusions = []
    max_gap = float((config.get("alignment") or {}).get("max_invalid_gap_s", 0.25))
    configured_anchors = (config.get("correction") or {}).get("anchors")
    if configured_anchors is None:
        configured_anchors = (config.get("metrics") or {}).get("anchors")
    for side in active_sides:
        healthy = []
        expected = ((config.get("activity") or {}).get(side) or {}).get("expected_events")
        for episode_id, trajectory in trajectories.items():
            events = extract_events(trajectory, side, max_invalid_gap_s=max_gap)
            health = event_health(events, expected)
            if not trajectory.sides[side].valid.any() or not health["valid"]:
                exclusions.append({"episode_id": episode_id, "side": side, "event_health": health})
                continue
            healthy.append((episode_id, trajectory, events))
        if len(healthy) < 2:
            raise ValueError(f"{side} needs at least two healthy calibration episodes")
        frame_names = {trajectory.sides[side].frame_name for _, trajectory, _ in healthy}
        if len(frame_names) != 1:
            raise ValueError(
                f"{side} calibration episodes use inconsistent coordinate frames: "
                f"{sorted(frame_names)}"
            )
        requested = (
            list(configured_anchors)
            if configured_anchors is not None
            else list(range(min(len(item[2]) for item in healthy)))
        )
        anchors = [normalize_anchor_id(value) for value in requested]
        if len(anchors) != len(set(anchors)):
            raise ValueError(f"{side} correction anchors contain duplicates: {anchors}")
        anchors.sort(key=anchor_sort_key)
        fitted = {}
        for anchor_id in anchors:
            if anchor_id in ("start", "end"):
                usable = healthy
                transition = None
            else:
                ordinal = int(anchor_id)
                usable = [
                    item
                    for item in healthy
                    if ordinal < len(item[2]) and not item[2][ordinal].ambiguous
                ]
                if len(usable) < 2:
                    raise ValueError(
                        f"{side} anchor {anchor_id} has fewer than two calibration samples"
                    )
                transitions = {events[ordinal].transition for _, _, events in usable}
                if len(transitions) != 1:
                    raise ValueError(
                        f"{side} anchor {anchor_id} mixes transitions: {sorted(transitions)}"
                    )
                transition = next(iter(transitions))
            if len(usable) < 2:
                raise ValueError(
                    f"{side} anchor {anchor_id} has fewer than two calibration samples"
                )
            frames = [
                resolve_anchor_frame(trajectory, side, events, anchor_id)
                for _, trajectory, events in usable
            ]
            positions = np.asarray(
                [
                    trajectory.sides[side].position[frame]
                    for (_, trajectory, _), frame in zip(usable, frames)
                ]
            )
            quaternions = np.asarray(
                [
                    trajectory.sides[side].quaternion_xyzw[frame]
                    for (_, trajectory, _), frame in zip(usable, frames)
                ]
            )
            mean, covariance = fit_anchor_distribution(positions, args.covariance_shrinkage)
            mean_rotation = Rotation.from_quat(quaternions).mean().as_quat()
            fitted[anchor_id] = {
                "kind": "trajectory_endpoint" if anchor_id in ("start", "end") else "gripper_event",
                "transition": transition,
                "n": len(usable),
                "episode_ids": [item[0] for item in usable],
                "source_frame_indices": frames,
                "position_mean_m": mean,
                "position_covariance_m2": covariance,
                "orientation_mean_quat_xyzw": mean_rotation,
            }
        side_results[side] = {
            "frame_name": next(iter(frame_names)),
            "anchors": fitted,
        }
    artifact = {
        "schema": "tate.real_anchor_correction",
        "schema_version": 1,
        "method": "independent_per_anchor_3d_gaussian",
        "task": split.get("task") or config.get("task"),
        "real_manifest": {"path": str(real_manifest_path), "fingerprint": manifest["_fingerprint"]},
        "real_split": {"path": str(split_path), "fingerprint": sha256_file(split_path)},
        "real_calibration_episode_ids": calibration_ids,
        "real_eval_episode_ids_not_used": eval_ids,
        "real_reserve_episode_ids_not_used": reserve_ids,
        "covariance_shrinkage": args.covariance_shrinkage,
        "sides": side_results,
        "exclusions": exclusions,
    }
    output_path = Path(args.out).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"correction artifact exists; pass --overwrite: {output_path}")
    write_json(output_path, artifact)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-manifest", required=True)
    parser.add_argument("--real-split", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--sides", nargs="*", choices=("left", "right"))
    parser.add_argument("--covariance-shrinkage", type=float, default=0.1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
