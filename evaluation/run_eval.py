#!/usr/bin/env python3
"""Run manifest-driven TATE alignment and evaluation without fitting corrections."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

from .aggregate import HEADLINE_FIELDS, aggregate_records, real_to_real_noise_floors
from .align import AlignmentError, align_pair
from .events import event_health, extract_events
from .loaders import load_eef_json, load_lerobot_episode, load_real_fk_json, load_real_manifest
from .report import save_alignment, write_episode_csv, write_json
from .schemas import SchemaError, read_json, require_schema, resolve_path, sha256_file


DEFAULT_CONFIG: dict[str, Any] = {
    "active_sides": None,
    "activity": {
        "left": {"expected_events": None, "require_expected_events": False},
        "right": {"expected_events": None, "require_expected_events": False},
    },
    "alignment": {
        "method": "position_dtw",
        "sample_count": 200,
        "sample_rate_hz": None,
        "max_invalid_gap_s": 0.25,
        "dtw_window_ratio": None,
        "segment": {"start_event": "first", "end_event": "last"},
    },
    "metrics": {"anchors": None},
    "pairing": {"policy": "all_pairs", "fixed_pairs": []},
}


class EvaluationLeakageError(ValueError):
    """A fitted correction used a held-out real evaluation episode."""


def _load_struct(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return read_json(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _merge(output[key], value)
        else:
            output[key] = value
    return output


def _canonical_ids(values: list[Any]) -> list[str]:
    return [str(value) for value in values]


def _parse_segment(value: str) -> dict[str, int | str | None]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("--segment must be START_EVENT:END_EVENT; either endpoint may be empty")

    def endpoint(raw: str) -> int | str | None:
        if raw == "":
            return None
        if raw in ("first", "last"):
            return raw
        return int(raw)

    return {"start_event": endpoint(parts[0]), "end_event": endpoint(parts[1])}


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=repo_root).returncode != 0
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(dirty or untracked), "untracked_count": len(untracked)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "untracked_count": None}


def _resolve_variants(manifest: dict[str, Any], requested: list[str] | None) -> list[str]:
    variants = manifest.get("variants") or {}
    selected = list(variants) if not requested else requested
    missing = [variant for variant in selected if variant not in variants]
    if missing:
        raise KeyError(f"unknown variants: {missing}")
    if not selected:
        raise ValueError("experiment manifest has no selected variants")
    return selected


def _resolve_cohort(
    manifest: dict[str, Any], variants: list[str], requested: list[str] | None
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    coverage = {}
    for variant in variants:
        episodes = (manifest["variants"][variant].get("episodes") or {})
        coverage[variant] = sorted(
            str(episode_id)
            for episode_id, record in episodes.items()
            if record.get("status") == "complete"
        )
    requested_ids = sorted(set(requested or [item for values in coverage.values() for item in values]))
    common = [episode_id for episode_id in requested_ids if all(episode_id in coverage[v] for v in variants)]
    if not common:
        raise ValueError("selected variants have no common complete ego episode cohort")
    return common, coverage, requested_ids


def _validate_split(
    split: dict[str, Any],
    real_outputs: dict[str, Path],
    config: dict[str, Any],
    experiment: dict[str, Any],
    real_manifest_fingerprint: str | None = None,
) -> tuple[list[str], list[str]]:
    if split.get("schema") is not None and (
        split.get("schema") != "tate.real_evaluation_split" or split.get("schema_version") != 1
    ):
        raise ValueError("unsupported real evaluation split schema/version")
    calibration = _canonical_ids(split.get("calibration") or [])
    evaluation = _canonical_ids(split.get("eval") or [])
    reserve = _canonical_ids(split.get("reserve") or [])
    for name, values in (
        ("calibration", calibration),
        ("eval", evaluation),
        ("reserve", reserve),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"real split {name} cohort contains duplicate episode IDs")
    overlaps = {
        "calibration/eval": sorted(set(calibration) & set(evaluation)),
        "calibration/reserve": sorted(set(calibration) & set(reserve)),
        "eval/reserve": sorted(set(evaluation) & set(reserve)),
    }
    overlaps = {name: values for name, values in overlaps.items() if values}
    if overlaps:
        raise ValueError(f"real split leakage/overlap: {overlaps}")
    missing = sorted(
        (set(calibration) | set(evaluation) | set(reserve)) - set(real_outputs)
    )
    if missing:
        raise ValueError(f"real split references missing episodes: {missing}")
    if not evaluation:
        raise ValueError("real split eval cohort is empty")
    expected_task = config.get("task")
    if expected_task and split.get("task") != expected_task:
        raise ValueError(f"real split task {split.get('task')!r} != eval task {expected_task!r}")
    expected_fingerprint = ((experiment.get("source_real_dataset") or {}).get("fingerprint"))
    split_fingerprint = split.get("dataset_fingerprint") or split.get("real_dataset_fingerprint")
    if expected_fingerprint and split_fingerprint and split_fingerprint != expected_fingerprint:
        raise ValueError("real split dataset fingerprint does not match the experiment manifest")
    requested_manifest_fingerprint = split.get("real_manifest_fingerprint")
    if (
        requested_manifest_fingerprint
        and real_manifest_fingerprint
        and requested_manifest_fingerprint != real_manifest_fingerprint
    ):
        raise ValueError("real split manifest fingerprint does not match the real FK manifest")
    return calibration, evaluation


def _correction_calibration_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if "ego" not in key_lower and "calibration" in key_lower and (
                "episode" in key_lower or key_lower == "calibration"
            ):
                if isinstance(item, (list, tuple)):
                    found.update(str(entry) for entry in item)
            found.update(_correction_calibration_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_correction_calibration_ids(item))
    return found


def _verify_no_leakage(correction: Any, eval_ids: list[str], label: str) -> list[str]:
    calibration_ids = sorted(_correction_calibration_ids(correction))
    overlap = sorted(set(calibration_ids) & set(eval_ids))
    if overlap:
        raise EvaluationLeakageError(
            f"correction/evaluation leakage in {label}: real episodes {overlap}"
        )
    return calibration_ids


def _load_variant_trajectory(
    manifest_path: Path,
    experiment_id: str,
    variant_id: str,
    variant: dict[str, Any],
    episode_id: str,
) -> Any:
    record = (variant.get("episodes") or {})[episode_id]
    final_eef = record.get("final_eef")
    if final_eef:
        path = resolve_path(final_eef, manifest_path.parent)
        expected = record.get("final_eef_fingerprint")
        if expected and sha256_file(path) != expected:
            raise ValueError(f"final EEF fingerprint mismatch for episode {episode_id}: {path}")
        dataset = variant.get("derived_lerobot_dataset")
        if dataset and resolve_path(dataset, manifest_path.parent).is_dir():
            _verify_packaged_dataset(
                resolve_path(dataset, manifest_path.parent),
                experiment_id=experiment_id,
                variant_id=variant_id,
                episode_id=episode_id,
                final_eef_path=path,
                package_signature=(variant.get("package") or {}).get("signature"),
            )
        return load_eef_json(path, episode_id=episode_id)
    dataset = variant.get("derived_lerobot_dataset")
    if dataset:
        return load_lerobot_episode(resolve_path(dataset, manifest_path.parent), episode_id)
    raise ValueError(f"episode {episode_id} has neither final_eef nor derived_lerobot_dataset")


def _verify_packaged_dataset(
    dataset_root: Path,
    *,
    experiment_id: str,
    variant_id: str,
    episode_id: str,
    final_eef_path: Path,
    package_signature: str | None,
) -> None:
    provenance_path = dataset_root / "meta" / "tate_preprocess.json"
    provenance = read_json(provenance_path)
    require_schema(provenance, "tate.lerobot_preprocess_provenance", 1, provenance_path)
    if provenance.get("experiment_id") != experiment_id or provenance.get("variant_id") != variant_id:
        raise ValueError(f"derived LeRobot provenance identity mismatch: {dataset_root}")
    if package_signature and provenance.get("package_signature") != package_signature:
        raise ValueError(f"derived LeRobot package fingerprint mismatch: {dataset_root}")
    episode = next(
        (
            value
            for value in provenance.get("episodes") or []
            if str(value.get("source_episode_index")) == episode_id
        ),
        None,
    )
    if episode is None:
        raise ValueError(f"derived LeRobot dataset does not contain source episode {episode_id}")
    packaged_source = resolve_path(episode["final_eef"], provenance_path.parent)
    if packaged_source != final_eef_path:
        raise ValueError(
            f"derived LeRobot episode {episode_id} points at a different canonical final EEF"
        )
    packaged_fingerprint = episode.get("final_eef_fingerprint")
    if packaged_fingerprint and packaged_fingerprint != sha256_file(final_eef_path):
        raise ValueError(
            f"derived LeRobot episode {episode_id} final EEF fingerprint mismatch"
        )


def _pairs(
    ego_ids: list[str], real_ids: list[str], policy: str, config: dict[str, Any]
) -> list[tuple[str, str]]:
    if policy == "all_pairs":
        return [(ego_id, real_id) for ego_id in ego_ids for real_id in real_ids]
    if policy == "fixed_pairs_from_manifest":
        raw = ((config.get("pairing") or {}).get("fixed_pairs") or [])
        pairs = []
        for pair in raw:
            if isinstance(pair, dict):
                item = (str(pair["ego"]), str(pair["real"]))
            else:
                item = (str(pair[0]), str(pair[1]))
            if item[0] in ego_ids and item[1] in real_ids:
                pairs.append(item)
        if not pairs:
            raise ValueError("fixed pairing policy resolved no pairs")
        return pairs
    raise ValueError(f"unsupported pairing policy {policy!r}")


def _health_reason(trajectory: Any, side: str, config: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    arm = trajectory.sides[side]
    if not arm.valid.any():
        return "no valid poses for active side", {}
    side_config = (config.get("activity") or {}).get(side) or {}
    events = extract_events(
        trajectory, side, max_invalid_gap_s=float(config["alignment"]["max_invalid_gap_s"])
    )
    health = event_health(
        events,
        side_config.get("expected_events"),
        allow_ambiguous=bool(side_config.get("allow_ambiguous_events", False)),
    )
    if side_config.get("require_expected_events", side_config.get("expected_events") is not None) and not health["valid"]:
        return "unhealthy gripper event sequence", health
    return None, health


def _distance_components(
    trajectory_a: Any,
    trajectory_b: Any,
    side: str,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from .metrics import evaluate_pair_side

    alignment_config = dict(config["alignment"])
    alignment = align_pair(trajectory_a, trajectory_b, side, alignment_config)
    return alignment, evaluate_pair_side(
        trajectory_a, trajectory_b, side, alignment, alignment_config
    )


def _compute_real_noise_floors(
    real_cache: dict[str, Any],
    real_ids: list[str],
    active_sides: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for real_id in real_ids:
        for side in active_sides:
            reason, health = _health_reason(real_cache[real_id], side, config)
            if reason:
                raise ValueError(
                    f"held-out real episode {real_id}/{side} cannot define the noise floor: "
                    f"{reason}; event health={health}"
                )
    for real_a, real_b in combinations(real_ids, 2):
        for side in active_sides:
            _, components = _distance_components(
                real_cache[real_a], real_cache[real_b], side, config
            )
            records.append(
                {
                    "real_episode_a": real_a,
                    "real_episode_b": real_b,
                    "side": side,
                    "distance_components": components,
                }
            )
    return real_to_real_noise_floors(records, real_ids, active_sides), records


def _compute_eta_dispersion(
    ego_cache: dict[str, Any],
    active_sides: list[str],
    config: dict[str, Any],
    noise_floors: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    ego_ids = sorted(ego_cache)
    output: dict[str, dict[str, Any]] = {}
    for side in active_sides:
        healthy_ids = []
        excluded_ids = []
        for ego_id in ego_ids:
            health_reason, _ = _health_reason(ego_cache[ego_id], side, config)
            (excluded_ids if health_reason else healthy_ids).append(ego_id)
        if len(healthy_ids) < 2:
            output[side] = {
                "available": False,
                "value": None,
                "reason": f"eta_disp requires at least two healthy ego episodes; got {len(healthy_ids)}",
                "ego_episode_ids": healthy_ids,
                "excluded_ego_episode_ids": excluded_ids,
            }
            continue
        values = []
        for ego_a, ego_b in combinations(healthy_ids, 2):
            _, components = _distance_components(
                ego_cache[ego_a], ego_cache[ego_b], side, config
            )
            values.append(float(components["D_pos_m"]))
        ego_within_m = float(sum(values) / len(values))
        real_within_m = float(noise_floors[side]["D_pos_mm"]["mean"]) / 1000.0
        output[side] = {
            "available": True,
            "value": ego_within_m / real_within_m,
            "ego_within_D_pos_mm": 1000.0 * ego_within_m,
            "real_within_D_pos_mm": 1000.0 * real_within_m,
            "n_ego_episodes": len(healthy_ids),
            "n_ego_pairs": len(values),
            "ego_episode_ids": healthy_ids,
            "excluded_ego_episode_ids": excluded_ids,
        }
    return output


def _short_variant_id(summary: dict[str, Any]) -> str:
    hand = str(summary.get("hand2gripper_id") or summary["variant_id"])
    hand = hand.removesuffix("_hys085")
    correction = str(summary.get("correction_id") or "none")
    if correction == "none":
        return hand
    if correction == "xyz_mean_target_min_bending":
        return f"{hand}+xyz"
    if correction == "xyz_mean_target_min_bending_all_anchors":
        return f"{hand}+xyz_all"
    return f"{hand}+{correction}"


def _write_comparison(
    out_dir: Path, summaries: list[dict[str, Any]], resolved: dict[str, Any]
) -> None:
    rows = []
    for summary in summaries:
        for side, block in summary["metrics"]["per_side"].items():
            row = {"variant_id": _short_variant_id(summary), "side": side}
            for name in HEADLINE_FIELDS:
                aggregate = block.get("aggregate", {}).get(name)
                row[name] = None if aggregate is None else aggregate["mean"]
            rows.append(row)
    evaluated: dict[str, dict[str, dict[str, Any]]] = {}
    for summary in summaries:
        by_side = {}
        for side in resolved["active_sides"]:
            records = [
                record
                for record in summary.get("pair_results", [])
                if record["side"] == side
            ]
            by_side[side] = {
                "ego_episode_ids": sorted(
                    {str(record["ego_episode_id"]) for record in records}
                ),
                "real_episode_ids": sorted(
                    {str(record["real_episode_id"]) for record in records}
                ),
                "ego_real_pairs": sorted(
                    [
                        [str(record["ego_episode_id"]), str(record["real_episode_id"])]
                        for record in records
                    ]
                ),
            }
        evaluated[summary["variant_id"]] = by_side
    baseline = evaluated[summaries[0]["variant_id"]] if summaries else {}
    differences = [
        variant_id for variant_id, actual in evaluated.items() if actual != baseline
    ]
    comparable = not differences
    for row in rows:
        row["comparable"] = comparable
    write_json(
        out_dir / "comparison.json",
        {
            "schema": "tate.evaluation_comparison",
            "schema_version": 2,
            "experiment_id": resolved["experiment_id"],
            "variants": resolved["variants"],
            "variant_id_map": {
                _short_variant_id(summary): summary["variant_id"] for summary in summaries
            },
            "ego_episode_ids": resolved["common_ego_episode_ids"],
            "real_eval_episode_ids": resolved["real_eval_episode_ids"],
            "active_sides": resolved["active_sides"],
            "pairing": resolved["pairing"],
            "source_manifests": resolved["source_manifests"],
            "resolved_config": resolved["config"],
            "comparable": comparable,
            "comparability": {
                "criterion": "identical evaluated ego-real pairs per active side",
                "evaluated_cohorts": evaluated,
                "variants_different_from_first": differences,
            },
            "rows": rows,
        },
    )
    with (out_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["variant_id", "side", "comparable", *HEADLINE_FIELDS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    experiment_path = Path(args.experiment_manifest).resolve()
    real_manifest_path = Path(args.real_manifest).resolve()
    split_path = Path(args.real_split).resolve()
    config_path = Path(args.eval_config).resolve()
    experiment = read_json(experiment_path)
    require_schema(experiment, "tate.preprocess_experiment_manifest", 1, experiment_path)
    config = _merge(DEFAULT_CONFIG, _load_struct(config_path))
    if args.sides:
        config["active_sides"] = args.sides
    if args.segment is not None:
        config["alignment"]["segment"] = _parse_segment(args.segment)
    if args.pairing:
        config["pairing"]["policy"] = args.pairing
    active_sides = list(
        config.get("active_sides")
        or experiment.get("active_sides")
        or ["right"]
    )
    if not active_sides or any(side not in ("left", "right") for side in active_sides):
        raise ValueError("active_sides must contain left and/or right")

    variants = _resolve_variants(experiment, args.variants)
    cohort, coverage, requested_cohort = _resolve_cohort(experiment, variants, args.episodes)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        cohort = cohort[: args.limit]
    real_manifest, real_outputs = load_real_manifest(real_manifest_path)
    requested_real_path = ((experiment.get("source_real_dataset") or {}).get("path"))
    manifest_real_path = real_manifest.get("dataset_root")
    if requested_real_path and manifest_real_path:
        if resolve_path(requested_real_path, experiment_path.parent) != resolve_path(
            manifest_real_path, real_manifest_path.parent
        ):
            raise ValueError("real FK manifest dataset_root does not match experiment source_real_dataset")
    split = _load_struct(split_path)
    calibration_ids, real_eval_ids = _validate_split(
        split,
        real_outputs,
        config,
        experiment,
        real_manifest_fingerprint=real_manifest["_fingerprint"],
    )
    pair_policy = str(config["pairing"].get("policy", "all_pairs"))
    resolved_pairs = _pairs(cohort, real_eval_ids, pair_policy, config)
    out_dir = Path(args.out).resolve()
    resolved = {
        "experiment_id": experiment.get("experiment_id"),
        "variants": variants,
        "requested_ego_episode_ids": requested_cohort,
        "common_ego_episode_ids": cohort,
        "variant_coverage": coverage,
        "real_calibration_episode_ids": calibration_ids,
        "real_eval_episode_ids": real_eval_ids,
        "active_sides": active_sides,
        "pairing": pair_policy,
        "pair_count_per_variant": len(resolved_pairs),
        "output": str(out_dir),
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
        "config": config,
    }
    if args.dry_run:
        print(yaml.safe_dump(resolved, sort_keys=False))
        return resolved
    if out_dir.exists() and any(out_dir.iterdir()) and not (args.force or args.resume):
        raise FileExistsError(f"evaluation output is not empty; pass --resume or --force: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    real_cache = {
        real_id: load_real_fk_json(real_outputs[real_id], episode_id=real_id)
        for real_id in real_eval_ids
    }
    noise_floors, real_pair_results = _compute_real_noise_floors(
        real_cache, real_eval_ids, active_sides, config
    )
    write_json(
        out_dir / "real_to_real_noise_floor.json",
        {
            "schema": "tate.real_to_real_noise_floor",
            "schema_version": 1,
            "real_eval_episode_ids": real_eval_ids,
            "active_sides": active_sides,
            "alignment": config["alignment"],
            "noise_floor": noise_floors,
            "pair_results": real_pair_results,
        },
    )
    summaries = []
    for variant_id in variants:
        variant_dir = out_dir / variant_id
        summary_path = variant_dir / "summary.json"
        if args.resume and summary_path.is_file():
            existing = read_json(summary_path)
            expected_sources = {
                "experiment": sha256_file(experiment_path),
                "real": real_manifest["_fingerprint"],
                "split": sha256_file(split_path),
                "eval_config": sha256_file(config_path),
            }
            actual_sources = {
                name: ((existing.get("source_manifests") or {}).get(name) or {}).get("fingerprint")
                for name in expected_sources
            }
            if (
                existing.get("selected_ego_episode_ids") != cohort
                or existing.get("selected_real_eval_episode_ids") != real_eval_ids
                or existing.get("active_sides") != active_sides
                or existing.get("resolved_config") != config
                or actual_sources != expected_sources
            ):
                raise ValueError(f"cannot resume incompatible evaluation result: {summary_path}")
            summaries.append(existing)
            continue
        variant = experiment["variants"][variant_id]
        artifact_calibration_ids: list[str] = []
        correction_artifact = variant.get("correction_artifact")
        if correction_artifact:
            artifact_path = resolve_path(correction_artifact, experiment_path.parent)
            artifact_calibration_ids = _verify_no_leakage(read_json(artifact_path), real_eval_ids, str(artifact_path))
        ego_cache = {}
        exclusions: list[dict[str, Any]] = []
        variant_episodes = variant.get("episodes") or {}
        for episode_id in requested_cohort:
            if episode_id in cohort:
                continue
            record = variant_episodes.get(episode_id)
            if record is None:
                reason = "episode missing from variant manifest"
            elif record.get("status") != "complete":
                reason = f"variant episode status is {record.get('status')!r}"
            else:
                reason = "excluded to preserve the common cohort across variants"
            exclusions.append(
                {"ego_episode_id": episode_id, "stage": "manifest_coverage", "reason": reason}
            )
        for episode_id in cohort:
            try:
                trajectory = _load_variant_trajectory(
                    experiment_path,
                    str(experiment.get("experiment_id")),
                    variant_id,
                    variant,
                    episode_id,
                )
                used = _verify_no_leakage(trajectory.source.get("correction"), real_eval_ids, f"{variant_id}/{episode_id}")
                artifact_calibration_ids = sorted(set(artifact_calibration_ids) | set(used))
                ego_cache[episode_id] = trajectory
            except EvaluationLeakageError:
                raise
            except Exception as exc:
                exclusions.append({"ego_episode_id": episode_id, "stage": "load", "reason": str(exc)})
        records: list[dict[str, Any]] = []
        for ego_id, real_id in resolved_pairs:
            if ego_id not in ego_cache:
                continue
            ego, real = ego_cache[ego_id], real_cache[real_id]
            for side in active_sides:
                ego_reason, ego_health = _health_reason(ego, side, config)
                real_reason, real_health = _health_reason(real, side, config)
                if ego_reason or real_reason:
                    exclusions.append(
                        {
                            "ego_episode_id": ego_id,
                            "real_episode_id": real_id,
                            "side": side,
                            "stage": "event_health",
                            "reason": ego_reason or real_reason,
                            "ego_event_health": ego_health,
                            "real_event_health": real_health,
                        }
                    )
                    continue
                try:
                    alignment, distance_components = _distance_components(
                        ego, real, side, config
                    )
                    alignment_name = f"{ego_id}__{real_id}__{side}.npz"
                    save_alignment(variant_dir / "alignments" / alignment_name, alignment)
                    records.append(
                        {
                            "variant_id": variant_id,
                            "ego_episode_id": ego_id,
                            "real_episode_id": real_id,
                            "side": side,
                            "alignment_artifact": str((variant_dir / "alignments" / alignment_name).resolve()),
                            "distance_components": distance_components,
                        }
                    )
                except (AlignmentError, SchemaError, ValueError) as exc:
                    exclusions.append(
                        {
                            "ego_episode_id": ego_id,
                            "real_episode_id": real_id,
                            "side": side,
                            "stage": "alignment_or_metrics",
                            "reason": str(exc),
                        }
                    )
        if not records:
            raise RuntimeError(f"variant {variant_id} produced no evaluable ego-real pairs")
        eta_dispersion = _compute_eta_dispersion(
            ego_cache, active_sides, config, noise_floors
        )
        metrics = aggregate_records(
            records, active_sides, noise_floors, eta_dispersion
        )
        summary = {
            "schema": "tate.alignment_evaluation_summary",
            "schema_version": 2,
            "experiment_id": experiment.get("experiment_id"),
            "variant_id": variant_id,
            "hand2gripper_id": variant.get("hand2gripper_id"),
            "correction_id": variant.get("correction_id"),
            "config_fingerprint": variant.get("config_fingerprint"),
            "source_manifests": {
                "experiment": {"path": str(experiment_path), "fingerprint": sha256_file(experiment_path)},
                "real": {"path": str(real_manifest_path), "fingerprint": real_manifest["_fingerprint"]},
                "split": {"path": str(split_path), "fingerprint": sha256_file(split_path)},
                "eval_config": {
                    "path": str(config_path),
                    "fingerprint": sha256_file(config_path),
                },
            },
            "selected_ego_episode_ids": cohort,
            "loaded_ego_episode_ids": sorted(ego_cache),
            "selected_real_eval_episode_ids": real_eval_ids,
            "correction_real_calibration_episode_ids": artifact_calibration_ids,
            "active_sides": active_sides,
            "resolved_config": config,
            "code": _git_provenance(repo_root),
            "counts": {
                "ego_real_pairs": len(
                    {
                        (record["ego_episode_id"], record["real_episode_id"])
                        for record in records
                    }
                ),
                "pair_side_results": len(records),
                "exclusions": len(exclusions),
            },
            "metrics": metrics,
            "real_to_real_noise_floor": noise_floors,
            "pair_results": records,
        }
        write_json(summary_path, summary)
        write_json(variant_dir / "exclusions.json", exclusions)
        write_episode_csv(variant_dir / "episodes.csv", variant_id, metrics)
        summaries.append(summary)
    _write_comparison(out_dir, summaries, resolved)
    return {"resolved": resolved, "summaries": summaries}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-manifest", required=True)
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--real-manifest", required=True)
    parser.add_argument("--real-split", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--sides", nargs="*", choices=("left", "right"))
    parser.add_argument(
        "--segment",
        default=None,
        help="START_EVENT:END_EVENT; supports first/last, blank means episode boundary",
    )
    parser.add_argument("--pairing", choices=("all_pairs", "fixed_pairs_from_manifest"))
    parser.add_argument(
        "--limit", type=int, help="debug limit on ego episodes; all held-out real references remain"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
