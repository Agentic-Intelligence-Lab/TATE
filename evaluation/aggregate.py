"""LifEgo real-to-real floors and ego-to-real normalized metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


METRIC_FIELDS = (
    "rho_se3",
    "D_pos_mm",
    "rho_pos",
    "D_rot_deg",
    "rho_rot",
    "rho_offset",
    "rho_shape",
    "eta_disp",
)
HEADLINE_FIELDS = METRIC_FIELDS


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _floor_summary(
    per_episode: np.ndarray, episode_ids: list[str], *, scale: float = 1.0
) -> dict[str, Any]:
    values = np.asarray(per_episode, dtype=np.float64) * scale
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "per_episode": {
            episode_id: float(value) for episode_id, value in zip(episode_ids, values)
        },
    }


def real_to_real_noise_floors(
    records: list[dict[str, Any]], real_ids: list[str], active_sides: list[str]
) -> dict[str, Any]:
    """Compute LifEgo leave-one-out floors solely from held-out real episodes."""
    if len(real_ids) < 2:
        raise ValueError("real-to-real noise floor requires at least two held-out real episodes")
    index = {episode_id: i for i, episode_id in enumerate(real_ids)}
    output: dict[str, Any] = {}
    for side in active_sides:
        n = len(real_ids)
        matrices = {
            name: np.zeros((n, n), dtype=np.float64)
            for name in ("D_pos_m", "D_rot_deg", "D_shape_m")
        }
        centroids: dict[str, np.ndarray] = {}
        selected = [record for record in records if record["side"] == side]
        for record in selected:
            a, b = str(record["real_episode_a"]), str(record["real_episode_b"])
            i, j = index[a], index[b]
            component = record["distance_components"]
            for name, matrix in matrices.items():
                matrix[i, j] = matrix[j, i] = float(component[name])
            centroids[a] = np.asarray(component["ego_centroid_m"], dtype=np.float64)
            centroids[b] = np.asarray(component["real_centroid_m"], dtype=np.float64)
        expected_pairs = n * (n - 1) // 2
        if len(selected) != expected_pairs or set(centroids) != set(real_ids):
            raise ValueError(
                f"incomplete {side} real-to-real cohort: got {len(selected)}/{expected_pairs} pairs"
            )
        per_position = matrices["D_pos_m"].sum(axis=1) / (n - 1)
        per_rotation = matrices["D_rot_deg"].sum(axis=1) / (n - 1)
        per_shape = matrices["D_shape_m"].sum(axis=1) / (n - 1)
        centroid_array = np.stack([centroids[episode_id] for episode_id in real_ids])
        per_offset = np.asarray(
            [
                np.linalg.norm(
                    centroid_array[i] - np.delete(centroid_array, i, axis=0).mean(axis=0)
                )
                for i in range(n)
            ],
            dtype=np.float64,
        )
        output[side] = {
            "n_real_episodes": n,
            "D_pos_mm": _floor_summary(per_position, real_ids, scale=1000.0),
            "D_rot_deg": _floor_summary(per_rotation, real_ids),
            "D_offset_mm": _floor_summary(per_offset, real_ids, scale=1000.0),
            "D_shape_mm": _floor_summary(per_shape, real_ids, scale=1000.0),
            "real_centroid_m": centroid_array.mean(axis=0).tolist(),
        }
    return output


def aggregate_records(
    records: list[dict[str, Any]],
    active_sides: list[str],
    noise_floors: dict[str, Any],
    eta_dispersion: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate pair distances into the eight metrics reported by LifEgo."""
    per_side: dict[str, Any] = {}
    for side in active_sides:
        selected = [record for record in records if record["side"] == side]
        by_ego: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in selected:
            by_ego[str(record["ego_episode_id"])].append(record["distance_components"])
        floor = noise_floors[side]
        f_pos_m = float(floor["D_pos_mm"]["mean"]) / 1000.0
        f_rot_deg = float(floor["D_rot_deg"]["mean"])
        f_shape_m = float(floor["D_shape_mm"]["mean"]) / 1000.0
        f_offset_m = float(floor["D_offset_mm"]["mean"]) / 1000.0
        if min(f_pos_m, f_rot_deg, f_shape_m, f_offset_m) <= 1e-12:
            raise ValueError(f"{side} real-to-real noise floor is zero; rho is undefined")
        real_centroid = np.asarray(floor["real_centroid_m"], dtype=np.float64)
        per_ego: dict[str, dict[str, Any]] = {}
        for ego_id, components in sorted(by_ego.items()):
            expected_references = int(floor["n_real_episodes"])
            if len(components) != expected_references:
                raise ValueError(
                    f"{side} ego episode {ego_id} has {len(components)}/{expected_references} "
                    "held-out real references; LifEgo rho requires the complete real cohort"
                )
            d_pos_m = float(np.mean([item["D_pos_m"] for item in components]))
            d_rot_deg = float(np.mean([item["D_rot_deg"] for item in components]))
            d_shape_m = float(np.mean([item["D_shape_m"] for item in components]))
            ego_centroid = np.mean(
                [np.asarray(item["ego_centroid_m"], dtype=np.float64) for item in components],
                axis=0,
            )
            d_offset_m = float(np.linalg.norm(ego_centroid - real_centroid))
            rho_pos = d_pos_m / f_pos_m
            rho_rot = d_rot_deg / f_rot_deg
            per_ego[ego_id] = {
                "rho_se3": float(np.sqrt(rho_pos * rho_rot)),
                "D_pos_mm": 1000.0 * d_pos_m,
                "rho_pos": rho_pos,
                "D_rot_deg": d_rot_deg,
                "rho_rot": rho_rot,
                "rho_offset": d_offset_m / f_offset_m,
                "rho_shape": d_shape_m / f_shape_m,
                "n_real_references": len(components),
            }
        aggregate = {
            name: _summary([float(values[name]) for values in per_ego.values()])
            for name in METRIC_FIELDS
            if name != "eta_disp" and per_ego
        }
        eta = eta_dispersion[side]
        if eta.get("available"):
            aggregate["eta_disp"] = _summary([float(eta["value"])])
        else:
            aggregate["eta_disp"] = {
                "n": 0,
                "mean": None,
                "std": None,
                "median": None,
                "min": None,
                "max": None,
            }
        per_side[side] = {
            "n_pairs": len(selected),
            "n_ego_episodes": len(per_ego),
            "per_ego": per_ego,
            "aggregate": aggregate,
            "eta_disp": eta,
        }
    bimanual = {}
    for metric in METRIC_FIELDS:
        side_means = [
            per_side[side]["aggregate"][metric]["mean"]
            for side in active_sides
            if metric in per_side.get(side, {}).get("aggregate", {})
            and per_side[side]["aggregate"][metric]["mean"] is not None
        ]
        if side_means:
            bimanual[metric] = float(np.mean(side_means))
    return {"per_side": per_side, "bimanual_active_side_mean": bimanual}
