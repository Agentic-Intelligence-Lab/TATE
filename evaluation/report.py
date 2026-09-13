"""Evaluation artifact writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .align import AlignmentResult
from .aggregate import HEADLINE_FIELDS
from .schemas import jsonable


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2), encoding="utf-8")


def save_alignment(path: Path, alignment: AlignmentResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ego_indices=alignment.ego_indices,
        real_indices=alignment.real_indices,
        ego_timestamps_s=alignment.ego.timestamps_s,
        real_timestamps_s=alignment.real.timestamps_s,
        ego_pose_xyzw=alignment.ego.pose_xyzw,
        real_pose_xyzw=alignment.real.pose_xyzw,
        ego_valid=alignment.ego.valid,
        real_valid=alignment.real.valid,
        normalized_cost_m=np.asarray(alignment.normalized_cost),
        method=np.asarray(alignment.method),
        cost_definition=np.asarray(alignment.cost_definition),
    )


def write_episode_csv(
    path: Path, variant_id: str, metrics: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant_id", "ego_episode_id", "side", *HEADLINE_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for side, block in metrics["per_side"].items():
            eta = block.get("eta_disp") or {}
            eta_value = eta.get("value") if eta.get("available") else None
            for ego_episode_id, values in block["per_ego"].items():
                row = {
                    "variant_id": variant_id,
                    "ego_episode_id": ego_episode_id,
                    "side": side,
                    **{name: values.get(name) for name in HEADLINE_FIELDS},
                }
                row["eta_disp"] = eta_value
                writer.writerow(row)
