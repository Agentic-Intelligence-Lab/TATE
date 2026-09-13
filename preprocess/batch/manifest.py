"""Atomic experiment manifest updates for resumable batch runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


class ExperimentManifest:
    def __init__(
        self,
        path: str | Path,
        *,
        config: dict[str, Any],
        source_dataset_fingerprint: str,
        source_episode_count: int,
        git: dict[str, Any],
    ) -> None:
        self.path = Path(path)
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            stored = self.data.get("config_fingerprint")
            if stored != config["config_fingerprint"]:
                raise RuntimeError(
                    f"experiment {config['experiment_id']!r} already exists with a different "
                    f"configuration ({stored} != {config['config_fingerprint']}). Use a new "
                    "experiment_id so existing variants are not mixed or overwritten."
                )
            stored_dataset = (
                self.data.get("source_ego_dataset") or {}
            ).get("fingerprint")
            if stored_dataset != source_dataset_fingerprint:
                raise RuntimeError(
                    "source ego dataset changed since this experiment manifest was created; "
                    "use a new experiment_id to preserve provenance"
                )
        else:
            self.data = {
                "schema": "tate.preprocess_experiment_manifest",
                "schema_version": 1,
                "experiment_id": config["experiment_id"],
                "config_path": config["config_path"],
                "config_fingerprint": config["config_fingerprint"],
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "git": git,
                "source_ego_dataset": {
                    "path": config["source"]["ego_dataset"],
                    "fingerprint": source_dataset_fingerprint,
                    "episode_count": source_episode_count,
                    "video_key": config["source"]["video_key"],
                },
                "arm_mode": config["trajectory"]["arm_mode"],
                "active_sides": config["trajectory"]["active_sides"],
                "temporal_trim": config["trajectory"]["trim"],
                "wilor": {},
                "hand2gripper": {},
                "variants": {},
            }
            if config["source"].get("real_dataset"):
                self.data["source_real_dataset"] = {
                    "path": config["source"]["real_dataset"],
                    "manifest": config["source"].get("real_manifest"),
                }
            self.save()

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        atomic_write_json(self.path, self.data)

    def episode_stage(
        self, group: str, group_id: str, episode_index: int
    ) -> dict[str, Any] | None:
        return (
            (self.data.get(group) or {})
            .get(group_id, {})
            .get("episodes", {})
            .get(str(episode_index))
        )

    def update_episode(
        self,
        group: str,
        group_id: str,
        episode_index: int,
        value: dict[str, Any],
        *,
        group_meta: dict[str, Any] | None = None,
    ) -> None:
        record = self.data.setdefault(group, {}).setdefault(group_id, {})
        if group_meta:
            for key, item in group_meta.items():
                if key != "episodes":
                    record[key] = item
        record.setdefault("episodes", {})[str(episode_index)] = value
        self.save()

    def update_variant(self, variant_id: str, values: dict[str, Any]) -> None:
        self.data.setdefault("variants", {}).setdefault(variant_id, {}).update(values)
        self.save()
