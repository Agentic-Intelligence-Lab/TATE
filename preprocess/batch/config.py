"""Experiment configuration loading, validation, and stable fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from preprocess.Hand2Gripper import MODES


REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require_id(value: Any, field: str) -> str:
    result = str(value or "")
    if not SAFE_ID.fullmatch(result):
        raise ValueError(f"{field} must match {SAFE_ID.pattern!r}, got {result!r}")
    return result


def _path(value: Any, field: str, *, required: bool = True) -> str | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"{field} is required")
        return None
    return str(as_repo_path(str(value)))


def _normalize_hand2gripper_variants(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("hand2gripper_variants must be a non-empty mapping")
    output: dict[str, dict[str, Any]] = {}
    for variant_id, value in raw.items():
        variant_id = _require_id(variant_id, "hand2gripper variant id")
        cfg = dict(value or {})
        mode = str(cfg.get("mode", variant_id))
        if mode not in MODES:
            raise ValueError(
                f"hand2gripper variant {variant_id!r} has unknown mode {mode!r}; "
                f"choose from {sorted(MODES)}"
            )
        cfg["mode"] = mode
        grasp_mode = str(cfg.get("grasp_mode", mode))
        if grasp_mode not in MODES:
            raise ValueError(
                f"hand2gripper variant {variant_id!r} has unknown grasp_mode "
                f"{grasp_mode!r}; choose from {sorted(MODES)}"
            )
        cfg["grasp_mode"] = grasp_mode
        cfg["eef_config"] = _path(
            cfg.get("eef_config", "cfg/preprocess/base/EEFExport.yaml"),
            f"hand2gripper_variants.{variant_id}.eef_config",
        )
        for key in ("grasp_close_ratio", "grasp_open_ratio"):
            if cfg.get(key) is not None:
                cfg[key] = float(cfg[key])
        if cfg.get("grasp_min_frames") is not None:
            cfg["grasp_min_frames"] = int(cfg["grasp_min_frames"])
        output[variant_id] = cfg
    return output


def _normalize_correction_variants(raw: Any) -> dict[str, dict[str, Any]]:
    raw = raw or {"none": {"mode": "none"}}
    if not isinstance(raw, dict) or not raw:
        raise ValueError("correction_variants must be a non-empty mapping")
    output: dict[str, dict[str, Any]] = {}
    for variant_id, value in raw.items():
        variant_id = _require_id(variant_id, "correction variant id")
        cfg = dict(value or {})
        cfg["mode"] = str(cfg.get("mode", "none"))
        if cfg["mode"] not in {"none", "real_anchor", "pose_correction"} and not cfg.get("entrypoint"):
            raise ValueError(
                f"correction variant {variant_id!r} uses mode {cfg['mode']!r} but has no "
                "entrypoint ('python.module:function')"
            )
        if cfg.get("artifact") is not None:
            cfg["artifact"] = _path(
                cfg["artifact"], f"correction_variants.{variant_id}.artifact"
            )
        if cfg.get("rotation_artifact") is not None:
            cfg["rotation_artifact"] = _path(
                cfg["rotation_artifact"],
                f"correction_variants.{variant_id}.rotation_artifact",
            )
        output[variant_id] = cfg
    return output


def _normalize_runs(
    raw: Any,
    hand2gripper: dict[str, dict[str, Any]],
    corrections: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    if raw is None:
        if "none" not in corrections:
            raise ValueError("runs is required when no correction variant named 'none' exists")
        raw = [
            {"id": f"{name}__none", "hand2gripper": name, "correction": "none"}
            for name in hand2gripper
        ]
    if not isinstance(raw, list) or not raw:
        raise ValueError("runs must be a non-empty list")
    output = []
    seen = set()
    for index, value in enumerate(raw):
        cfg = dict(value or {})
        run_id = _require_id(cfg.get("id"), f"runs[{index}].id")
        if run_id in seen:
            raise ValueError(f"duplicate run id: {run_id}")
        seen.add(run_id)
        h2g = str(cfg.get("hand2gripper", ""))
        correction = str(cfg.get("correction", "none"))
        if h2g not in hand2gripper:
            raise ValueError(f"run {run_id!r} references unknown hand2gripper {h2g!r}")
        if correction not in corrections:
            raise ValueError(f"run {run_id!r} references unknown correction {correction!r}")
        output.append({"id": run_id, "hand2gripper": h2g, "correction": correction})
    return output


def _normalize_trajectory(raw: Any) -> dict[str, Any]:
    cfg = dict(raw or {})
    arm_mode = str(cfg.get("arm_mode", "single_arm"))
    if arm_mode not in {"single_arm", "bimanual"}:
        raise ValueError("trajectory.arm_mode must be 'single_arm' or 'bimanual'")

    default_sides = ["right"] if arm_mode == "single_arm" else ["left", "right"]
    active_sides = list(cfg.get("active_sides") or default_sides)
    if (
        not active_sides
        or any(side not in {"left", "right"} for side in active_sides)
        or len(active_sides) != len(set(active_sides))
    ):
        raise ValueError(
            "trajectory.active_sides must contain unique values chosen from left/right"
        )
    if arm_mode == "single_arm" and len(active_sides) != 1:
        raise ValueError("single_arm requires exactly one trajectory.active_sides entry")
    if arm_mode == "bimanual" and set(active_sides) != {"left", "right"}:
        raise ValueError("bimanual requires trajectory.active_sides: [left, right]")

    # Keep side ordering canonical in every manifest and fingerprint.
    active_sides = [side for side in ("left", "right") if side in active_sides]
    trim = dict(cfg.get("trim") or {})
    start_seconds = float(trim.get("start_seconds", 0.0))
    end_seconds = float(trim.get("end_seconds", 0.0))
    if (
        not math.isfinite(start_seconds)
        or not math.isfinite(end_seconds)
        or start_seconds < 0.0
        or end_seconds < 0.0
    ):
        raise ValueError("trajectory.trim start/end seconds must be finite and non-negative")

    return {
        "arm_mode": arm_mode,
        "active_sides": active_sides,
        "trim": {
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
        },
    }


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load and normalize one batch experiment YAML.

    Repository-relative paths are resolved against the TATE repository root so
    commands behave consistently regardless of the caller's working directory.
    """

    config_path = as_repo_path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"experiment config must contain a mapping: {config_path}")

    experiment_id = _require_id(raw.get("experiment_id"), "experiment_id")
    source = dict(raw.get("source") or {})
    source["ego_dataset"] = _path(source.get("ego_dataset"), "source.ego_dataset")
    source["video_key"] = str(source.get("video_key", "observation.images.head"))
    source["real_dataset"] = _path(
        source.get("real_dataset"), "source.real_dataset", required=False
    )
    source["real_manifest"] = _path(
        source.get("real_manifest"), "source.real_manifest", required=False
    )

    outputs = dict(raw.get("outputs") or {})
    outputs["cache_root"] = _path(
        outputs.get("cache_root", "outputs/cache/wilor"), "outputs.cache_root"
    )
    outputs["experiment_root"] = _path(
        outputs.get("experiment_root", "outputs/experiments"),
        "outputs.experiment_root",
    )
    outputs["video_cache_root"] = _path(
        outputs.get("video_cache_root", "outputs/cache/video_segments"),
        "outputs.video_cache_root",
    )

    wilor = dict(raw.get("wilor") or {})
    wilor["id"] = _require_id(wilor.get("id", "default"), "wilor.id")
    wilor["preprocess_config"] = _path(
        wilor.get("preprocess_config", "cfg/preprocess/base/Preprocess.yaml"),
        "wilor.preprocess_config",
    )
    wilor["pretrained_dir"] = _path(
        wilor.get("pretrained_dir"), "wilor.pretrained_dir", required=False
    )

    geometry = dict(raw.get("geometry") or {})
    geometry["camera_calibration"] = _path(
        geometry.get("camera_calibration", "cfg/preprocess/base/RealSenseD405.yaml"),
        "geometry.camera_calibration",
    )

    trajectory = _normalize_trajectory(raw.get("trajectory"))

    hand2gripper = _normalize_hand2gripper_variants(raw.get("hand2gripper_variants"))
    corrections = _normalize_correction_variants(raw.get("correction_variants"))
    runs = _normalize_runs(raw.get("runs"), hand2gripper, corrections)

    retarget = dict(raw.get("retarget") or {})
    retarget["enabled"] = bool(retarget.get("enabled", True))
    retarget["scene"] = _path(
        retarget.get("scene", "assets/mujoco_arx_scene/scene.xml"),
        "retarget.scene",
    )
    retarget.setdefault("options", {})

    lerobot = dict(raw.get("lerobot") or {})
    lerobot["enabled"] = bool(lerobot.get("enabled", True))
    lerobot["video_mode"] = str(lerobot.get("video_mode", "hardlink"))
    if lerobot["video_mode"] not in {"hardlink", "copy"}:
        raise ValueError("lerobot.video_mode must be 'hardlink' or 'copy'")
    lerobot["replace_state_action"] = bool(lerobot.get("replace_state_action", True))
    lerobot["require_all_episodes"] = bool(lerobot.get("require_all_episodes", True))
    lerobot["action_alignment"] = str(lerobot.get("action_alignment", "same_frame"))
    if lerobot["action_alignment"] not in {"same_frame", "next_frame"}:
        raise ValueError("lerobot.action_alignment must be 'same_frame' or 'next_frame'")
    lerobot["gripper_open_raw"] = float(lerobot.get("gripper_open_raw", -3.4))
    lerobot["gripper_closed_raw"] = float(lerobot.get("gripper_closed_raw", 0.1))
    lerobot["task"] = (
        None if lerobot.get("task") in (None, "") else str(lerobot["task"])
    )
    if (
        lerobot["enabled"]
        and lerobot["replace_state_action"]
        and not retarget["enabled"]
    ):
        raise ValueError(
            "lerobot.replace_state_action=true requires retarget.enabled=true"
        )

    resolved = {
        "schema": "tate.batch_preprocess_config",
        "schema_version": 1,
        "experiment_id": experiment_id,
        "source": source,
        "outputs": outputs,
        "wilor": wilor,
        "geometry": geometry,
        "trajectory": trajectory,
        "hand2gripper_variants": hand2gripper,
        "correction_variants": corrections,
        "runs": runs,
        "retarget": retarget,
        "lerobot": lerobot,
        "config_path": str(config_path),
    }
    resolved["config_fingerprint"] = fingerprint(
        {key: value for key, value in resolved.items() if key != "config_path"}
    )
    return resolved


def dependency_fingerprint(paths: list[str | Path], extra: Any = None) -> str:
    records = []
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": str(path), "sha256": file_sha256(path)})
    return fingerprint({"files": records, "extra": extra})
