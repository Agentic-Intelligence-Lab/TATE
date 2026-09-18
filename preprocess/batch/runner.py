"""Stage runner for reusable WiLoR, EEF variants, correction, and retargeting."""

from __future__ import annotations

import json
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from preprocess.PipelineIO import export_eef_from_cache
from preprocess.batch.config import (
    REPO_ROOT,
    dependency_fingerprint,
    file_sha256,
    fingerprint,
)
from preprocess.batch.corrections import apply_correction
from preprocess.batch.dataset import EpisodeSource, materialize_video_segment
from preprocess.batch.manifest import ExperimentManifest, atomic_write_json, utc_now


STAGE_ORDER = ("wilor", "eef", "correct", "retarget", "package")


def git_state() -> dict[str, Any]:
    def command(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(command("status", "--porcelain")),
    }


def _checkpoint_identity(path_value: str | None) -> Any:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    records = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        stat = child.stat()
        records.append(
            {
                "path": str(child.relative_to(path)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {"path": str(path), "files": records}


def _preprocess_dependencies(config_path: str) -> list[str]:
    """Include directly referenced YAML files in the WiLoR cache identity."""

    path = Path(config_path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    paths = [str(path)]
    for key, value in cfg.items():
        if not key.endswith("_path") or not isinstance(value, str):
            continue
        candidate = Path(value).expanduser()
        candidate = candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()
        if candidate.is_file():
            paths.append(str(candidate))
    return sorted(set(paths))


def _correction_dependencies(corrections: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for correction_id, cfg in corrections.items():
        record: dict[str, Any] = {"id": correction_id, "mode": cfg["mode"]}
        if cfg["mode"] == "real_anchor":
            builtin_paths = [
                REPO_ROOT / "correction/apply.py",
                REPO_ROOT / "correction/methods/anchors.py",
            ]
            record["implementation_sha256"] = {
                str(path.relative_to(REPO_ROOT)): file_sha256(path)
                if path.is_file()
                else "missing"
                for path in builtin_paths
            }
        for artifact_key in ("artifact", "rotation_artifact"):
            artifact = cfg.get(artifact_key)
            if artifact:
                path = Path(artifact)
                record[artifact_key] = {
                    "path": str(path),
                    "sha256": file_sha256(path) if path.is_file() else "missing",
                }
        entrypoint = cfg.get("entrypoint")
        if entrypoint:
            module_name = str(entrypoint).split(":", 1)[0]
            spec = importlib.util.find_spec(module_name)
            source = None if spec is None else spec.origin
            record["entrypoint"] = {
                "name": entrypoint,
                "source": source,
                "sha256": file_sha256(source)
                if source and Path(source).is_file()
                else "unresolved",
            }
        output.append(record)
    return output


def _is_complete(record: dict[str, Any] | None, signature: str, paths: Iterable[Path]) -> bool:
    return bool(
        record
        and record.get("status") == "complete"
        and record.get("signature") == signature
        and all(path.is_file() for path in paths)
    )


def _override_incompatible_experiment(
    experiment_dir: Path,
    *,
    config_fingerprint: str,
    source_dataset_fingerprint: str,
    enabled: bool,
) -> None:
    """Remove only an incompatible experiment directory when explicitly requested.

    WiLoR caches deliberately live outside ``experiment_dir`` and are therefore
    retained. A missing or malformed manifest is never removed implicitly.
    """

    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot inspect existing experiment manifest: {manifest_path}") from error
    config_changed = existing.get("config_fingerprint") != config_fingerprint
    dataset_changed = (
        (existing.get("source_ego_dataset") or {}).get("fingerprint")
        != source_dataset_fingerprint
    )
    if not (config_changed or dataset_changed) or not enabled:
        return
    if not experiment_dir.is_dir() or experiment_dir.is_symlink():
        raise RuntimeError(f"refusing to override non-directory experiment path: {experiment_dir}")
    reasons = []
    if config_changed:
        reasons.append("configuration fingerprint changed")
    if dataset_changed:
        reasons.append("source dataset fingerprint changed")
    print(f"Override experiment {experiment_dir} ({'; '.join(reasons)}); preserving shared WiLoR cache.")
    shutil.rmtree(experiment_dir)


def _eef_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tate.dual_arm_eef" or payload.get("schema_version") != 2:
        raise ValueError(
            f"unexpected EEF schema in {path}: {payload.get('schema')} "
            f"v{payload.get('schema_version')}"
        )
    frames = payload.get("frames") or []
    if int(payload.get("total_frames", -1)) != len(frames):
        raise ValueError(f"EEF total_frames does not match frames array: {path}")
    output: dict[str, Any] = {"total_frames": len(frames), "valid_frames": {}, "grasp_events": {}}
    for side, key in (("left", "hand_l"), ("right", "hand_r")):
        records = [frame.get(key) for frame in frames]
        output["valid_frames"][side] = sum(record is not None for record in records)
        states = [None if record is None else int(record["grasp_state"]) for record in records]
        previous = None
        count = 0
        for state in states:
            if state is None:
                continue
            if previous is not None and state != previous:
                count += 1
            previous = state
        output["grasp_events"][side] = count
    return output


def _cli_options(options: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in options.items():
        flag = f"--{str(key).replace('_', '-')}"
        if value is None or value is False:
            continue
        if value is True:
            result.append(flag)
        elif isinstance(value, (list, tuple)):
            result.append(flag)
            result.extend(str(item) for item in value)
        else:
            result.extend([flag, str(value)])
    return result


class BatchRunner:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        episodes: list[EpisodeSource],
        source_dataset_fingerprint: str,
        source_episode_count: int,
        selected_run_ids: set[str] | None = None,
        force_stages: set[str] | None = None,
        fail_fast: bool = False,
        override_experiment: bool = False,
    ) -> None:
        self.config = config
        self.episodes = episodes
        self.source_dataset_fingerprint = source_dataset_fingerprint
        self.force_stages = force_stages or set()
        self.fail_fast = fail_fast
        self.runs = [
            run
            for run in config["runs"]
            if selected_run_ids is None or run["id"] in selected_run_ids
        ]
        self.wilor_fingerprint = dependency_fingerprint(
            _preprocess_dependencies(config["wilor"]["preprocess_config"]),
            extra={
                "wilor_id": config["wilor"]["id"],
                "pretrained": _checkpoint_identity(config["wilor"].get("pretrained_dir")),
                "dataset": source_dataset_fingerprint,
                "video_key": config["source"]["video_key"],
                "temporal_trim": config["trajectory"]["trim"],
            },
        )
        self.cache_id = f"{config['wilor']['id']}-{self.wilor_fingerprint[:12]}"
        implementation_paths = [
            REPO_ROOT / "preprocess/Hand2Gripper.py",
            REPO_ROOT / "preprocess/PipelineIO.py",
            REPO_ROOT / "preprocess/Preprocess.py",
            REPO_ROOT / "preprocess/WiLoRHands.py",
            REPO_ROOT / "preprocess/batch/dataset.py",
            REPO_ROOT / "preprocess/batch/lerobot.py",
            REPO_ROOT / "preprocess/batch/runner.py",
            REPO_ROOT / "real2sim/retarget_arx_with_mink.py",
        ]
        execution_dependencies = {
            "wilor_fingerprint": self.wilor_fingerprint,
            "camera_calibration_sha256": file_sha256(
                config["geometry"]["camera_calibration"]
            ),
            "eef_config_sha256": {
                name: file_sha256(value["eef_config"])
                for name, value in config["hand2gripper_variants"].items()
            },
            "implementation_sha256": {
                str(path.relative_to(REPO_ROOT)): file_sha256(path)
                for path in implementation_paths
            },
            "correction_dependencies": _correction_dependencies(
                config["correction_variants"]
            ),
        }
        config["base_config_fingerprint"] = config["config_fingerprint"]
        config["execution_dependencies"] = execution_dependencies
        config["config_fingerprint"] = fingerprint(
            {
                "base": config["base_config_fingerprint"],
                "dependencies": execution_dependencies,
            }
        )

        self.experiment_dir = (
            Path(config["outputs"]["experiment_root"]) / config["experiment_id"]
        )
        _override_incompatible_experiment(
            self.experiment_dir,
            config_fingerprint=config["config_fingerprint"],
            source_dataset_fingerprint=source_dataset_fingerprint,
            enabled=override_experiment,
        )
        self.logs_dir = self.experiment_dir / "logs"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = ExperimentManifest(
            self.experiment_dir / "manifest.json",
            config=config,
            source_dataset_fingerprint=source_dataset_fingerprint,
            source_episode_count=source_episode_count,
            git=git_state(),
        )
        # Keep the fully resolved configuration next to the manifest.
        resolved_path = self.experiment_dir / "resolved_experiment.yaml"
        resolved_text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        if not resolved_path.exists():
            resolved_path.write_text(resolved_text, encoding="utf-8")

    def cache_path(self, episode: EpisodeSource) -> Path:
        dataset_name = Path(self.config["source"]["ego_dataset"]).name
        return (
            Path(self.config["outputs"]["cache_root"])
            / dataset_name
            / self.source_dataset_fingerprint[:12]
            / self.cache_id
            / f"episode_{episode.episode_index:06d}"
            / "wilor_hands.json"
        )

    def video_segment_path(self, episode: EpisodeSource, video_key: str) -> Path:
        dataset_name = Path(self.config["source"]["ego_dataset"]).name
        window_id = (
            f"s{episode.crop_start_frames:06d}-e{episode.crop_end_frames:06d}"
        )
        safe_key = video_key.replace("/", "__")
        return (
            Path(self.config["outputs"]["video_cache_root"])
            / dataset_name
            / self.source_dataset_fingerprint[:12]
            / "h264-crf18-v1"
            / window_id
            / f"episode_{episode.episode_index:06d}"
            / f"{safe_key}.mp4"
        )

    def processing_video_path(self, episode: EpisodeSource) -> tuple[Path, int]:
        if not episode.has_trim:
            return episode.video_path, episode.start_frame
        output = self.video_segment_path(episode, episode.video_key)
        materialize_video_segment(
            episode.video_path,
            output,
            start_frame=episode.processing_start_frame,
            frame_count=episode.effective_length,
        )
        return output, 0

    def raw_eef_path(self, h2g_id: str, episode: EpisodeSource) -> Path:
        return (
            self.experiment_dir
            / "artifacts"
            / "eef"
            / h2g_id
            / f"episode_{episode.episode_index:06d}"
            / "eef_raw.json"
        )

    def variant_dir(self, run_id: str, episode: EpisodeSource) -> Path:
        return (
            self.experiment_dir
            / "artifacts"
            / "variants"
            / run_id
            / f"episode_{episode.episode_index:06d}"
        )

    def _record_failure(
        self,
        group: str,
        group_id: str,
        episode: EpisodeSource,
        signature: str,
        error: BaseException,
        *,
        group_meta: dict[str, Any] | None = None,
    ) -> None:
        value = {
            "status": "failed",
            "signature": signature,
            "updated_at": utc_now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        self.manifest.update_episode(
            group, group_id, episode.episode_index, value, group_meta=group_meta
        )
        print(f"    FAILED: {value['error']}")
        if self.fail_fast:
            raise error

    def run_logged(self, command: list[str], log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + " ".join(command) + "\n\n")
            log.flush()
            subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def run_wilor(self, episode: EpisodeSource) -> bool:
        output = self.cache_path(episode)
        cache_meta_path = output.with_name("wilor_hands.meta.json")
        signature = fingerprint(
            {
                "stage": "wilor",
                "version": 2,
                "episode": episode.identity(),
                "wilor_fingerprint": self.wilor_fingerprint,
            }
        )
        record = self.manifest.episode_stage("wilor", self.cache_id, episode.episode_index)
        shared_record = None
        if cache_meta_path.is_file():
            try:
                shared_record = json.loads(cache_meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                shared_record = None
        shared_valid = _is_complete(shared_record, signature, [output, cache_meta_path])
        if shared_valid and shared_record.get("output_fingerprint") != file_sha256(output):
            shared_valid = False
        record_valid = _is_complete(record, signature, [output])
        if record_valid and record.get("output_fingerprint") != file_sha256(output):
            record_valid = False
        if "wilor" not in self.force_stages and record_valid:
            print("    wilor: skip")
            return True
        group_meta = {
            "id": self.cache_id,
            "config_fingerprint": self.wilor_fingerprint,
            "config": self.config["wilor"],
        }
        if "wilor" not in self.force_stages and shared_valid:
            self.manifest.update_episode(
                "wilor",
                self.cache_id,
                episode.episode_index,
                dict(shared_record),
                group_meta=group_meta,
            )
            print("    wilor: skip (shared cache)")
            return True
        try:
            self.manifest.update_episode(
                "wilor",
                self.cache_id,
                episode.episode_index,
                {"status": "running", "signature": signature, "updated_at": utc_now()},
                group_meta=group_meta,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            processing_video, processing_start_frame = self.processing_video_path(episode)
            with tempfile.TemporaryDirectory(prefix="tate-wilor-", dir=output.parent) as tmp:
                session = Path(tmp) / "session"
                command = [
                    sys.executable,
                    "-m",
                    "preprocess.reconstruct_wilor",
                    "--video",
                    str(processing_video),
                    "--session",
                    str(session),
                    "--cfg",
                    self.config["wilor"]["preprocess_config"],
                    "--start-frame",
                    str(processing_start_frame),
                    "--max-frames",
                    str(episode.effective_length),
                ]
                if self.config["wilor"].get("pretrained_dir"):
                    command.extend(
                        ["--wilor-pretrained-dir", self.config["wilor"]["pretrained_dir"]]
                    )
                log_path = self.logs_dir / "wilor" / f"episode_{episode.episode_index:06d}.log"
                self.run_logged(command, log_path)
                generated = session / "preprocess" / "wilor_hands.json"
                payload = json.loads(generated.read_text(encoding="utf-8"))
                if payload.get("schema") != "tate.wilor_hands":
                    raise ValueError(f"unexpected WiLoR cache schema: {generated}")
                if int(payload.get("total_frames", -1)) != episode.effective_length:
                    raise ValueError(
                        f"WiLoR produced {payload.get('total_frames')} frames for episode "
                        f"{episode.episode_index}, expected {episode.effective_length}"
                    )
                payload["source_video_original"] = str(episode.video_path)
                payload["temporal_window"] = episode.temporal_window()
                temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, output)
            completed_record = {
                "status": "complete",
                "signature": signature,
                "updated_at": utc_now(),
                "source_video": str(episode.video_path),
                "source_episode_index": episode.episode_index,
                "output": str(output),
                "output_fingerprint": file_sha256(output),
                "total_frames": episode.effective_length,
                "temporal_window": episode.temporal_window(),
                "producer_experiment": self.config["experiment_id"],
                "log": str(log_path),
            }
            atomic_write_json(cache_meta_path, completed_record)
            self.manifest.update_episode(
                "wilor",
                self.cache_id,
                episode.episode_index,
                completed_record,
                group_meta=group_meta,
            )
            print("    wilor: complete")
            return True
        except Exception as error:  # noqa: BLE001
            self._record_failure(
                "wilor", self.cache_id, episode, signature, error, group_meta=group_meta
            )
            return False

    def run_eef(self, h2g_id: str, episode: EpisodeSource) -> bool:
        cache = self.cache_path(episode)
        output = self.raw_eef_path(h2g_id, episode)
        h2g = self.config["hand2gripper_variants"][h2g_id]
        calibration = Path(self.config["geometry"]["camera_calibration"])
        eef_config = Path(h2g["eef_config"])
        signature = fingerprint(
            {
                "stage": "eef",
                "version": 2,
                "cache_sha256": file_sha256(cache) if cache.is_file() else None,
                "h2g": h2g,
                "eef_config_sha256": file_sha256(eef_config),
                "calibration_sha256": file_sha256(calibration),
                "arm_mode": self.config["trajectory"]["arm_mode"],
                "active_sides": self.config["trajectory"]["active_sides"],
            }
        )
        record = self.manifest.episode_stage("hand2gripper", h2g_id, episode.episode_index)
        if "eef" not in self.force_stages and _is_complete(record, signature, [output]):
            print(f"    eef[{h2g_id}]: skip")
            return True
        group_meta = {
            "id": h2g_id,
            "config": h2g,
            "config_fingerprint": fingerprint(h2g),
        }
        try:
            if not cache.is_file():
                raise FileNotFoundError(
                    f"WiLoR cache is missing for episode {episode.episode_index}: {cache}"
                )
            self.manifest.update_episode(
                "hand2gripper",
                h2g_id,
                episode.episode_index,
                {"status": "running", "signature": signature, "updated_at": utc_now()},
                group_meta=group_meta,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
            export_eef_from_cache(
                cache,
                calibration,
                temporary,
                eef_config,
                grasp_close_ratio=h2g.get("grasp_close_ratio"),
                grasp_open_ratio=h2g.get("grasp_open_ratio"),
                grasp_min_frames=h2g.get("grasp_min_frames"),
                hand2gripper_mode=h2g["mode"],
                grasp_mode=h2g["grasp_mode"],
                arm_mode=self.config["trajectory"]["arm_mode"],
                active_sides=self.config["trajectory"]["active_sides"],
            )
            summary = _eef_summary(temporary)
            if summary["total_frames"] != episode.effective_length:
                raise ValueError(
                    f"EEF has {summary['total_frames']} frames, expected "
                    f"{episode.effective_length}"
                )
            os.replace(temporary, output)
            self.manifest.update_episode(
                "hand2gripper",
                h2g_id,
                episode.episode_index,
                {
                    "status": "complete",
                    "signature": signature,
                    "updated_at": utc_now(),
                    "source_wilor_cache": str(cache),
                    "raw_eef": str(output),
                    "raw_eef_fingerprint": file_sha256(output),
                    **summary,
                },
                group_meta=group_meta,
            )
            print(
                f"    eef[{h2g_id}]: complete "
                f"valid L/R={summary['valid_frames']['left']}/{summary['valid_frames']['right']}"
            )
            return True
        except Exception as error:  # noqa: BLE001
            self._record_failure(
                "hand2gripper", h2g_id, episode, signature, error, group_meta=group_meta
            )
            return False

    def _variant_values(self, run_id: str, episode: EpisodeSource) -> dict[str, Any]:
        current = self.manifest.episode_stage("variants", run_id, episode.episode_index)
        return dict(current or {})

    def run_correction(self, run: dict[str, str], episode: EpisodeSource) -> bool:
        run_id = run["id"]
        raw_eef = self.raw_eef_path(run["hand2gripper"], episode)
        output = self.variant_dir(run_id, episode) / "eef.json"
        correction = self.config["correction_variants"][run["correction"]]
        artifact = correction.get("artifact")
        signature = fingerprint(
            {
                "stage": "correct",
                "version": 1,
                "raw_eef_sha256": file_sha256(raw_eef) if raw_eef.is_file() else None,
                "correction": correction,
                "artifact_sha256": file_sha256(artifact)
                if artifact and Path(artifact).is_file()
                else None,
                "dependencies": _correction_dependencies({run["correction"]: correction}),
            }
        )
        record = self._variant_values(run_id, episode)
        if "correct" not in self.force_stages and _is_complete(
            record.get("correction"), signature, [output]
        ):
            print(f"    correct[{run_id}]: skip")
            return True
        group_meta = {
            "id": run_id,
            "hand2gripper_id": run["hand2gripper"],
            "correction_id": run["correction"],
            "config_fingerprint": fingerprint(
                {
                    "hand2gripper": self.config["hand2gripper_variants"][run["hand2gripper"]],
                    "correction": correction,
                    "correction_dependencies": _correction_dependencies(
                        {run["correction"]: correction}
                    ),
                    "camera_calibration_sha256": file_sha256(
                        self.config["geometry"]["camera_calibration"]
                    ),
                    "eef_config_sha256": file_sha256(
                        self.config["hand2gripper_variants"][run["hand2gripper"]][
                            "eef_config"
                        ]
                    ),
                }
            ),
            "correction_artifact": correction.get("artifact"),
            "correction_config": correction,
        }
        try:
            if not raw_eef.is_file():
                raise FileNotFoundError(f"raw EEF is missing: {raw_eef}")
            running_values = self._variant_values(run_id, episode)
            running_values.update(
                {
                    "status": "running",
                    "source_episode_index": episode.episode_index,
                    "raw_eef": str(raw_eef),
                    "correction": {
                        "status": "running",
                        "signature": signature,
                        "updated_at": utc_now(),
                    },
                }
            )
            self.manifest.update_episode(
                "variants",
                run_id,
                episode.episode_index,
                running_values,
                group_meta=group_meta,
            )
            apply_correction(
                correction,
                source_eef=raw_eef,
                output_eef=output,
                episode_index=episode.episode_index,
                experiment_id=self.config["experiment_id"],
                variant_id=run_id,
            )
            summary = _eef_summary(output)
            values = self._variant_values(run_id, episode)
            values.update(
                {
                    "status": "complete",
                    "source_episode_index": episode.episode_index,
                    "raw_eef": str(raw_eef),
                    "final_eef": str(output),
                    "final_eef_fingerprint": file_sha256(output),
                    "valid_frames": summary["valid_frames"],
                    "grasp_events": summary["grasp_events"],
                    "correction": {
                        "status": "complete",
                        "signature": signature,
                        "updated_at": utc_now(),
                        "mode": correction["mode"],
                    },
                }
            )
            self.manifest.update_episode(
                "variants", run_id, episode.episode_index, values, group_meta=group_meta
            )
            print(f"    correct[{run_id}]: complete")
            return True
        except Exception as error:  # noqa: BLE001
            from correction.apply import CorrectionNotApplicableError

            expected_data_mismatch = isinstance(error, CorrectionNotApplicableError)
            status = "skipped" if expected_data_mismatch else "failed"
            values = self._variant_values(run_id, episode)
            values["status"] = status
            for key in ("final_eef", "final_eef_fingerprint", "ik"):
                values.pop(key, None)
            values["correction"] = {
                "status": status,
                "signature": signature,
                "updated_at": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
            self.manifest.update_episode(
                "variants", run_id, episode.episode_index, values, group_meta=group_meta
            )
            label = "SKIPPED" if expected_data_mismatch else "FAILED"
            print(f"    correct[{run_id}] {label}: {type(error).__name__}: {error}")
            if self.fail_fast and not expected_data_mismatch:
                raise
            return False

    def run_retarget(self, run: dict[str, str], episode: EpisodeSource) -> bool:
        run_id = run["id"]
        final_eef = self.variant_dir(run_id, episode) / "eef.json"
        output = self.variant_dir(run_id, episode) / "ik.npz"
        retarget = self.config["retarget"]
        signature = fingerprint(
            {
                "stage": "retarget",
                "version": 1,
                "eef_sha256": file_sha256(final_eef) if final_eef.is_file() else None,
                "retarget": retarget,
                "script_sha256": file_sha256(REPO_ROOT / "real2sim/retarget_arx_with_mink.py"),
            }
        )
        record = self._variant_values(run_id, episode)
        if "retarget" not in self.force_stages and _is_complete(
            record.get("retarget"), signature, [output]
        ):
            print(f"    retarget[{run_id}]: skip")
            return True
        try:
            if not final_eef.is_file():
                raise FileNotFoundError(f"final EEF is missing: {final_eef}")
            running_values = self._variant_values(run_id, episode)
            running_values["retarget"] = {
                "status": "running",
                "signature": signature,
                "updated_at": utc_now(),
            }
            self.manifest.update_episode(
                "variants", run_id, episode.episode_index, running_values
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.stem}.tmp-{os.getpid()}.npz")
            command = [
                sys.executable,
                "real2sim/retarget_arx_with_mink.py",
                "--eef",
                str(final_eef),
                "--out",
                str(temporary),
                "--scene",
                retarget["scene"],
                *_cli_options(retarget.get("options") or {}),
            ]
            log_path = self.logs_dir / "retarget" / run_id / f"episode_{episode.episode_index:06d}.log"
            self.run_logged(command, log_path)
            with np.load(temporary, allow_pickle=False) as result:
                if len(result["time_s"]) != episode.effective_length:
                    raise ValueError(
                        f"IK has {len(result['time_s'])} frames, expected "
                        f"{episode.effective_length}"
                    )
            os.replace(temporary, output)
            values = self._variant_values(run_id, episode)
            values.update({"status": "complete", "ik": str(output)})
            values["retarget"] = {
                "status": "complete",
                "signature": signature,
                "updated_at": utc_now(),
                "output_fingerprint": file_sha256(output),
                "log": str(log_path),
            }
            self.manifest.update_episode("variants", run_id, episode.episode_index, values)
            print(f"    retarget[{run_id}]: complete")
            return True
        except Exception as error:  # noqa: BLE001
            values = self._variant_values(run_id, episode)
            # Evaluation consumes final EEF and should remain available even if
            # the optional policy-training retarget stage fails.
            values["status"] = (
                "complete"
                if (values.get("correction") or {}).get("status") == "complete"
                else "failed"
            )
            values["retarget"] = {
                "status": "failed",
                "signature": signature,
                "updated_at": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
            self.manifest.update_episode("variants", run_id, episode.episode_index, values)
            print(f"    retarget[{run_id}] FAILED: {type(error).__name__}: {error}")
            if self.fail_fast:
                raise
            return False

    def run_episode_stages(self, stages: set[str]) -> None:
        h2g_ids = sorted({run["hand2gripper"] for run in self.runs})
        for number, episode in enumerate(self.episodes, start=1):
            print(
                f"[{number}/{len(self.episodes)}] episode_{episode.episode_index:06d} "
                f"frames={episode.length}->{episode.effective_length}"
            )
            if "wilor" in stages and not self.run_wilor(episode):
                continue
            for h2g_id in h2g_ids:
                if "eef" in stages:
                    self.run_eef(h2g_id, episode)
            for run in self.runs:
                correction_ready = True
                if "correct" in stages:
                    correction_ready = self.run_correction(run, episode)
                if (
                    correction_ready
                    and "retarget" in stages
                    and self.config["retarget"]["enabled"]
                ):
                    self.run_retarget(run, episode)

    def write_summary_tsv(self) -> None:
        lines = [
            "variant\tepisode\tstatus\tvalid_left\tvalid_right\tevents_left\tevents_right\tfinal_eef\tik"
        ]
        for run in self.config["runs"]:
            records = (self.manifest.data.get("variants", {}).get(run["id"], {}).get("episodes", {}))
            for episode_id, record in sorted(records.items(), key=lambda item: int(item[0])):
                valid = record.get("valid_frames") or {}
                events = record.get("grasp_events") or {}
                lines.append(
                    "\t".join(
                        str(value)
                        for value in (
                            run["id"], episode_id, record.get("status", "unknown"),
                            valid.get("left", ""), valid.get("right", ""),
                            events.get("left", ""), events.get("right", ""),
                            record.get("final_eef", ""), record.get("ik", ""),
                        )
                    )
                )
        path = self.experiment_dir / "manifest.tsv"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, path)
