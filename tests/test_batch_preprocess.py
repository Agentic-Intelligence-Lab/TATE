import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from preprocess.batch.dataset import (
    EpisodeSource,
    discover_episodes,
    materialize_video_segment,
    video_frame_count,
)
from preprocess.batch.lerobot import _package_one_variant, build_episode_table, numeric_stats
from preprocess.batch.runner import _override_incompatible_experiment
from preprocess.batch_preprocess import parse_stages


def fixed_list(values: np.ndarray) -> pa.Array:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()), values.shape[1]
    )


def eef_payload(valid_left=(True, False), valid_right=(True, True)) -> dict:
    frames = []
    for index in range(2):
        frame = {"idx": index, "ts": index * 33_333_333}
        for side, key, valid in (
            ("left", "hand_l", valid_left[index]),
            ("right", "hand_r", valid_right[index]),
        ):
            if not valid:
                frame[key] = None
                continue
            pose = np.eye(4)
            pose[:3, 3] = [index, 1.0 if side == "left" else -1.0, 0.5]
            frame[key] = {
                "tcp_pose_eef_frame": pose.tolist(),
                "grasp_state": index % 2,
                "grasp_ratio": 0.4 + 0.1 * index,
            }
        frames.append(frame)
    return {
        "schema": "tate.dual_arm_eef",
        "schema_version": 2,
        "total_frames": 2,
        "fps": 30.0,
        "eef_coordinate_convention": {
            "pose_semantics": "arx_tcp",
            "per_side_frames": {
                "left": "left_flange_zero",
                "right": "right_flange_zero",
            },
        },
        "hand2gripper": {"mode": "finger_center"},
        "frames": frames,
    }


class BatchPreprocessTest(unittest.TestCase):
    def test_default_stages_skip_retarget_but_all_includes_it(self):
        self.assertEqual(
            parse_stages("default"),
            {"wilor", "eef", "correct", "visualize", "package"},
        )
        self.assertEqual(
            parse_stages("all"),
            {"wilor", "eef", "correct", "retarget", "visualize", "package"},
        )

    def test_override_removes_only_an_incompatible_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / "experiments" / "task"
            experiment.mkdir(parents=True)
            (experiment / "manifest.json").write_text(
                json.dumps(
                    {
                        "config_fingerprint": "old-config",
                        "source_ego_dataset": {"fingerprint": "old-dataset"},
                    }
                ),
                encoding="utf-8",
            )
            (experiment / "artifact.txt").write_text("replace me", encoding="utf-8")
            cache = root / "cache" / "wilor_hands.json"
            cache.parent.mkdir()
            cache.write_text("preserve me", encoding="utf-8")

            _override_incompatible_experiment(
                experiment,
                config_fingerprint="new-config",
                source_dataset_fingerprint="new-dataset",
                enabled=False,
            )
            self.assertTrue(experiment.is_dir())

            _override_incompatible_experiment(
                experiment,
                config_fingerprint="new-config",
                source_dataset_fingerprint="new-dataset",
                enabled=True,
            )
            self.assertFalse(experiment.exists())
            self.assertEqual(cache.read_text(encoding="utf-8"), "preserve me")

    def test_discovers_episode_from_lerobot_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta/episodes/chunk-000").mkdir(parents=True)
            (root / "data/chunk-000").mkdir(parents=True)
            (root / "videos/observation.images.head/chunk-000").mkdir(parents=True)
            (root / "meta/info.json").write_text(
                json.dumps(
                    {
                        "fps": 30,
                        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                    }
                ),
                encoding="utf-8",
            )
            pq.write_table(
                pa.table({"episode_index": [7], "frame_index": [0]}),
                root / "data/chunk-000/file-003.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "episode_index": 7,
                            "length": 1,
                            "data/chunk_index": 0,
                            "data/file_index": 3,
                            "videos/observation.images.head/chunk_index": 0,
                            "videos/observation.images.head/file_index": 4,
                            "videos/observation.images.head/from_timestamp": 2.0,
                            "videos/observation.images.head/to_timestamp": 2.033333,
                        }
                    ]
                ),
                root / "meta/episodes/chunk-000/file-009.parquet",
            )
            (root / "videos/observation.images.head/chunk-000/file-004.mp4").touch()
            episodes = discover_episodes(root)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].episode_index, 7)
            self.assertEqual(episodes[0].data_path.name, "file-003.parquet")
            self.assertEqual(episodes[0].video_path.name, "file-004.mp4")
            self.assertEqual(episodes[0].start_frame, 60)

    def test_episode_trim_window_and_exact_video_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_video = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(source_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (16, 16),
            )
            self.assertTrue(writer.isOpened())
            for index in range(20):
                writer.write(np.full((16, 16, 3), index * 10, dtype=np.uint8))
            writer.release()
            episode = EpisodeSource(
                episode_index=1,
                length=20,
                data_path=root / "data.parquet",
                metadata_path=root / "meta.parquet",
                metadata_row={},
                video_key="observation.images.head",
                video_path=source_video,
                video_from_timestamp=0.0,
                video_to_timestamp=2.0,
                fps=10.0,
            ).with_trim_seconds(0.3, 0.5)
            self.assertEqual(episode.crop_start_frames, 3)
            self.assertEqual(episode.crop_end_frames, 5)
            self.assertEqual(episode.effective_length, 12)
            output_video = root / "trimmed.mp4"
            materialize_video_segment(
                source_video,
                output_video,
                start_frame=episode.processing_start_frame,
                frame_count=episode.effective_length,
            )
            self.assertEqual(video_frame_count(output_video), 12)

    def test_builds_training_and_evaluation_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_data = root / "source.parquet"
            source = pa.table(
                {
                    "observation.state": fixed_list(np.zeros((2, 14))),
                    "action": fixed_list(np.zeros((2, 14))),
                    "timestamp": pa.array([0.0, 1.0 / 30.0], type=pa.float32()),
                    "frame_index": pa.array([0, 1], type=pa.int64()),
                    "episode_index": pa.array([4, 4], type=pa.int64()),
                    "index": pa.array([10, 11], type=pa.int64()),
                    "task_index": pa.array([0, 0], type=pa.int64()),
                }
            )
            pq.write_table(source, source_data)
            video = root / "video.mp4"
            video.touch()
            episode = EpisodeSource(
                episode_index=4,
                length=2,
                data_path=source_data,
                metadata_path=root / "meta.parquet",
                metadata_row={},
                video_key="observation.images.head",
                video_path=video,
                video_from_timestamp=0.0,
                video_to_timestamp=2.0 / 30.0,
                fps=30.0,
            )
            raw_path = root / "raw.json"
            final_path = root / "final.json"
            raw_path.write_text(json.dumps(eef_payload()), encoding="utf-8")
            final_path.write_text(json.dumps(eef_payload()), encoding="utf-8")
            ik_path = root / "ik.npz"
            np.savez_compressed(
                ik_path,
                left_arm_qpos=np.ones((2, 6), dtype=np.float32),
                right_arm_qpos=np.full((2, 6), 2.0, dtype=np.float32),
            )
            table, _ = build_episode_table(
                episode,
                derived_episode_index=0,
                dataset_start_index=100,
                raw_eef_path=raw_path,
                final_eef_path=final_path,
                ik_path=ik_path,
                lerobot_config={
                    "replace_state_action": True,
                    "gripper_open_raw": -3.4,
                    "gripper_closed_raw": 0.1,
                    "action_alignment": "same_frame",
                },
            )
            self.assertIn("tate.eef.left.pose", table.column_names)
            self.assertIn("tate.eef.right.valid", table.column_names)
            self.assertEqual(table["episode_index"].to_pylist(), [0, 0])
            self.assertEqual(table["tate.source_episode_index"].to_pylist(), [4, 4])
            self.assertEqual(table["tate.source_frame_index"].to_pylist(), [0, 1])
            np.testing.assert_allclose(
                table["tate.source_timestamp"].to_numpy(), [0.0, 1.0 / 30.0]
            )
            self.assertEqual(table["index"].to_pylist(), [100, 101])
            state = np.asarray(table["observation.state"].to_pylist())
            np.testing.assert_allclose(state[:, :6], 1.0)
            np.testing.assert_allclose(state[:, 7:13], 2.0)
            self.assertAlmostEqual(state[0, 6], -3.4, places=6)
            self.assertAlmostEqual(state[1, 13], 0.1, places=6)
            stats = numeric_stats(table)
            self.assertEqual(stats["tate.eef.left.pose"]["count"], [1])
            self.assertEqual(stats["tate.eef.right.pose"]["count"], [2])

    def test_crops_source_table_and_preserves_source_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_data = root / "source.parquet"
            source = pa.table(
                {
                    "observation.state": fixed_list(np.zeros((4, 14))),
                    "action": fixed_list(np.zeros((4, 14))),
                    "timestamp": pa.array(
                        [index / 30.0 for index in range(4)], type=pa.float32()
                    ),
                    "frame_index": pa.array([0, 1, 2, 3], type=pa.int64()),
                    "episode_index": pa.array([4, 4, 4, 4], type=pa.int64()),
                    "index": pa.array([10, 11, 12, 13], type=pa.int64()),
                    "task_index": pa.array([0, 0, 0, 0], type=pa.int64()),
                }
            )
            pq.write_table(source, source_data)
            video = root / "video.mp4"
            video.touch()
            episode = EpisodeSource(
                episode_index=4,
                length=4,
                data_path=source_data,
                metadata_path=root / "meta.parquet",
                metadata_row={},
                video_key="observation.images.head",
                video_path=video,
                video_from_timestamp=0.0,
                video_to_timestamp=4.0 / 30.0,
                fps=30.0,
                crop_start_frames=1,
                crop_end_frames=1,
            )
            raw_path = root / "raw.json"
            final_path = root / "final.json"
            raw_path.write_text(json.dumps(eef_payload()), encoding="utf-8")
            final_path.write_text(json.dumps(eef_payload()), encoding="utf-8")
            ik_path = root / "ik.npz"
            np.savez_compressed(
                ik_path,
                left_arm_qpos=np.ones((2, 6), dtype=np.float32),
                right_arm_qpos=np.full((2, 6), 2.0, dtype=np.float32),
            )
            table, _ = build_episode_table(
                episode,
                derived_episode_index=0,
                dataset_start_index=0,
                raw_eef_path=raw_path,
                final_eef_path=final_path,
                ik_path=ik_path,
                lerobot_config={
                    "replace_state_action": True,
                    "gripper_open_raw": -3.4,
                    "gripper_closed_raw": 0.1,
                    "action_alignment": "same_frame",
                },
            )
            self.assertEqual(table["frame_index"].to_pylist(), [0, 1])
            self.assertEqual(table["tate.source_frame_index"].to_pylist(), [1, 2])
            np.testing.assert_allclose(
                table["tate.source_timestamp"].to_numpy(), [1.0 / 30.0, 2.0 / 30.0]
            )
            np.testing.assert_allclose(
                table["timestamp"].to_numpy(), [0.0, 1.0 / 30.0], atol=1e-7
            )

    def test_materializes_self_contained_lerobot_variant(self):
        class FakeManifest:
            def __init__(self, path, data):
                self.path = path
                self.data = data

            def update_variant(self, variant_id, values):
                self.data.setdefault("variants", {}).setdefault(variant_id, {}).update(values)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            experiment_dir = root / "experiment"
            (source_root / "data/chunk-000").mkdir(parents=True)
            (source_root / "meta/episodes/chunk-000").mkdir(parents=True)
            (source_root / "videos/observation.images.head/chunk-000").mkdir(parents=True)
            source_table = pa.table(
                {
                    "observation.state": fixed_list(np.zeros((2, 14))),
                    "action": fixed_list(np.zeros((2, 14))),
                    "timestamp": pa.array([0.0, 1.0 / 30.0], type=pa.float32()),
                    "frame_index": pa.array([0, 1], type=pa.int64()),
                    "episode_index": pa.array([5, 5], type=pa.int64()),
                    "index": pa.array([20, 21], type=pa.int64()),
                    "task_index": pa.array([0, 0], type=pa.int64()),
                }
            )
            source_data = source_root / "data/chunk-000/file-002.parquet"
            pq.write_table(source_table, source_data)
            source_video = source_root / "videos/observation.images.head/chunk-000/file-003.mp4"
            source_video.write_bytes(b"video")
            metadata_row = {
                "episode_index": 5,
                "tasks": ["test task"],
                "length": 2,
                "data/chunk_index": 0,
                "data/file_index": 2,
                "dataset_from_index": 20,
                "dataset_to_index": 22,
                "videos/observation.images.head/chunk_index": 0,
                "videos/observation.images.head/file_index": 3,
                "videos/observation.images.head/from_timestamp": 0.0,
                "videos/observation.images.head/to_timestamp": 2.0 / 30.0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            metadata_path = source_root / "meta/episodes/chunk-000/file-000.parquet"
            pq.write_table(pa.Table.from_pylist([metadata_row]), metadata_path)
            pq.write_table(
                pa.Table.from_pylist([{"task_index": 0, "task": "test task"}]),
                source_root / "meta/tasks.parquet",
            )
            info = {
                "codebase_version": "v3.0",
                "robot_type": "source",
                "fps": 30,
                "total_episodes": 1,
                "total_frames": 2,
                "total_tasks": 1,
                "chunks_size": 1000,
                "splits": {"train": "0:1"},
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [14], "names": None},
                    "action": {"dtype": "float32", "shape": [14], "names": None},
                    "observation.images.head": {
                        "dtype": "video", "shape": [1, 1, 3],
                        "names": ["height", "width", "channels"],
                    },
                },
            }
            (source_root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
            (source_root / "meta/stats.json").write_text(
                json.dumps({"observation.images.head": {"count": [2]}}), encoding="utf-8"
            )

            artifact_dir = experiment_dir / "artifact"
            artifact_dir.mkdir(parents=True)
            raw_path = artifact_dir / "raw.json"
            final_path = artifact_dir / "final.json"
            raw_path.write_text(json.dumps(eef_payload()), encoding="utf-8")
            final_path.write_text(json.dumps(eef_payload()), encoding="utf-8")
            ik_path = artifact_dir / "ik.npz"
            np.savez_compressed(
                ik_path,
                left_arm_qpos=np.ones((2, 6), dtype=np.float32),
                right_arm_qpos=np.full((2, 6), 2.0, dtype=np.float32),
            )
            episode = EpisodeSource(
                episode_index=5,
                length=2,
                data_path=source_data,
                metadata_path=metadata_path,
                metadata_row=metadata_row,
                video_key="observation.images.head",
                video_path=source_video,
                video_from_timestamp=0.0,
                video_to_timestamp=2.0 / 30.0,
                fps=30.0,
            )
            run = {"id": "finger_center__none", "hand2gripper": "finger_center", "correction": "none"}
            manifest = FakeManifest(
                experiment_dir / "manifest.json",
                {
                    "variants": {
                        run["id"]: {
                            "episodes": {
                                "5": {
                                    "final_eef": str(final_path),
                                    "raw_eef": str(raw_path),
                                    "ik": str(ik_path),
                                }
                            }
                        }
                    }
                },
            )
            runner = SimpleNamespace(
                config={
                    "experiment_id": "test_experiment",
                    "config_fingerprint": "cfg",
                    "source": {"ego_dataset": str(source_root)},
                    "trajectory": {
                        "arm_mode": "single_arm",
                        "active_sides": ["right"],
                        "trim": {"start_seconds": 0.0, "end_seconds": 0.0},
                    },
                    "lerobot": {
                        "replace_state_action": True,
                        "require_all_episodes": True,
                        "video_mode": "hardlink",
                        "action_alignment": "same_frame",
                        "gripper_open_raw": -3.4,
                        "gripper_closed_raw": 0.1,
                        "task": "derived task",
                    },
                },
                source_dataset_fingerprint="source-fingerprint",
                experiment_dir=experiment_dir,
                episodes=[episode],
                manifest=manifest,
            )
            _package_one_variant(runner, run, force=False)
            destination = experiment_dir / "datasets" / run["id"]
            output = pq.read_table(destination / "data/chunk-000/file-000.parquet")
            self.assertIn("tate.eef.right.pose", output.column_names)
            self.assertEqual(output["tate.source_episode_index"].to_pylist(), [5, 5])
            self.assertEqual(output["task_index"].to_pylist(), [0, 0])
            tasks = pq.read_table(destination / "meta/tasks.parquet").to_pylist()
            self.assertEqual(tasks, [{"task_index": 0, "task": "derived task"}])
            self.assertEqual(
                (destination / "videos/observation.images.head/chunk-000/file-000.mp4").read_bytes(),
                b"video",
            )
            provenance = json.loads(
                (destination / "meta/tate_preprocess.json").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["variant_id"], run["id"])
            self.assertEqual(provenance["episodes"][0]["source_episode_index"], 5)
            self.assertEqual(
                provenance["episodes"][0]["final_eef_fingerprint"],
                provenance["episodes"][0]["raw_eef_fingerprint"],
            )
            # This is the integration boundary used by the independently
            # migrated alignment/evaluation pipeline.
            from evaluation.loaders import load_eef_json, load_lerobot_episode

            trajectory = load_lerobot_episode(destination, 5)
            sidecar = load_eef_json(final_path, episode_id=5)
            self.assertEqual(trajectory.episode_id, 5)
            self.assertEqual(trajectory.sides["left"].frame_name, "left_flange_zero")
            self.assertEqual(trajectory.sides["left"].valid.tolist(), [True, False])
            np.testing.assert_allclose(trajectory.timestamps_s, sidecar.timestamps_s, atol=1e-7)
            for side in ("left", "right"):
                valid = sidecar.sides[side].valid
                np.testing.assert_array_equal(trajectory.sides[side].valid, valid)
                np.testing.assert_array_equal(
                    trajectory.sides[side].gripper_binary,
                    sidecar.sides[side].gripper_binary,
                )
                np.testing.assert_allclose(
                    trajectory.sides[side].pose_xyzw[valid],
                    sidecar.sides[side].pose_xyzw[valid],
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
