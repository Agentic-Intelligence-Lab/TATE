#!/usr/bin/env python3
"""Compute OpenPI normalization stats for an ARX joint dataset."""

# The repository root is added to sys.path before importing project modules.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.arx_joint_config import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_REPO_ID,
    build_config,
    dataset_home_from_root,
    train_steps_for_dataset,
)
from openpi.shared import normalize


def compute(args: argparse.Namespace) -> None:
    os.environ["HF_LEROBOT_HOME"] = str(dataset_home_from_root(args.dataset_root, args.repo_id))

    num_train_steps = args.num_train_steps
    if num_train_steps is None:
        num_train_steps = train_steps_for_dataset(args.dataset_root, args.batch_size)
    config = build_config(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        model=args.model,
        low_mem=not args.full_finetune,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=num_train_steps,
        assets_base_dir=args.assets_base_dir,
        wandb_enabled=False,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"missing converted dataset metadata: {episodes_path}")

    episode_records = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for episode in tqdm.tqdm(episode_records, desc="Computing ARX norm stats"):
        episode_index = int(episode["episode_index"])
        parquet_path = (
            dataset_root
            / "data"
            / f"chunk-{episode_index // 1000:03d}"
            / f"file-{episode_index:03d}.parquet"
        )
        table = pq.read_table(parquet_path)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if state.shape != action.shape or state.shape[-1] != 14:
            raise ValueError(f"invalid ARX episode shapes in {parquet_path}: {state.shape}, {action.shape}")

        action_windows = np.stack(
            [
                action[np.minimum(np.arange(index, index + config.model.action_horizon), len(action) - 1)]
                for index in range(len(action))
            ],
            axis=0,
        )
        stats["state"].update(state)
        stats["actions"].update(action_windows)

    output = config.assets_dirs / data_config.repo_id
    output.mkdir(parents=True, exist_ok=True)
    normalize.save(output, {key: value.get_statistics() for key, value in stats.items()})
    print(f"Wrote norm stats: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--assets-base-dir", default=None)
    parser.add_argument("--exp-name", default="arx_joint_debug")
    parser.add_argument("--model", choices=["pi0", "pi05"], default="pi05")
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-train-steps", type=int, default=None)
    args = parser.parse_args()
    compute(args)


if __name__ == "__main__":
    main()
