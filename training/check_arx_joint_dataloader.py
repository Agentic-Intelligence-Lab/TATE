#!/usr/bin/env python3
"""Smoke test the ARX OpenPI data loader."""

# The repository root is added to sys.path before importing project modules.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.arx_joint_config import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_REPO_ID,
    build_config,
    dataset_home_from_root,
)
from training.arx_joint_policy import ACTION_DIM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--model", choices=["pi0", "pi05"], default="pi05")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-norm-stats", action="store_true")
    args = parser.parse_args()

    os.environ["HF_LEROBOT_HOME"] = str(dataset_home_from_root(args.dataset_root, args.repo_id))
    from openpi.training import data_loader

    config = build_config(
        repo_id=args.repo_id,
        model=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=10,
        wandb_enabled=False,
    )
    loader = data_loader.create_data_loader(
        config,
        shuffle=False,
        num_batches=1,
        skip_norm_stats=args.skip_norm_stats,
        framework="pytorch",
    )
    observation, actions = next(iter(loader))
    print("images:")
    for key, value in observation.images.items():
        print(f"  {key}: {tuple(value.shape)} {value.dtype}")
    print("image masks:")
    for key, value in observation.image_masks.items():
        print(f"  {key}: {tuple(value.shape)} {value.dtype} first={value[0]}")
    print(f"state: {tuple(observation.state.shape)} {observation.state.dtype}")
    print(f"actions: {tuple(actions.shape)} {actions.dtype}")
    print(f"ARX action dimensions: {ACTION_DIM}")
    print(f"state tail abs max: {float(np.abs(observation.state[..., ACTION_DIM:]).max()):.6g}")
    print(f"action tail abs max: {float(np.abs(actions[..., ACTION_DIM:]).max()):.6g}")


if __name__ == "__main__":
    main()
