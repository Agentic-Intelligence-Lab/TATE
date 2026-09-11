#!/usr/bin/env python3
"""PyTorch training entry point for ARX dual-arm joint Pi0/Pi0.5 models."""

# OpenPI and this repository are added to sys.path before project imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "thirdparty" / "openpi"
for path in (REPO_ROOT, OPENPI_ROOT, OPENPI_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.arx_joint_config import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_REPO_ID,
    build_config,
    dataset_home_from_root,
    train_steps_for_dataset,
)
from training.arx_joint_policy import ACTION_DIM


DEFAULT_PI05_WEIGHT_PATH = "/mnt/workspace/sunxiaoquan/models/pi05_base"


def _patch_pytorch_action_loss_dim(loss_action_dim: int) -> None:
    if loss_action_dim <= 0:
        return

    from openpi.models_pytorch import pi0_pytorch

    original_forward = pi0_pytorch.PI0Pytorch.forward
    if getattr(original_forward, "_arx_joint_loss_patched", False):
        return

    def forward_with_cropped_loss(self, observation, actions, noise=None, time=None):
        losses = original_forward(self, observation, actions, noise=noise, time=time)
        if loss_action_dim >= losses.shape[-1]:
            return losses
        return losses[..., :loss_action_dim]

    forward_with_cropped_loss._arx_joint_loss_patched = True
    pi0_pytorch.PI0Pytorch.forward = forward_with_cropped_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--assets-base-dir", default=None)
    parser.add_argument("--checkpoint-base-dir", default=None)
    parser.add_argument("--exp-name", default="arx_joint_pi05_pytorch")
    parser.add_argument("--model", choices=["pi0", "pi05"], default="pi05")
    parser.add_argument("--pytorch-weight-path", default=DEFAULT_PI05_WEIGHT_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--loss-action-dim", type=int, default=ACTION_DIM)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.loss_action_dim > ACTION_DIM:
        raise ValueError(f"--loss-action-dim cannot exceed ARX action dimension {ACTION_DIM}")
    if args.loss_action_dim > 32:
        raise ValueError("--loss-action-dim cannot exceed the model action dimension 32")

    weight_path = Path(args.pytorch_weight_path).expanduser()
    if not (weight_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"missing model.safetensors under {weight_path}")
    num_train_steps = args.num_train_steps
    if num_train_steps is None:
        num_train_steps = train_steps_for_dataset(args.dataset_root, args.batch_size)

    os.environ["HF_LEROBOT_HOME"] = str(dataset_home_from_root(args.dataset_root, args.repo_id))
    config = build_config(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        model=args.model,
        low_mem=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=num_train_steps,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        pytorch_weight_path=str(weight_path),
        wandb_enabled=args.wandb,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    from scripts import train_pytorch as openpi_train_pytorch

    openpi_train_pytorch.init_logging()
    _patch_pytorch_action_loss_dim(args.loss_action_dim)
    openpi_train_pytorch.train_loop(config)


if __name__ == "__main__":
    main()
