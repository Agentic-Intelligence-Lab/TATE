#!/usr/bin/env python3
"""PyTorch training entry point for ARX dual-arm EEF Pi0/Pi0.5 models."""

# OpenPI and this repository are added to sys.path before project imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "thirdparty" / "openpi"
for path in (REPO_ROOT, OPENPI_ROOT, OPENPI_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.arx_eef_config import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_REPO_ID,
    build_config,
    dataset_home_from_root,
    train_steps_for_dataset,
)
from training.arx_eef_policy import ACTION_DIM


DEFAULT_PI05_WEIGHT_PATH = "/mnt/workspace/sunxiaoquan/models/pi05_base"


def dataset_action_mask(dataset_root: str | Path) -> list[bool]:
    """Read the dataset-level 16-D loss mask written by our cotrain exporter."""
    info_path = Path(dataset_root).expanduser() / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    mask = (info.get("arx_eef") or {}).get("default_action_mask")
    if mask is None:
        return [True] * ACTION_DIM
    if not isinstance(mask, list) or len(mask) != ACTION_DIM:
        raise ValueError(f"invalid arx_eef.default_action_mask in {info_path}")
    if not any(mask):
        raise ValueError(f"action mask in {info_path} masks every action dimension")
    return [bool(value) for value in mask]


def _patch_pytorch_action_loss(loss_action_dim: int, action_mask: list[bool]) -> None:
    if loss_action_dim <= 0:
        raise ValueError("loss action dimension must be positive")

    from openpi.models_pytorch import pi0_pytorch

    original_forward = pi0_pytorch.PI0Pytorch.forward
    if getattr(original_forward, "_arx_eef_loss_patched", False):
        return

    def forward_with_masked_loss(self, observation, actions, noise=None, time=None):
        losses = original_forward(self, observation, actions, noise=noise, time=time)
        if losses.shape[-1] < loss_action_dim:
            raise ValueError(f"model returned only {losses.shape[-1]} action losses")
        losses = losses[..., :loss_action_dim]
        import torch
        active = torch.as_tensor(action_mask[:loss_action_dim], dtype=losses.dtype, device=losses.device)
        active_count = active.sum()
        if active_count <= 0:
            raise ValueError("action mask has no active dimensions")
        # OpenPI's outer loop calls mean().  Rescale so it remains the mean
        # over active dimensions rather than being diluted by masked entries.
        return losses * active * (loss_action_dim / active_count)

    forward_with_masked_loss._arx_eef_loss_patched = True
    pi0_pytorch.PI0Pytorch.forward = forward_with_masked_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--assets-base-dir", default=None)
    parser.add_argument("--checkpoint-base-dir", default=None)
    parser.add_argument("--exp-name", default="arx_eef_pi05_pytorch")
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
    parser.add_argument(
        "--action-mask",
        nargs=ACTION_DIM,
        type=int,
        default=None,
        metavar="MASK",
        help="Optional 16-value 0/1 loss-mask override; defaults to dataset metadata.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.loss_action_dim > ACTION_DIM:
        raise ValueError(f"--loss-action-dim cannot exceed ARX EEF action dimension {ACTION_DIM}")
    if args.loss_action_dim > 32:
        raise ValueError("--loss-action-dim cannot exceed the model action dimension 32")
    if args.action_mask is not None and any(value not in (0, 1) for value in args.action_mask):
        raise ValueError("--action-mask values must be 0 or 1")

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
    action_mask = [bool(value) for value in args.action_mask] if args.action_mask is not None else dataset_action_mask(args.dataset_root)
    if args.loss_action_dim != ACTION_DIM and not all(action_mask[args.loss_action_dim:]):
        raise ValueError("--loss-action-dim cannot hide masked dimensions; use the 16-D default")
    print(f"ARX EEF action loss mask: {[int(value) for value in action_mask]}")
    _patch_pytorch_action_loss(args.loss_action_dim, action_mask)
    openpi_train_pytorch.train_loop(config)


if __name__ == "__main__":
    main()
