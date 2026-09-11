#!/usr/bin/env python3
"""Evaluate an ARX joint checkpoint on samples from a converted dataset."""

# OpenPI and this repository are added to sys.path before project imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
import sys

import numpy as np
import safetensors.torch
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "thirdparty" / "openpi"
for path in (REPO_ROOT, OPENPI_ROOT, OPENPI_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.arx_joint_config import DEFAULT_DATASET_ROOT, DEFAULT_REPO_ID, build_config, dataset_home_from_root
from training.arx_joint_policy import ACTION_DIM, JOINT_NAMES


def _resolve_checkpoint_dir(checkpoint_dir: Path, step: int | None) -> Path:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if (checkpoint_dir / "model.safetensors").is_file():
        if step is not None:
            raise ValueError("--step cannot be used with an exact checkpoint directory")
        return checkpoint_dir
    if step is not None:
        path = checkpoint_dir / str(step)
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError(f"missing model.safetensors under {path}")
        return path
    steps = sorted(int(path.name) for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"no numeric checkpoints under {checkpoint_dir}")
    return checkpoint_dir / str(steps[-1])


def _build_model(config, device: torch.device):
    import openpi.models.pi0_config
    import openpi.models_pytorch.pi0_pytorch

    model_cfg = config.model
    if not isinstance(model_cfg, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = dataclasses.replace(model_cfg, dtype=config.pytorch_training_precision, pytorch_compile_mode=None)
    return openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device).eval()


def _unnormalize(actions: np.ndarray, norm_stats, use_quantiles: bool) -> np.ndarray:
    stats = norm_stats["actions"]
    if use_quantiles:
        q01 = np.asarray(stats.q01, dtype=np.float32)
        q99 = np.asarray(stats.q99, dtype=np.float32)
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    mean = np.asarray(stats.mean, dtype=np.float32)
    std = np.asarray(stats.std, dtype=np.float32)
    return actions * (std + 1e-6) + mean


def _move_to_device(tree, device: torch.device):
    import jax

    return jax.tree.map(lambda value: value.to(device) if hasattr(value, "to") else value, tree)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--exp-name", default="arx_joint_pi05_pytorch")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--model", choices=["pi0", "pi05"], default="pi05")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--output-npz", default=None)
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.batch_size <= 0 or args.sample_steps <= 0:
        raise ValueError("--batch-size and --sample-steps must be positive")

    os.environ["HF_LEROBOT_HOME"] = str(dataset_home_from_root(args.dataset_root, args.repo_id))
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    config = build_config(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        model=args.model,
        low_mem=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=1,
        pytorch_weight_path=None,
        wandb_enabled=False,
    )
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else config.checkpoint_dir
    checkpoint = _resolve_checkpoint_dir(checkpoint_root, args.step)
    model = _build_model(config, device)
    safetensors.torch.load_model(model, checkpoint / "model.safetensors", device=str(device))

    from openpi.training import data_loader

    loader = data_loader.create_data_loader(
        config,
        shuffle=False,
        num_batches=max((args.num_samples + args.batch_size - 1) // args.batch_size, 1),
        framework="pytorch",
    )
    predictions = []
    labels = []
    with torch.inference_mode():
        remaining = args.num_samples
        for observation, actions in loader:
            take = min(remaining, actions.shape[0])
            observation = _move_to_device(observation, device)
            labels_t = actions[:take].to(device=device, dtype=torch.float32)
            if args.zero_noise:
                batch_size = observation.state.shape[0]
                noise = torch.zeros(
                    batch_size,
                    config.model.action_horizon,
                    config.model.action_dim,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                noise = None
            predicted = model.sample_actions(
                device,
                observation,
                noise=noise,
                num_steps=args.sample_steps,
            )
            predictions.append(predicted[:take, 0, :ACTION_DIM].cpu().numpy())
            labels.append(labels_t[:, 0, :ACTION_DIM].cpu().numpy())
            remaining -= take
            if remaining <= 0:
                break

    pred_norm = np.concatenate(predictions, axis=0)
    label_norm = np.concatenate(labels, axis=0)
    data_config = config.data.create(config.assets_dirs, config.model)
    pred = _unnormalize(pred_norm, data_config.norm_stats, data_config.use_quantile_norm)
    label = _unnormalize(label_norm, data_config.norm_stats, data_config.use_quantile_norm)
    rmse = np.sqrt(np.mean(np.square(pred - label), axis=0))

    print(f"checkpoint: {checkpoint}")
    print(f"samples: {len(pred)}")
    print(f"overall_rmse: {float(np.sqrt(np.mean(np.square(pred - label)))):.6f}")
    for name, value in zip(JOINT_NAMES, rmse, strict=True):
        print(f"{name}: {float(value):.6f}")

    if args.output_npz:
        output = Path(args.output_npz).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, pred=pred, label=label, pred_normalized=pred_norm, label_normalized=label_norm)
        print(f"wrote: {output}")


if __name__ == "__main__":
    main()
