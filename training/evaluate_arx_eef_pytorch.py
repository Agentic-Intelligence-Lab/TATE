#!/usr/bin/env python3
"""Offline first-action regression evaluation for an ARX EEF Pi0/Pi0.5 checkpoint.

This compares the first action in each sampled action chunk with the action in
a LeRobot dataset. It is not a robot rollout or a task-success evaluation.
"""

# OpenPI and this repository are added to sys.path before project imports.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

import numpy as np
import safetensors.torch
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_openpi_root() -> Path:
    """Find OpenPI when TATE and the OpenPI checkout live in separate trees."""
    candidates: list[Path] = []
    if configured := os.environ.get("TATE_OPENPI_ROOT"):
        candidates.append(Path(configured).expanduser())
    candidates.extend((REPO_ROOT / "thirdparty" / "openpi", Path.cwd()))
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "src" / "openpi").is_dir() and (candidate / "scripts" / "train_pytorch.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find OpenPI. Set TATE_OPENPI_ROOT to an OpenPI checkout containing "
        "src/openpi and scripts/train_pytorch.py. Looked in: " + ", ".join(map(str, candidates))
    )


OPENPI_ROOT = _resolve_openpi_root()
for path in (REPO_ROOT, OPENPI_ROOT, OPENPI_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.arx_eef_config import DEFAULT_DATASET_ROOT, DEFAULT_REPO_ID, build_config, dataset_home_from_root
from training.arx_eef_policy import ACTION_DIM, EEF_NAMES


def _resolve_checkpoint(exact: str | None, root_arg: str | None, step: int | None, default: Path) -> Path:
    if exact and root_arg:
        raise ValueError("use only one of --checkpoint and --checkpoint-dir")
    root = Path(exact or root_arg or default).expanduser().resolve()
    if (root / "model.safetensors").is_file():
        if step is not None:
            raise ValueError("--step cannot be used with an exact checkpoint directory")
        return root
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint path does not exist: {root}")
    if step is not None:
        checkpoint = root / str(step)
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"missing model.safetensors under {checkpoint}")
        return checkpoint
    steps = sorted(int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"no numeric checkpoints under {root}; pass --checkpoint for an exact directory")
    return root / str(steps[-1])


def _dataset_action_mask(dataset_root: Path) -> list[bool]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing dataset metadata: {info_path}")
    mask = (json.loads(info_path.read_text(encoding="utf-8")).get("arx_eef") or {}).get("default_action_mask")
    if mask is None:
        return [True] * ACTION_DIM
    if not isinstance(mask, list) or len(mask) != ACTION_DIM or not any(mask):
        raise ValueError(f"invalid arx_eef.default_action_mask in {info_path}")
    return [bool(value) for value in mask]


def _set_normalization_source(config, norm_repo_id: str | None, norm_stats_dir: str | None):
    """Override only the stats source; keep the evaluation dataset unchanged."""
    if norm_repo_id and norm_stats_dir:
        raise ValueError("use only one of --norm-repo-id and --norm-stats-dir")
    if not norm_repo_id and not norm_stats_dir:
        return config, None
    factory = config.data
    if not hasattr(factory, "assets"):
        raise RuntimeError("this OpenPI version does not expose DataConfigFactory.assets")
    if norm_stats_dir:
        stats_dir = Path(norm_stats_dir).expanduser().resolve()
        if not stats_dir.is_dir():
            raise FileNotFoundError(f"normalization stats directory does not exist: {stats_dir}")
        assets_dir, asset_id, source = str(stats_dir.parent), stats_dir.name, str(stats_dir)
    else:
        assets_dir, asset_id = None, norm_repo_id
        source = str(Path(config.assets_dirs) / norm_repo_id)
    assets = dataclasses.replace(factory.assets, assets_dir=assets_dir, asset_id=asset_id)
    return dataclasses.replace(config, data=dataclasses.replace(factory, assets=assets)), source


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
        q01, q99 = np.asarray(stats.q01, dtype=np.float32), np.asarray(stats.q99, dtype=np.float32)
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    mean, std = np.asarray(stats.mean, dtype=np.float32), np.asarray(stats.std, dtype=np.float32)
    return actions * (std + 1e-6) + mean


def _move_to_device(tree, device: torch.device):
    import jax

    return jax.tree.map(lambda value: value.to(device) if hasattr(value, "to") else value, tree)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-repo-id", default=None, help="LeRobot repository ID used only for evaluation.")
    parser.add_argument("--eval-dataset-root", default=None, help="Path to the evaluation LeRobot dataset.")
    parser.add_argument("--repo-id", default=None, help="Deprecated alias for --eval-repo-id.")
    parser.add_argument("--dataset-root", default=None, help="Deprecated alias for --eval-dataset-root.")
    parser.add_argument("--assets-base-dir", default=None, help="OpenPI assets root.")
    parser.add_argument("--norm-repo-id", default=None, help="Training repo ID whose normalization assets are used.")
    parser.add_argument("--norm-stats-dir", "--norm-stats", dest="norm_stats_dir", default=None, help="Exact saved normalization-statistics directory.")
    parser.add_argument("--exp-name", default="arx_eef_offline_eval")
    parser.add_argument("--checkpoint", default=None, help="Exact checkpoint directory containing model.safetensors.")
    parser.add_argument("--checkpoint-dir", default=None, help="Experiment checkpoint root; defaults to its latest step.")
    parser.add_argument("--step", type=int, default=None, help="Specific numeric step below --checkpoint-dir.")
    parser.add_argument("--model", choices=["pi0", "pi05"], default="pi05")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--zero-noise", action="store_true", help="Use zero diffusion noise for repeatable sampling.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-npz", default=None, help="Optional predictions/labels output.")
    parser.add_argument("--output-json", default=None, help="Optional machine-readable metrics output.")
    args = parser.parse_args()

    if args.eval_repo_id and args.repo_id and args.eval_repo_id != args.repo_id:
        raise ValueError("--eval-repo-id and --repo-id disagree")
    if args.eval_dataset_root and args.dataset_root and Path(args.eval_dataset_root) != Path(args.dataset_root):
        raise ValueError("--eval-dataset-root and --dataset-root disagree")
    if args.num_samples <= 0 or args.batch_size <= 0 or args.sample_steps <= 0:
        raise ValueError("--num-samples, --batch-size, and --sample-steps must be positive")
    eval_repo_id = args.eval_repo_id or args.repo_id or DEFAULT_REPO_ID
    eval_root = Path(args.eval_dataset_root or args.dataset_root or DEFAULT_DATASET_ROOT).expanduser().resolve()
    if not (eval_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"not a LeRobot dataset root: {eval_root}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["HF_LEROBOT_HOME"] = str(dataset_home_from_root(eval_root, eval_repo_id))
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = build_config(
        repo_id=eval_repo_id, exp_name=args.exp_name, model=args.model, low_mem=False,
        batch_size=args.batch_size, num_workers=args.num_workers, num_train_steps=1,
        assets_base_dir=args.assets_base_dir, pytorch_weight_path=None, wandb_enabled=False,
    )
    config, norm_source = _set_normalization_source(config, args.norm_repo_id, args.norm_stats_dir)
    checkpoint = _resolve_checkpoint(args.checkpoint, args.checkpoint_dir, args.step, config.checkpoint_dir)
    model = _build_model(config, device)
    safetensors.torch.load_model(model, checkpoint / "model.safetensors", device=str(device))

    from openpi.training import data_loader

    loader = data_loader.create_data_loader(
        config, shuffle=False, num_batches=(args.num_samples + args.batch_size - 1) // args.batch_size, framework="pytorch"
    )
    predictions, labels, remaining = [], [], args.num_samples
    with torch.inference_mode():
        for observation, actions in loader:
            take = min(remaining, actions.shape[0])
            observation = _move_to_device(observation, device)
            labels_t = actions[:take].to(device=device, dtype=torch.float32)
            noise = None
            if args.zero_noise:
                noise = torch.zeros(observation.state.shape[0], config.model.action_horizon, config.model.action_dim, dtype=torch.float32, device=device)
            predicted = model.sample_actions(device, observation, noise=noise, num_steps=args.sample_steps)
            predictions.append(predicted[:take, 0, :ACTION_DIM].cpu().numpy())
            labels.append(labels_t[:, 0, :ACTION_DIM].cpu().numpy())
            remaining -= take
            if remaining <= 0:
                break

    pred_norm, label_norm = np.concatenate(predictions), np.concatenate(labels)
    data_config = config.data.create(config.assets_dirs, config.model)
    pred = _unnormalize(pred_norm, data_config.norm_stats, data_config.use_quantile_norm)
    label = _unnormalize(label_norm, data_config.norm_stats, data_config.use_quantile_norm)
    action_mask = np.asarray(_dataset_action_mask(eval_root), dtype=bool)
    rmse = np.sqrt(np.mean(np.square(pred - label), axis=0))
    overall = float(np.sqrt(np.mean(np.square(pred[:, action_mask] - label[:, action_mask]))))
    metrics = {
        "checkpoint": str(checkpoint), "eval_repo_id": eval_repo_id, "eval_dataset_root": str(eval_root),
        "normalization_source": norm_source or f"default: {Path(config.assets_dirs) / eval_repo_id}",
        "samples": int(len(pred)), "sample_steps": args.sample_steps, "zero_noise": args.zero_noise,
        "action_mask": action_mask.astype(int).tolist(), "overall_active_rmse": overall,
        "per_dimension_rmse": {name: float(value) for name, value in zip(EEF_NAMES, rmse, strict=True)},
    }
    print(f"checkpoint: {checkpoint}")
    print(f"eval_dataset: {eval_root} ({eval_repo_id})")
    print(f"normalization: {metrics['normalization_source']}")
    print(f"samples: {len(pred)}")
    print(f"overall_active_rmse: {overall:.6f}")
    for name, value, active in zip(EEF_NAMES, rmse, action_mask, strict=True):
        print(f"{name}: {float(value):.6f}" + ("" if active else " (masked; excluded)"))
    if args.output_npz:
        output = Path(args.output_npz).expanduser(); output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, pred=pred, label=label, pred_normalized=pred_norm, label_normalized=label_norm)
        print(f"wrote predictions: {output}")
    if args.output_json:
        output = Path(args.output_json).expanduser(); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"wrote metrics: {output}")


if __name__ == "__main__":
    main()
