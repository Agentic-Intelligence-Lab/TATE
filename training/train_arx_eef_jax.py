#!/usr/bin/env python3
"""JAX/Flax training entry point for ARX dual-arm EEF Pi0/Pi0.5 models.

This is deliberately a separate entry point from ``train_arx_eef_pytorch.py``:
OpenPI's JAX trainer consumes its native Orbax/JAX base checkpoint, whereas the
PyTorch trainer consumes ``model.safetensors``.  A PyTorch checkpoint therefore
cannot be supplied to this script.
"""

# OpenPI and this repository are added to sys.path before project imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_openpi_root() -> Path:
    """Find an OpenPI checkout which contains the native JAX trainer."""
    candidates: list[Path] = []
    configured = os.environ.get("TATE_OPENPI_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend((REPO_ROOT / "thirdparty" / "openpi", Path.cwd()))
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "scripts" / "train.py").is_file() and (candidate / "src" / "openpi").is_dir():
            return candidate
    looked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find an OpenPI checkout containing scripts/train.py. "
        f"Looked in: {looked}. Set TATE_OPENPI_ROOT explicitly."
    )


OPENPI_ROOT = _resolve_openpi_root()
for path in (REPO_ROOT, OPENPI_ROOT, OPENPI_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.arx_eef_config import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_REPO_ID,
    DEFAULT_TRAIN_EPOCHS,
    build_config,
    dataset_home_from_root,
    train_steps_for_dataset,
)
from training.arx_eef_policy import ACTION_DIM, MODEL_ACTION_DIM


def dataset_action_mask(dataset_root: str | Path) -> list[bool]:
    """Read the 16-D loss mask written by the ARX cotrain exporter."""
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


def _patch_jax_action_loss(action_mask: list[bool], loss_action_dim: int) -> None:
    """Make OpenPI JAX loss respect ARX's active action dimensions.

    OpenPI's stock Pi0 ``compute_loss`` averages over all padded model action
    dimensions.  ARX labels occupy only the first 16 of the 32 model dimensions;
    for single-arm cube data, some of those 16 are intentionally unsupervised as
    well.  Replace that final reduction with a mean over active ARX dimensions.
    """
    if not 0 < loss_action_dim <= ACTION_DIM:
        raise ValueError(f"loss_action_dim must be in [1, {ACTION_DIM}]")

    import jax
    import jax.numpy as jnp
    from openpi.models import pi0 as pi0_model

    original_compute_loss = pi0_model.Pi0.compute_loss
    if getattr(original_compute_loss, "_arx_eef_loss_patched", False):
        return

    full_mask = jnp.asarray(
        [*action_mask[:loss_action_dim], *([False] * (MODEL_ACTION_DIM - loss_action_dim))], dtype=jnp.float32
    )
    active_count = int(sum(action_mask[:loss_action_dim]))
    if active_count == 0:
        raise ValueError("action mask has no active dimensions")

    def compute_loss_with_mask(self, rng, observation, actions, *, train: bool = False):
        # This is OpenPI Pi0.compute_loss with only its last reduction changed.
        # Keep it here (rather than masking labels) so zero-valued inactive
        # labels cannot contribute a gradient.
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = pi0_model._model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0_model.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        squared_error = jnp.square(v_t - u_t)
        return jnp.sum(squared_error * full_mask, axis=-1) / active_count

    compute_loss_with_mask._arx_eef_loss_patched = True
    pi0_model.Pi0.compute_loss = compute_loss_with_mask


def _load_openpi_jax_trainer():
    """Load OpenPI's JAX trainer by path, avoiding ``scripts`` name clashes."""
    module_path = OPENPI_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("_tate_openpi_train_jax", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load OpenPI JAX trainer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "main", None)):
        raise ImportError(f"OpenPI JAX trainer has no main(config): {module_path}")
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--assets-base-dir", default=None)
    parser.add_argument("--checkpoint-base-dir", default=None)
    parser.add_argument("--exp-name", default="arx_eef_pi05_jax")
    parser.add_argument("--model", choices=["pi0", "pi05"], default="pi05")
    parser.add_argument("--batch-size", type=int, default=8, help="Global batch size across all JAX devices.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--state-dropout-probability", type=float, default=0.0)
    parser.add_argument("--loss-action-dim", type=int, default=ACTION_DIM)
    parser.add_argument("--action-mask", nargs=ACTION_DIM, type=int, default=None, metavar="MASK")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers must be non-negative")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.num_train_steps is not None and args.num_train_steps <= 0:
        raise ValueError("--num-train-steps must be positive")
    if args.fsdp_devices <= 0:
        raise ValueError("--fsdp-devices must be positive")
    if not 0.0 <= args.state_dropout_probability <= 1.0:
        raise ValueError("--state-dropout-probability must be in [0, 1]")
    if not 0 < args.loss_action_dim <= ACTION_DIM:
        raise ValueError(f"--loss-action-dim must be in [1, {ACTION_DIM}]")
    if args.action_mask is not None and any(value not in (0, 1) for value in args.action_mask):
        raise ValueError("--action-mask values must be 0 or 1")

    num_train_steps = args.num_train_steps
    if num_train_steps is None:
        num_train_steps = train_steps_for_dataset(args.dataset_root, args.batch_size, epochs=args.epochs)

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
        # JAX loads config.weight_loader's native checkpoint, never safetensors.
        pytorch_weight_path=None,
        wandb_enabled=args.wandb,
        overwrite=args.overwrite,
        resume=args.resume,
        state_dropout_probability=args.state_dropout_probability,
    )
    config = dataclasses.replace(config, fsdp_devices=args.fsdp_devices, seed=args.seed)

    action_mask = (
        [bool(value) for value in args.action_mask]
        if args.action_mask is not None
        else dataset_action_mask(args.dataset_root)
    )
    if args.loss_action_dim != ACTION_DIM and not all(action_mask[args.loss_action_dim:]):
        raise ValueError("--loss-action-dim cannot hide masked dimensions; use the 16-D default")
    print(f"ARX EEF action loss mask: {[int(value) for value in action_mask]}")
    print(f"ARX EEF training state dropout probability: {args.state_dropout_probability:.3f}")
    print("Using OpenPI native JAX checkpoint loader (not PyTorch model.safetensors).")
    _patch_jax_action_loss(action_mask, args.loss_action_dim)
    _load_openpi_jax_trainer().main(config)


if __name__ == "__main__":
    main()
