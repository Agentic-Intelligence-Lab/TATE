"""OpenPI training configuration for ARX dual-arm joint datasets."""

# OpenPI is vendored in this repository and is added to sys.path below.
# ruff: noqa: E402

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = REPO_ROOT / "thirdparty" / "openpi"
OPENPI_SRC = OPENPI_ROOT / "src"
for path in (REPO_ROOT, OPENPI_ROOT, OPENPI_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import flax.nnx as nnx
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.config as openpi_config
import openpi.training.optimizer as openpi_optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as transforms

from training import arx_joint_policy


DEFAULT_TASK_PROMPT = "fold the paper boxes."
DEFAULT_REPO_ID = "local/arx_stack_cube_ego"
DEFAULT_DATASET_ROOT = REPO_ROOT / "outputs" / "lerobot" / DEFAULT_REPO_ID
DEFAULT_BATCH_SIZE = 8
DEFAULT_TRAIN_EPOCHS = 2
MODEL_ACTION_DIM = arx_joint_policy.MODEL_ACTION_DIM


def train_steps_for_epochs(batch_size: int, *, total_frames: int, epochs: int = DEFAULT_TRAIN_EPOCHS) -> int:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    return (total_frames // batch_size) * epochs


def dataset_total_frames(dataset_root: str | Path) -> int:
    info_path = Path(dataset_root).expanduser() / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing dataset metadata: {info_path}")
    total = int(json.loads(info_path.read_text(encoding="utf-8"))["total_frames"])
    if total <= 0:
        raise ValueError(f"invalid total_frames in {info_path}: {total}")
    return total


def train_steps_for_dataset(
    dataset_root: str | Path,
    batch_size: int,
    *,
    epochs: int = DEFAULT_TRAIN_EPOCHS,
) -> int:
    return train_steps_for_epochs(
        batch_size,
        total_frames=dataset_total_frames(dataset_root),
        epochs=epochs,
    )


@dataclasses.dataclass(frozen=True)
class ArxJointDataConfig(openpi_config.DataConfigFactory):
    """LeRobot DataConfig for three-camera ARX joint training."""

    default_prompt: str | None = DEFAULT_TASK_PROMPT

    @override
    def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> openpi_config.DataConfig:
        repack_transform = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.head",
                            "cam_left_wrist": "observation.images.left",
                            "cam_right_wrist": "observation.images.right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[arx_joint_policy.ArxJointInputs(model_type=model_config.model_type)],
            outputs=[arx_joint_policy.ArxJointOutputs()],
        )
        model_transforms = openpi_config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        model_transforms = transforms.Group(
            inputs=[arx_joint_policy.CastFloat32(), *model_transforms.inputs],
            outputs=model_transforms.outputs,
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


def build_config(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    exp_name: str = "arx_joint_debug",
    model: str = "pi05",
    low_mem: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_train_steps: int = 10,
    save_interval: int = 1000,
    log_interval: int = 100,
    num_workers: int = 2,
    assets_base_dir: str | None = None,
    checkpoint_base_dir: str | None = None,
    pytorch_weight_path: str | None = None,
    wandb_enabled: bool = False,
    overwrite: bool = False,
    resume: bool = False,
) -> openpi_config.TrainConfig:
    if model == "pi0":
        model_config = pi0_config.Pi0Config(
            action_dim=MODEL_ACTION_DIM,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora" if low_mem else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if low_mem else "gemma_300m",
        )
        checkpoint = "gs://openpi-assets/checkpoints/pi0_base/params"
    elif model == "pi05":
        model_config = pi0_config.Pi0Config(
            pi05=True,
            action_dim=MODEL_ACTION_DIM,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora" if low_mem else "gemma_2b",
            action_expert_variant="gemma_300m_lora" if low_mem else "gemma_300m",
        )
        checkpoint = "gs://openpi-assets/checkpoints/pi05_base/params"
    else:
        raise ValueError(f"unsupported model: {model}")

    freeze_filter = model_config.get_freeze_filter() if low_mem else nnx.Nothing
    return openpi_config.TrainConfig(
        name="arx_joint",
        project_name="lifego",
        exp_name=exp_name,
        model=model_config,
        data=ArxJointDataConfig(
            repo_id=repo_id,
            base_config=openpi_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(checkpoint),
        pytorch_weight_path=pytorch_weight_path,
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        save_interval=save_interval,
        log_interval=log_interval,
        assets_base_dir=str(REPO_ROOT / "outputs" / "openpi_assets") if assets_base_dir is None else assets_base_dir,
        checkpoint_base_dir=str(REPO_ROOT / "outputs" / "openpi_checkpoints")
        if checkpoint_base_dir is None
        else checkpoint_base_dir,
        lr_schedule=openpi_optimizer.CosineDecaySchedule(
            warmup_steps=min(1_000, max(num_train_steps // 10, 1)),
            peak_lr=5e-5,
            decay_steps=max(num_train_steps, 1),
            decay_lr=5e-5,
        ),
        optimizer=openpi_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=freeze_filter,
        ema_decay=None if low_mem else 0.99,
        wandb_enabled=wandb_enabled,
        overwrite=overwrite,
        resume=resume,
    )


def dataset_home_from_root(dataset_root: str | Path, repo_id: str = DEFAULT_REPO_ID) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    repo_parts = Path(repo_id).parts
    if len(root.parts) >= len(repo_parts) and root.parts[-len(repo_parts) :] == repo_parts:
        return Path(*root.parts[: -len(repo_parts)])
    if (root / repo_id / "meta" / "info.json").is_file():
        return root
    return root.parent.parent if len(root.parts) >= 2 else root
