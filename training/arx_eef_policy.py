"""OpenPI transforms for ARX dual-arm EEF actions."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


ACTION_DIM = 16
MODEL_ACTION_DIM = 32
EEF_NAMES = (
    "left_eef_x",
    "left_eef_y",
    "left_eef_z",
    "left_eef_qx",
    "left_eef_qy",
    "left_eef_qz",
    "left_eef_qw",
    "left_gripper",
    "right_eef_x",
    "right_eef_y",
    "right_eef_z",
    "right_eef_qx",
    "right_eef_qy",
    "right_eef_qz",
    "right_eef_qw",
    "right_gripper",
)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(255.0 * image, 0.0, 255.0).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected an RGB image, got shape {image.shape}")
    return image


def _validate_vector(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] != ACTION_DIM:
        raise ValueError(f"ARX EEF {name} must be {ACTION_DIM}D, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"ARX EEF {name} contains non-finite values")
    return value


@dataclasses.dataclass(frozen=True)
class ArxEefInputs(transforms.DataTransformFn):
    """Map ARX EEF LeRobot samples to OpenPI's three-camera observation format."""

    model_type: _model.ModelType
    mask_wrist_images: bool = False

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        expected = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        missing = [key for key in expected if key not in images]
        if missing:
            raise ValueError(f"ARX images missing keys: {missing}")

        head = _parse_image(images["cam_high"])
        left = _parse_image(images["cam_left_wrist"])
        right = _parse_image(images["cam_right_wrist"])
        state = _validate_vector(data["state"], "state")

        if self.model_type == _model.ModelType.PI0_FAST:
            image = {
                "base_0_rgb": head,
                "base_1_rgb": left,
                "wrist_0_rgb": right,
            }
            image_mask = {
                "base_0_rgb": np.True_,
                "base_1_rgb": np.False_ if self.mask_wrist_images else np.True_,
                "wrist_0_rgb": np.False_ if self.mask_wrist_images else np.True_,
            }
        else:
            image = {
                "base_0_rgb": head,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            }
            image_mask = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_ if self.mask_wrist_images else np.True_,
                "right_wrist_0_rgb": np.False_ if self.mask_wrist_images else np.True_,
            }

        inputs = {
            "image": image,
            "image_mask": image_mask,
            "state": state,
        }
        if "actions" in data:
            inputs["actions"] = _validate_vector(data["actions"], "actions")
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class ArxEefOutputs(transforms.DataTransformFn):
    """Return only the 16 ARX EEF dimensions from a model action tensor."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"model actions must have at least {ACTION_DIM} dimensions, got {actions.shape}")
        return {"actions": actions[..., :ACTION_DIM]}


@dataclasses.dataclass(frozen=True)
class CastFloat32(transforms.DataTransformFn):
    """Keep normalized state/action tensors compatible with PyTorch model layers."""

    def __call__(self, data: dict) -> dict:
        for key in ("state", "actions"):
            if key in data:
                data[key] = np.asarray(data[key], dtype=np.float32)
        return data
