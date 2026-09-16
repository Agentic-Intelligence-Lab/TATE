"""Paths and checkpoint contract shared by the ARX deployment programs."""

from __future__ import annotations

import json
import os
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get(
        "TATE_CHECKPOINT_DIR",
        "/home/qijun/models/TATE/openpi_checkpoints/arx_eef/"
        "stack_cube_cotrain_camdrop_pi05_bs16/2857",
    )
).expanduser()
CHECKPOINT_DIR = DEFAULT_CHECKPOINT_DIR
MODEL_NAME = "model.safetensors"
ASSET_ID = "local/arx_eef_stack_cube_cotrain_camdrop_train"
TOKENIZER = Path(
    os.environ.get(
        "TATE_TOKENIZER_PATH",
        "/home/qijun/.cache/openpi/big_vision/paligemma_tokenizer.model",
    )
).expanduser()
PROMPT = "pick the cube and stack it on the blue plate"
ACTION_HORIZON = 10
CAMERA_SERIALS = {
    "head": "409122273248",
    "left": "260322272716",
    "right": "409122274457",
}
# Keep this synchronized with left_tcp/right_tcp in the MuJoCo scene.  SDK
# poses are flange poses, so deployment explicitly maps between flange and TCP.
TCP_OFFSET_M = (0.15, 0.0, 0.0)
PREVIEW_DIR = Path("/tmp/tate_arx_camera_policy")


def norm_stats_path(checkpoint_dir: Path) -> Path:
    flat = checkpoint_dir / "norm_stats.json"
    if flat.is_file():
        return flat
    return checkpoint_dir / "assets" / ASSET_ID / "norm_stats.json"


def safetensors_complete(model: Path) -> bool:
    """Check the header's tensor offsets against the file length without reading weights."""
    try:
        size = model.stat().st_size
        with model.open("rb") as stream:
            header_size = int.from_bytes(stream.read(8), "little")
            if header_size < 2 or header_size > min(16_000_000, size - 8):
                return False
            header = json.loads(stream.read(header_size))
        offsets = [value["data_offsets"] for key, value in header.items() if key != "__metadata__"]
        return bool(offsets) and size == 8 + header_size + max(end for _, end in offsets)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def checkpoint_status(checkpoint_dir: Path = CHECKPOINT_DIR) -> dict[str, object]:
    model = checkpoint_dir / MODEL_NAME
    size = model.stat().st_size if model.is_file() else None
    model_complete = model.is_file() and safetensors_complete(model)
    return {
        "checkpoint": str(checkpoint_dir),
        "model_present": model.is_file(),
        "model_bytes": size,
        "model_complete": model_complete,
        "norm_stats_present": norm_stats_path(checkpoint_dir).is_file(),
        "tokenizer_present": TOKENIZER.is_file(),
        "ready": model_complete and norm_stats_path(checkpoint_dir).is_file() and TOKENIZER.is_file(),
    }


def verify_checkpoint_files(checkpoint_dir: Path = CHECKPOINT_DIR) -> None:
    model = checkpoint_dir / MODEL_NAME
    if not safetensors_complete(model):
        raise RuntimeError(f"model missing or incomplete: {model}")
    norm_stats = norm_stats_path(checkpoint_dir)
    if not norm_stats.is_file():
        raise FileNotFoundError(f"missing normalization stats: {norm_stats}")
    if not TOKENIZER.is_file():
        raise FileNotFoundError(f"missing PaliGemma tokenizer: {TOKENIZER}")
