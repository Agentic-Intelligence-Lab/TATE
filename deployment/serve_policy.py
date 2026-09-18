#!/usr/bin/env python3
"""Load the trained OpenPI policy and serve local ARX EEF inference over HTTP."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image


APP_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = APP_ROOT / "thirdparty" / "openpi"
for path in (APP_ROOT, OPENPI_ROOT, OPENPI_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deployment.constants import (  # noqa: E402
    ACTION_HORIZON,
    ASSET_ID,
    CAMERA_SERIALS,
    CHECKPOINT_DIR,
    PROMPT,
    norm_stats_path,
    verify_checkpoint_files,
)


def load_policy(device: str, sample_steps: int):
    import torch
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from training.arx_eef_config import build_config

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    verify_checkpoint_files()
    config = build_config(
        repo_id=ASSET_ID,
        exp_name="stack_cube_cotrain_camdrop_pi05_bs16",
        model="pi05",
        low_mem=False,
        batch_size=1,
        num_train_steps=1,
        num_workers=0,
        wandb_enabled=False,
    )
    # The training default compiles sample_actions with max-autotune. That can
    # spend several minutes compiling before the first ARX inference request.
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, pytorch_compile_mode=None),
    )
    return policy_config.create_trained_policy(
        config,
        CHECKPOINT_DIR,
        default_prompt=PROMPT,
        norm_stats=normalize.load(norm_stats_path(CHECKPOINT_DIR).parent),
        sample_kwargs={"num_steps": sample_steps},
        pytorch_device=device,
    )


def decode_observation(payload: dict) -> dict:
    state = np.asarray(payload.get("state"), dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("state must be 16 finite floats")
    image_mask = np.asarray(payload.get("image_mask"), dtype=bool)
    if image_mask.shape != (3,) or not image_mask[0]:
        raise ValueError("image_mask must be [head, left, right] with head available")
    images = payload.get("images")
    if not isinstance(images, dict):
        raise ValueError("images must contain head, left and right JPEG data")
    decoded = {}
    for role in CAMERA_SERIALS:
        encoded = images.get(role)
        if not isinstance(encoded, str):
            raise ValueError(f"missing {role} JPEG")
        data = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(data)) as frame:
            decoded[{"head": "cam_high", "left": "cam_left_wrist", "right": "cam_right_wrist"}[role]] = (
                np.asarray(frame.convert("RGB"), dtype=np.uint8)
            )
    prompt = payload.get("prompt", PROMPT)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    return {"images": decoded, "state": state, "image_mask": image_mask, "prompt": prompt}


def infer(policy, payload: dict) -> dict:
    observation = decode_observation(payload)
    result = policy.infer(observation)
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, 16) or not np.isfinite(actions).all():
        raise RuntimeError(f"policy returned invalid actions: {actions.shape}")
    return {
        "actions": actions.tolist(),
        "infer_ms": float(result.get("policy_timing", {}).get("infer_ms", 0.0)),
    }


def smoke_result(policy) -> dict:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    state = np.asarray([0, 0, 0, 0, 0, 0, 1, 0, 0.25, 0.13, -0.08, 0, 0, 0, 1, 0.7], dtype=np.float32)
    observation = {
        "images": {"cam_high": blank, "cam_left_wrist": blank, "cam_right_wrist": blank},
        "state": state,
        "image_mask": np.asarray([True, True, True]),
        "prompt": PROMPT,
    }
    result = policy.infer(observation)
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, 16) or not np.isfinite(actions).all():
        raise RuntimeError(f"smoke inference returned invalid actions: {actions.shape}")
    return {"shape": list(actions.shape), "infer_ms": float(result["policy_timing"]["infer_ms"])}


def smoke_inference(policy) -> None:
    result = smoke_result(policy)
    print(f"SMOKE_INFERENCE_OK shape={tuple(result['shape'])} infer_ms={result['infer_ms']:.1f}", flush=True)


def serve(policy, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, body: dict) -> None:
            encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/infer", "/smoke"}:
                self._json(404, {"error": "not found"})
                return
            size = int(self.headers.get("Content-Length", "0"))
            if self.path == "/infer" and (size < 1 or size > 4_000_000):
                self._json(413, {"error": "invalid observation size"})
                return
            try:
                if self.path == "/smoke":
                    self._json(200, smoke_result(policy))
                else:
                    payload = json.loads(self.rfile.read(size))
                    self._json(200, infer(policy, payload))
            except Exception as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, format: str, *args) -> None:
            return

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"POLICY_READY http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("load-only", "smoke", "serve"), default="serve")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--port", type=int, default=8019)
    args = parser.parse_args()
    if not 1 <= args.sample_steps <= 20:
        parser.error("--sample-steps must be in [1, 20]")
    policy = load_policy(args.device, args.sample_steps)
    print(f"LOAD_ONLY_OK checkpoint={CHECKPOINT_DIR} device={args.device}", flush=True)
    if args.mode == "smoke":
        smoke_inference(policy)
    elif args.mode == "serve":
        serve(policy, args.port)


if __name__ == "__main__":
    main()
