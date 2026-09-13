"""Correction-stage adapter kept independent from alignment/evaluation code."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import shutil
from pathlib import Path
from typing import Any, Callable


def materialize_identity(source: Path, output: Path) -> None:
    """Create an immutable-looking final EEF path without modifying the raw EEF."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, output)


def load_entrypoint(value: str) -> Callable[..., Any]:
    module_name, separator, function_name = value.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(f"correction entrypoint must be 'module:function', got {value!r}")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"correction entrypoint is not callable: {value}")
    return function


def apply_correction(
    spec: dict[str, Any],
    *,
    source_eef: Path,
    output_eef: Path,
    episode_index: int,
    experiment_id: str,
    variant_id: str,
) -> None:
    """Apply one correction variant.

    External correction entrypoints receive keyword arguments. They must write a
    complete ``tate.dual_arm_eef`` JSON to ``output_eef`` and must not modify
    ``source_eef``. Extra correction parameters are passed through ``config``.
    """

    mode = str(spec.get("mode", "none"))
    if mode == "none":
        materialize_identity(source_eef, output_eef)
        return
    output_eef.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_eef.with_name(f".{output_eef.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    if mode in {"real_anchor", "pose_correction"}:
        from correction.apply import run

        options = dict(spec.get("options") or {})
        artifact = spec.get("artifact")
        rotation_artifact = spec.get("rotation_artifact")
        if not artifact and not rotation_artifact:
            raise ValueError("pose correction requires a position or rotation artifact")
        base_seed = int(options.get("seed", 0))
        seed_policy = str(options.get("seed_policy", "per_episode"))
        if seed_policy == "per_episode":
            correction_seed = base_seed
            correction_target_key = f"{experiment_id}|{episode_index}"
        elif seed_policy == "fixed":
            correction_seed = base_seed
            correction_target_key = str(options.get("target_key", "fixed"))
        else:
            raise ValueError("real_anchor seed_policy must be 'per_episode' or 'fixed'")
        run(
            argparse.Namespace(
                input=str(source_eef),
                correction=str(artifact) if artifact else None,
                rotation_correction=rotation_artifact,
                out=str(temporary),
                sides=options.get("sides"),
                target_mode=str(options.get("target_mode", "mean")),
                seed=correction_seed,
                target_key=correction_target_key,
                max_mahalanobis=float(options.get("max_mahalanobis", 0.0)),
                space=str(options.get("space", "free_xyz")),
                propagation=str(options.get("propagation", "linear")),
                max_invalid_gap_s=float(options.get("max_invalid_gap_s", 0.25)),
                endpoint_weight=float(options.get("endpoint_weight", 1.0)),
                anchor_weight=float(options.get("anchor_weight", 1.0)),
                bend_weight=float(options.get("bend_weight", 100.0)),
                magnitude_weight=float(options.get("magnitude_weight", 1e-3)),
                overwrite=False,
            )
        )
        if not temporary.is_file():
            raise RuntimeError("real_anchor correction did not produce its requested output")
        os.replace(temporary, output_eef)
        return
    function = load_entrypoint(str(spec["entrypoint"]))
    kwargs = {
        "source_eef": source_eef,
        "output_eef": temporary,
        "config": dict(spec),
        "episode_index": int(episode_index),
        "experiment_id": experiment_id,
        "variant_id": variant_id,
    }
    accepted = inspect.signature(function).parameters
    if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in accepted.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    result = function(**kwargs)
    if not temporary.is_file() and result is not None:
        candidate = Path(result)
        if candidate.is_file():
            shutil.copy2(candidate, temporary)
    if not temporary.is_file():
        raise RuntimeError(
            f"correction entrypoint {spec['entrypoint']!r} did not write {temporary}"
        )
    os.replace(temporary, output_eef)
