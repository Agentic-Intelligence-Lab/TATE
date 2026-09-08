#!/usr/bin/env python3
"""Update TATE's RealSense YAML from one arm-specific calibration result.

The calibration result must contain ``metadata.intrinsics`` and
``T_cam_in_base.T`` in the format written by
``calibrate_realsense_extrinsics.py``.  Here ``base`` means the selected ARX
arm base, so left and right results are intentionally stored separately.

This updater edits only the numeric calibration blocks and keeps the comments
and the rest of the YAML file intact.  The legacy/global ``extrinsics.c2w`` is
not overwritten by an arm-specific solve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "cfg" / "preprocess" / "base" / "RealSenseD405.yaml"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _find_key(lines: list[str], key: str, indent: int, start: int = 0, end: int | None = None) -> int:
    end = len(lines) if end is None else end
    prefix = " " * indent + key + ":"
    for i in range(start, end):
        if lines[i].startswith(prefix) and lines[i][len(prefix) :].strip() in ("",):
            return i
    raise ValueError(f"YAML key not found: {' ' * indent}{key}:")


def _mapping_end(lines: list[str], key_index: int, indent: int) -> int:
    for i in range(key_index + 1, len(lines)):
        stripped = lines[i].strip()
        # A same-level comment introduces the next sibling just as a key does;
        # it must not be swallowed when replacing the current mapping.
        if stripped and _indent(lines[i]) <= indent:
            return i
    return len(lines)


def _replace_scalar(lines: list[str], key: str, indent: int, value: Any, start: int, end: int) -> None:
    prefix = " " * indent + key + ":"
    for i in range(start, end):
        if lines[i].startswith(prefix):
            suffix = lines[i][len(prefix) :]
            comment = ""
            if "#" in suffix:
                comment = "  #" + suffix.split("#", 1)[1].rstrip("\n")
            lines[i] = f"{prefix} {json.dumps(value)}{comment}\n"
            return
    raise ValueError(f"YAML scalar not found: {key}")


def _replace_sequence_block(
    lines: list[str], key: str, indent: int, rows: list[Any], start: int, end: int
) -> tuple[int, int]:
    key_index = _find_key(lines, key, indent, start, end)
    block_end = key_index + 1
    while block_end < len(lines):
        stripped = lines[block_end].strip()
        # A same-level comment belongs to the next YAML field, not to this
        # sequence. Keeping it is important for the intrinsics/distortion notes.
        if stripped and _indent(lines[block_end]) <= indent:
            break
        block_end += 1
    replacement = [lines[key_index]] + [f"{' ' * (indent + 2)}- {json.dumps(row)}\n" for row in rows]
    lines[key_index:block_end] = replacement
    return key_index, key_index + len(replacement)


def _validate_result(result: dict[str, Any], arm: str) -> tuple[dict[str, Any], list[list[float]]]:
    metadata = result.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("intrinsics"), dict):
        raise ValueError("calibration result is missing metadata.intrinsics")
    intrinsics = metadata["intrinsics"]
    required = ("fx", "fy", "cx", "cy", "dist_coeffs", "width", "height")
    missing = [key for key in required if intrinsics.get(key) is None]
    if missing:
        raise ValueError(f"calibration result intrinsics are missing: {missing}")

    record = result.get("T_cam_in_base")
    matrix = record.get("T") if isinstance(record, dict) else None
    if not isinstance(matrix, list) or len(matrix) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in matrix):
        raise ValueError("calibration result is missing a 4x4 T_cam_in_base.T")
    matrix = [[float(value) for value in row] for row in matrix]
    if any(abs(matrix[3][i] - expected) > 1e-9 for i, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ValueError("T_cam_in_base.T has an invalid homogeneous last row")

    result_arm = metadata.get("arm")
    if result_arm is not None and result_arm != arm:
        raise ValueError(f"result metadata.arm={result_arm!r} does not match --arm {arm!r}")
    return intrinsics, matrix


def update_yaml_from_result(
    yaml_path: str | Path,
    result_path: str | Path,
    arm: str,
    *,
    dry_run: bool = False,
) -> str:
    """Apply one calibration result and return the updated YAML text."""
    if arm not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")

    yaml_path = Path(yaml_path).expanduser().resolve()
    result_path = Path(result_path).expanduser().resolve()
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes.decode("utf-8"))
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    intrinsics, matrix = _validate_result(result, arm)
    lines = yaml_path.read_text(encoding="utf-8").splitlines(keepends=True)

    resolution_i = _find_key(lines, "resolution", 0)
    resolution_end = _mapping_end(lines, resolution_i, 0)
    _replace_scalar(lines, "width", 2, int(intrinsics["width"]), resolution_i + 1, resolution_end)
    _replace_scalar(lines, "height", 2, int(intrinsics["height"]), resolution_i + 1, resolution_end)

    intrinsics_i = _find_key(lines, "intrinsics", 0)
    intrinsics_end = _mapping_end(lines, intrinsics_i, 0)
    K = [
        [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
        [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
        [0.0, 0.0, 1.0],
    ]
    _replace_sequence_block(lines, "K", 2, K, intrinsics_i + 1, intrinsics_end)
    # Re-find the section end because replacing K changes line indices.
    intrinsics_i = _find_key(lines, "intrinsics", 0)
    intrinsics_end = _mapping_end(lines, intrinsics_i, 0)
    _replace_scalar(
        lines,
        "d",
        2,
        [float(value) for value in intrinsics["dist_coeffs"]],
        intrinsics_i + 1,
        intrinsics_end,
    )

    arms_i = _find_key(lines, "arm_extrinsics", 0)
    arms_end = _mapping_end(lines, arms_i, 0)
    arm_i = _find_key(lines, arm, 2, arms_i + 1, arms_end)
    arm_end = _mapping_end(lines, arm_i, 2)
    matrix_key = f"T_cam_in_{arm}_arm_base"
    matrix_i, matrix_end = _replace_sequence_block(lines, matrix_key, 4, matrix, arm_i + 1, arm_end)

    metadata = result["metadata"]
    provenance = [
        "    calibration:\n",
        f"      result_json: {json.dumps(result_path.name)}\n",
        f"      result_sha256: {json.dumps(result_sha256)}\n",
        f"      tag_corners_base_source: {json.dumps(metadata.get('tag_corners_base_source'))}\n",
        f"      tag_family: {json.dumps(metadata.get('tag_family'))}\n",
        f"      tags_used: {json.dumps(metadata.get('tags_used', []))}\n",
        f"      reprojection_rmse_px: {json.dumps(metadata.get('reprojection_rmse_px'))}\n",
    ]
    # Replace an existing per-arm provenance block, otherwise insert after the matrix.
    arm_i = _find_key(lines, arm, 2, arms_i + 1, len(lines))
    arm_end = _mapping_end(lines, arm_i, 2)
    try:
        calibration_i = _find_key(lines, "calibration", 4, arm_i + 1, arm_end)
    except ValueError:
        lines[matrix_end:matrix_end] = provenance
    else:
        calibration_end = _mapping_end(lines, calibration_i, 4)
        while calibration_end > calibration_i + 1 and not lines[calibration_end - 1].strip():
            calibration_end -= 1
        lines[calibration_i:calibration_end] = provenance

    updated = "".join(lines)
    if not dry_run:
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{yaml_path.name}.", dir=yaml_path.parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(updated)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, yaml_path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("left", "right"))
    parser.add_argument("--result", required=True, help="camera_extrinsics JSON produced for this arm")
    parser.add_argument("--yaml", default=str(DEFAULT_YAML), help="TATE camera calibration YAML")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the updated YAML without writing it")
    args = parser.parse_args()
    updated = update_yaml_from_result(args.yaml, args.result, args.arm, dry_run=args.dry_run)
    if args.dry_run:
        print(updated, end="")
    else:
        print(f"Updated {Path(args.yaml).resolve()} for {args.arm} arm from {Path(args.result).resolve()}")


if __name__ == "__main__":
    main()
