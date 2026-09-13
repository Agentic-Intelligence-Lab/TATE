"""Visualize a WiLoR cache, hand2gripper targets, and grasp hysteresis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from preprocess.Hand2Gripper import FingerCenter, MODES, debounce_grasp, make_hand2gripper
from preprocess.PipelineIO import load_yaml


HAND_CHAINS = (
    (5, 6, 7, 0),
    (5, 8, 9, 10, 1),
    (5, 11, 12, 13, 2),
    (5, 14, 15, 16, 3),
    (5, 17, 18, 19, 4),
)
CHAIN_COLORS = (
    (160, 160, 255),
    (160, 255, 160),
    (255, 210, 160),
    (150, 250, 250),
    (250, 160, 250),
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _draw_pose_axis(
    image: np.ndarray,
    pose_value: Any,
    K: np.ndarray,
    d: np.ndarray,
    label: str,
    axis_length: float,
) -> None:
    if pose_value is None:
        return
    pose = np.asarray(pose_value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float64,
    )
    camera_points = (pose[:3, :3] @ points.T).T + pose[:3, 3]
    if np.any(camera_points[:, 2] <= 0.0):
        return
    projected, _ = cv2.projectPoints(
        points,
        cv2.Rodrigues(pose[:3, :3])[0],
        pose[:3, 3],
        K,
        d,
    )
    pixels = np.rint(projected.reshape(-1, 2)).astype(int)
    origin = tuple(pixels[0])
    for endpoint, color in zip(pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.arrowedLine(image, origin, tuple(endpoint), color, 2, cv2.LINE_AA, tipLength=0.2)
    cv2.putText(
        image,
        label,
        (origin[0] + 5, origin[1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_skeleton(image: np.ndarray, hand: dict[str, Any], grasp_state: int) -> None:
    value = hand.get("keypoints_2d")
    if value is None:
        return
    points = np.asarray(value, dtype=np.float64)
    if points.shape != (21, 2) or not np.all(np.isfinite(points)):
        return
    points = np.rint(points).astype(int)
    closed = int(grasp_state) == 1
    for chain, base_color in zip(HAND_CHAINS, CHAIN_COLORS):
        color = (0, 0, 255) if closed else base_color
        for first, second in zip(chain[:-1], chain[1:]):
            cv2.line(
                image,
                tuple(points[first]),
                tuple(points[second]),
                color,
                2,
                cv2.LINE_AA,
            )
        for index in chain[1:]:
            cv2.circle(image, tuple(points[index]), 3, color, -1, cv2.LINE_AA)
    cv2.circle(image, tuple(points[5]), 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image, tuple(points[20]), 4, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.line(
        image,
        tuple(points[0]),
        tuple(points[1]),
        (0, 0, 255) if closed else (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _grasp_values(
    hand: dict[str, Any], grasp_mode: str
) -> tuple[float | None, float, float]:
    value = hand.get("keypoints_3d_cam")
    if value is None:
        return None, float("nan"), float("nan")
    keypoints = np.asarray(value, dtype=np.float64)
    if keypoints.shape != (21, 3):
        return None, float("nan"), float("nan")
    comparison_tip = keypoints[1]
    if grasp_mode == "qwen":
        comparison_tip = 0.7 * keypoints[1] + 0.3 * keypoints[2]
    tip_distance = float(np.linalg.norm(keypoints[0] - comparison_tip))
    palm_size = float(np.linalg.norm(keypoints[11] - keypoints[5]))
    ratio = tip_distance / palm_size if palm_size > 0.01 else None
    return ratio, tip_distance, palm_size


def _draw_grasp_panel(
    image: np.ndarray,
    side: str,
    hand: dict[str, Any],
    state: int,
    ratio: float | None,
    close_ratio: float,
    open_ratio: float,
    ratio_plot_max: float,
    grasp_mode: str,
) -> None:
    _, tip_distance, palm_size = _grasp_values(hand, grasp_mode)
    closed = int(state) == 1
    if ratio is None:
        band = "raw-distance fallback"
        ratio_text = "N/A"
    elif ratio < close_ratio:
        band = "below CLOSE threshold"
        ratio_text = f"{ratio:.3f}"
    elif ratio > open_ratio:
        band = "above OPEN threshold"
        ratio_text = f"{ratio:.3f}"
    else:
        band = "HYSTERESIS HOLD"
        ratio_text = f"{ratio:.3f}"

    panel_width = min(365, image.shape[1] - 20)
    x = 10 if side == "right" else image.shape[1] - panel_width - 10
    y, height = 12, 154
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, image)
    state_color = (0, 70, 255) if closed else (0, 210, 255)
    lines = (
        f"{side.upper()}  {'CLOSED' if closed else 'OPEN'}",
        f"grasp_ratio ({grasp_mode}) = tip / palm = {ratio_text}",
        f"tip={tip_distance:.4f} m   palm={palm_size:.4f} m",
        f"close < {close_ratio:.3f}   open > {open_ratio:.3f}",
        band,
    )
    for line_index, line in enumerate(lines):
        color = state_color if line_index in (0, 4) else (235, 235, 235)
        cv2.putText(
            image,
            line,
            (x + 10, y + 24 + 23 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    bar_x0, bar_x1, bar_y = x + 10, x + panel_width - 10, y + height - 15
    cv2.line(image, (bar_x0, bar_y), (bar_x1, bar_y), (130, 130, 130), 3)
    scale = max(float(ratio_plot_max), open_ratio, close_ratio, 1e-6)
    to_x = lambda value: int(round(bar_x0 + np.clip(value / scale, 0.0, 1.0) * (bar_x1 - bar_x0)))
    cv2.line(image, (to_x(close_ratio), bar_y - 7), (to_x(close_ratio), bar_y + 7), (0, 0, 255), 2)
    cv2.line(image, (to_x(open_ratio), bar_y - 7), (to_x(open_ratio), bar_y + 7), (0, 255, 0), 2)
    if ratio is not None:
        cv2.circle(image, (to_x(ratio), bar_y), 5, state_color, -1, cv2.LINE_AA)


def _build_targets(
    frames: list[dict[str, Any]],
    mode: str,
    grasp_mode: str,
    close_ratio: float,
    open_ratio: float,
    min_frames: int,
) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for side, hand_key in (("right", "hand_r"), ("left", "hand_l")):
        pose_converter = make_hand2gripper(
            mode,
            grasp_close_ratio=close_ratio,
            grasp_open_ratio=open_ratio,
            grasp_min_frames=min_frames,
        )
        grasp_converter = (
            pose_converter
            if grasp_mode == mode
            else make_hand2gripper(
                grasp_mode,
                grasp_close_ratio=close_ratio,
                grasp_open_ratio=open_ratio,
                grasp_min_frames=min_frames,
            )
        )
        targets = []
        for frame in frames:
            hand = frame.get(hand_key)
            if hand is None:
                targets.append(None)
                continue
            target = pose_converter.from_hand_record(hand, is_right=side == "right")
            grasp_target = (
                target
                if grasp_converter is pose_converter
                else grasp_converter.from_hand_record(hand, is_right=side == "right")
            )
            if target is not None and grasp_target is not None:
                target.grasp_state = grasp_target.grasp_state
                target.grasp_ratio = grasp_target.grasp_ratio
            targets.append(target)
        states = [None if target is None else target.grasp_state for target in targets]
        states, _ = debounce_grasp(states, grasp_converter.grasp_min_frames)
        for target, state in zip(targets, states):
            if target is not None:
                target.grasp_state = int(state)
        result[side] = targets
    return result


def visualize_wilor_cache(
    video_path: str | Path,
    cache_path: str | Path,
    output_path: str | Path,
    *,
    eef_path: str | Path | None = None,
    eef_config_path: str | Path | None = None,
    grasp_close_ratio: float | None = None,
    grasp_open_ratio: float | None = None,
    grasp_min_frames: int | None = None,
    hand2gripper_mode: str | None = None,
    grasp_mode: str | None = None,
    axis_length: float = 0.06,
    ratio_plot_max: float = 1.5,
    max_frames: int | None = None,
) -> Path:
    cache = _load_json(cache_path)
    if cache.get("schema") != "tate.wilor_hands":
        raise ValueError(f"Unsupported WiLoR cache schema: {cache.get('schema')!r}")
    if cache.get("keypoint_order") != "aria_humanego_21":
        raise ValueError("Visualization requires aria_humanego_21 keypoints")

    config = load_yaml(eef_config_path) if eef_config_path else {}
    hand_config = config.get("hand2gripper") or {}
    mode = str(hand2gripper_mode or hand_config.get("mode", FingerCenter.MODE_NAME))
    resolved_grasp_mode = str(grasp_mode or hand_config.get("grasp_mode", mode))
    if mode not in MODES:
        raise ValueError(f"Unknown hand2gripper mode {mode!r}; choose from {sorted(MODES)}")
    if resolved_grasp_mode not in MODES:
        raise ValueError(
            f"Unknown grasp mode {resolved_grasp_mode!r}; choose from {sorted(MODES)}"
        )
    close_ratio = float(
        grasp_close_ratio
        if grasp_close_ratio is not None
        else hand_config.get("grasp_close_ratio", FingerCenter.DEFAULT_GRASP_CLOSE_RATIO)
    )
    open_ratio = float(
        grasp_open_ratio
        if grasp_open_ratio is not None
        else hand_config.get("grasp_open_ratio", FingerCenter.DEFAULT_GRASP_OPEN_RATIO)
    )
    min_frames = int(
        grasp_min_frames
        if grasp_min_frames is not None
        else hand_config.get("grasp_min_frames", FingerCenter.DEFAULT_GRASP_MIN_FRAMES)
    )
    frames = list(cache.get("frames") or [])
    targets = _build_targets(
        frames, mode, resolved_grasp_mode, close_ratio, open_ratio, min_frames
    )

    eef_by_index: dict[int, dict[str, Any]] = {}
    if eef_path is not None:
        eef = _load_json(eef_path)
        eef_by_index = {int(frame["idx"]): frame for frame in eef.get("frames", [])}

    K = np.asarray(cache.get("K"), dtype=np.float64)
    d = np.asarray(cache.get("d"), dtype=np.float64).reshape(-1)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(cache.get("fps") or capture.get(cv2.CAP_PROP_FPS) or 30.0),
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open output video writer: {output}")

    selected = frames if max_frames is None else frames[: int(max_frames)]
    counts = {"right": 0, "left": 0}
    try:
        for sequence_index, frame in enumerate(tqdm(selected, desc="Visualize WiLoR cache")):
            frame_index = int(frame["idx"])
            if int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != frame_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode video frame {frame_index}")
            cv2.putText(
                image,
                f"frame {frame_index}",
                (12, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            eef_frame = eef_by_index.get(frame_index, {})
            for side, hand_key in (("right", "hand_r"), ("left", "hand_l")):
                hand = frame.get(hand_key)
                target = targets[side][sequence_index]
                if hand is None or target is None:
                    continue
                counts[side] += 1
                _draw_skeleton(image, hand, target.grasp_state)
                _draw_grasp_panel(
                    image,
                    side,
                    hand,
                    target.grasp_state,
                    target.grasp_ratio,
                    close_ratio,
                    open_ratio,
                    ratio_plot_max,
                    resolved_grasp_mode,
                )
                eef_record = eef_frame.get(hand_key) or {}
                hand_pose = eef_record.get("hand2gripper_hand_pose_cam", target.T_hand_in_cam)
                _draw_pose_axis(image, hand_pose, K, d, f"{side[0].upper()} HAND", axis_length)
                _draw_pose_axis(
                    image,
                    eef_record.get("tcp_pose_cam"),
                    K,
                    d,
                    f"{side[0].upper()} TCP",
                    axis_length,
                )
            writer.write(image)
    finally:
        capture.release()
        writer.release()

    print(f"[Output] Visualization saved to: {output}")
    print(f"[Output] Visualized right/left detections: {counts['right']}/{counts['left']}")
    print(
        f"[Output] Hand2gripper pose/grasp modes: {mode}/{resolved_grasp_mode}; "
        f"grasp hysteresis: close<{close_ratio:.3f}, "
        f"open>{open_ratio:.3f}, min_frames={min_frames}"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Source RGB video")
    parser.add_argument("--hands", required=True, help="WiLoR cache JSON")
    parser.add_argument("--out", required=True, help="Output MP4")
    parser.add_argument("--eef", default=None, help="Optional exported EEF JSON for TCP axes")
    parser.add_argument("--eef-config", default="./cfg/preprocess/base/EEFExport.yaml")
    parser.add_argument("--grasp-close-ratio", type=float, default=None)
    parser.add_argument("--grasp-open-ratio", type=float, default=None)
    parser.add_argument("--grasp-min-frames", type=int, default=None)
    parser.add_argument("--hand2gripper-mode", choices=sorted(MODES), default=None,
                        help="Override hand2gripper.mode in --eef-config")
    parser.add_argument("--grasp-mode", choices=sorted(MODES), default=None,
                        help="Override hand2gripper.grasp_mode in --eef-config")
    parser.add_argument("--axis-length", type=float, default=0.06)
    parser.add_argument("--ratio-plot-max", type=float, default=1.5)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    visualize_wilor_cache(
        args.video,
        args.hands,
        args.out,
        eef_path=args.eef,
        eef_config_path=args.eef_config,
        grasp_close_ratio=args.grasp_close_ratio,
        grasp_open_ratio=args.grasp_open_ratio,
        grasp_min_frames=args.grasp_min_frames,
        hand2gripper_mode=args.hand2gripper_mode,
        grasp_mode=args.grasp_mode,
        axis_length=args.axis_length,
        ratio_plot_max=args.ratio_plot_max,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
