#!/usr/bin/env python3
"""Compare raw ego EEF, corrected ego EEF, and real ARX TCP paths in MuJoCo.

The robot remains at its home pose.  The script draws static dual-arm paths in
the shared MuJoCo scene frame and compares corresponding gripper-event poses.
Without ``--viewer`` it renders an orbiting MP4; ``--summary-only`` only loads
the trajectories and prints event errors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import replay_arx_mujoco as replay  # noqa: E402
from evaluation.events import extract_events  # noqa: E402
from evaluation.loaders import load_eef_json, load_real_fk_json  # noqa: E402


ROLE_STYLES = {
    "raw": ("Ego EEF before correction", (0.05, 0.85, 0.95, 0.92)),
    "corrected": ("Ego EEF after correction", (1.0, 0.55, 0.05, 0.95)),
    "real": ("Real robot TCP", (0.92, 0.92, 0.92, 0.95)),
}
START_RGBA = (0.15, 0.95, 0.25, 1.0)
END_RGBA = (0.95, 0.15, 0.15, 1.0)
ANCHOR_RGBA = (
    (1.0, 0.90, 0.05, 1.0),
    (0.75, 0.20, 1.0, 1.0),
    (0.10, 1.00, 0.45, 1.0),
    (1.00, 0.20, 0.35, 1.0),
    (0.10, 0.55, 1.0, 1.0),
)
SIDES = ("left", "right")


@dataclass(frozen=True)
class Track:
    role: str
    side: str
    name: str
    points: np.ndarray
    rotations: np.ndarray
    valid: np.ndarray
    grasp: np.ndarray
    events: tuple[Any, ...]
    rgba: tuple[float, float, float, float]

    def anchor_point(self, ordinal: int) -> np.ndarray:
        return self.points[self.events[ordinal].frame_index]

    def anchor_rotation(self, ordinal: int) -> np.ndarray:
        return self.rotations[self.events[ordinal].frame_index]


def as_abs(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def require_matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError(f"{label} must be a finite 4x4 matrix")
    return matrix


def scene_transforms(path: Path, role: str) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if role in ("raw", "corrected"):
        values = payload.get("eef_frame_in_scene") or (
            payload.get("arm_transforms") or {}
        ).get("T_eef_frame_in_scene")
        field = "eef_frame_in_scene"
    else:
        values = (payload.get("coordinate_convention") or {}).get(
            "T_output_frame_in_scene"
        )
        field = "coordinate_convention.T_output_frame_in_scene"
    if not isinstance(values, dict):
        raise RuntimeError(f"{path}: missing {field}")
    return {
        side: require_matrix(values.get(side), f"{path}:{field}.{side}")
        for side in SIDES
    }


def trajectory_to_tracks(path: Path, role: str, selected_sides: tuple[str, ...]) -> list[Track]:
    trajectory = load_real_fk_json(path) if role == "real" else load_eef_json(path)
    transforms = scene_transforms(path, role)
    role_name, color = ROLE_STYLES[role]
    tracks: list[Track] = []
    for side in selected_sides:
        arm = trajectory.sides[side]
        points = np.full((arm.n, 3), np.nan, dtype=np.float64)
        rotations = np.full((arm.n, 3, 3), np.nan, dtype=np.float64)
        for index in np.flatnonzero(arm.valid):
            local = np.eye(4, dtype=np.float64)
            local[:3, 3] = arm.position[index]
            local[:3, :3] = R.from_quat(arm.quaternion_xyzw[index]).as_matrix()
            scene_pose = transforms[side] @ local
            points[index] = scene_pose[:3, 3]
            rotations[index] = scene_pose[:3, :3]
        events = tuple(extract_events(trajectory, side))
        if not events:
            raise RuntimeError(f"{path}: {side} has no gripper events")
        tracks.append(
            Track(
                role=role,
                side=side,
                name=f"{role_name} [{side}]",
                points=points,
                rotations=rotations,
                valid=np.asarray(arm.valid, dtype=bool),
                grasp=np.asarray(arm.gripper_binary, dtype=np.int8),
                events=events,
                rgba=color,
            )
        )
    return tracks


def load_tracks(args: argparse.Namespace) -> list[Track]:
    selected_sides = SIDES if args.side == "both" else (args.side,)
    inputs = (
        ("raw", as_abs(args.eef)),
        ("corrected", as_abs(args.corrected_eef)),
        ("real", as_abs(args.realbot)),
    )
    tracks: list[Track] = []
    for role, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"{role} trajectory not found: {path}")
        tracks.extend(trajectory_to_tracks(path, role, selected_sides))
    return tracks


def tracks_for_side(tracks: list[Track], side: str) -> list[Track]:
    order = {"raw": 0, "corrected": 1, "real": 2}
    return sorted((track for track in tracks if track.side == side), key=lambda t: order[t.role])


def resolve_anchors(
    tracks: list[Track], requested: list[int] | None
) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    for side in dict.fromkeys(track.side for track in tracks):
        side_tracks = tracks_for_side(tracks, side)
        anchors = (
            list(range(min(len(track.events) for track in side_tracks)))
            if requested is None
            else sorted(requested)
        )
        if any(anchor < 0 for anchor in anchors) or len(set(anchors)) != len(anchors):
            raise ValueError(f"--anchors must be unique non-negative values: {anchors}")
        for track in side_tracks:
            missing = [anchor for anchor in anchors if anchor >= len(track.events)]
            if missing:
                raise ValueError(
                    f"{track.name} has {len(track.events)} events; missing anchors {missing}"
                )
        for anchor in anchors:
            transitions = {
                (track.events[anchor].from_state, track.events[anchor].to_state)
                for track in side_tracks
            }
            if len(transitions) != 1:
                raise ValueError(
                    f"{side} anchor {anchor} has inconsistent transitions: "
                    f"{sorted(transitions)}"
                )
        output[side] = anchors
    return output


def event_error_lines(tracks: list[Track], anchors_by_side: dict[str, list[int]]) -> list[str]:
    lines: list[str] = []
    for side, anchors in anchors_by_side.items():
        by_role = {track.role: track for track in tracks_for_side(tracks, side)}
        raw, corrected, real = by_role["raw"], by_role["corrected"], by_role["real"]
        for anchor in anchors:
            event = real.events[anchor]
            values = []
            for label, candidate in (("raw", raw), ("corrected", corrected)):
                position_mm = 1000.0 * np.linalg.norm(
                    candidate.anchor_point(anchor) - real.anchor_point(anchor)
                )
                rotation_deg = np.degrees(
                    (
                        R.from_matrix(real.anchor_rotation(anchor)).inv()
                        * R.from_matrix(candidate.anchor_rotation(anchor))
                    ).magnitude()
                )
                values.append(f"{label}-real={position_mm:.1f}mm/{rotation_deg:.1f}deg")
            lines.append(
                f"{side} A{anchor} {event.from_state}->{event.to_state}  "
                + "  ".join(values)
            )
    return lines


def segment_bounds(track: Track, args: argparse.Namespace) -> tuple[int, int]:
    low = 0 if args.segment_start is None else track.events[args.segment_start].frame_index
    high = len(track.points) - 1 if args.segment_end is None else track.events[args.segment_end].frame_index
    if high < low:
        raise ValueError(f"invalid segment for {track.name}: {low}..{high}")
    return int(low), int(high)


def add_capsule(scn, p0: np.ndarray, p1: np.ndarray, rgba, radius: float) -> None:
    _, mj = replay.require_runtime()
    replay._add_capsule(scn, mj, p0, p1, rgba, radius)


def add_sphere(scn, point: np.ndarray, rgba, radius: float) -> None:
    _, mj = replay.require_runtime()
    replay._add_sphere(scn, mj, point, rgba, radius)


def draw_pose_axes(scn, point: np.ndarray, rotation: np.ndarray, args: argparse.Namespace) -> None:
    colors = (
        (1.0, 0.05, 0.05, 0.9),
        (0.05, 1.0, 0.05, 0.9),
        (0.10, 0.35, 1.0, 0.9),
    )
    for axis, color in enumerate(colors):
        add_capsule(
            scn,
            point,
            point + rotation[:, axis] * args.pose_axis_length,
            color,
            args.pose_axis_radius,
        )


def draw_path(scn, track: Track, args: argparse.Namespace) -> None:
    low, high = segment_bounds(track, args)
    indices = np.flatnonzero(track.valid[low : high + 1]) + low
    indices = indices[:: args.path_stride]
    if not len(indices):
        return
    last_valid = int(np.flatnonzero(track.valid[: high + 1])[-1])
    if indices[-1] != last_valid and last_valid >= low:
        indices = np.append(indices, last_valid)
    for first, second in zip(indices[:-1], indices[1:]):
        add_capsule(scn, track.points[first], track.points[second], track.rgba, args.path_radius)
    add_sphere(scn, track.points[indices[0]], START_RGBA, args.endpoint_radius)
    add_sphere(scn, track.points[indices[-1]], END_RGBA, args.endpoint_radius)
    pose_indices = indices[:: max(1, args.pose_stride // args.path_stride)]
    for index in pose_indices:
        draw_pose_axes(scn, track.points[index], track.rotations[index], args)


def draw_tracks(
    scn, tracks: list[Track], anchors_by_side: dict[str, list[int]], args: argparse.Namespace
) -> None:
    # Draw anchors first so dense paths cannot consume their geometry budget.
    for side, anchors in anchors_by_side.items():
        side_tracks = tracks_for_side(tracks, side)
        for anchor in anchors:
            color = ANCHOR_RGBA[anchor % len(ANCHOR_RGBA)]
            points = [track.anchor_point(anchor) for track in side_tracks]
            if not args.no_anchor_links:
                for first, second in zip(points[:-1], points[1:]):
                    add_capsule(scn, first, second, (*color[:3], 0.38), args.anchor_link_radius)
            for point in points:
                add_sphere(scn, point, color, args.anchor_radius)
    for track in tracks:
        draw_path(scn, track, args)


def fit_camera(points: np.ndarray, args: argparse.Namespace):
    _, mj = replay.require_runtime()
    center = points.mean(axis=0)
    extent = float(np.max(np.linalg.norm(points - center, axis=1)))
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center
    camera.distance = max(extent * args.camera_margin, 0.30)
    camera.azimuth = args.azimuth_start
    camera.elevation = args.elevation
    return camera


def visible_points(tracks: list[Track], args: argparse.Namespace) -> np.ndarray:
    values = []
    for track in tracks:
        low, high = segment_bounds(track, args)
        mask = track.valid[low : high + 1]
        values.append(track.points[low : high + 1][mask])
    return np.concatenate(values, axis=0)


def rgba_to_bgr(rgba) -> tuple[int, int, int]:
    red, green, blue, _ = rgba
    return int(255 * blue), int(255 * green), int(255 * red)


def draw_hud(
    frame: np.ndarray,
    tracks: list[Track],
    anchors_by_side: dict[str, list[int]],
    header: str,
) -> np.ndarray:
    cv, _ = replay.require_runtime()
    x, y = 18, 28

    def add_text(line: str, color=(245, 245, 245), scale=0.48) -> None:
        nonlocal y
        cv.putText(frame, line, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(frame, line, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv.LINE_AA)
        y += 22

    add_text(header, scale=0.58)
    add_text("axes X/Y/Z = red/green/blue; spheres = grasp events", (220, 220, 220), 0.45)
    for role in ("raw", "corrected", "real"):
        label, color = ROLE_STYLES[role]
        cv.rectangle(frame, (x, y - 12), (x + 18, y + 2), rgba_to_bgr(color), -1)
        add_text(f"      {label}")
    for line in event_error_lines(tracks, anchors_by_side):
        add_text(line, scale=0.43)
    return frame


def setup_scene(scene: Path):
    _, mj = replay.require_runtime()
    model = mj.MjModel.from_xml_path(str(scene))
    data = mj.MjData(model)
    key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "open")
    if key_id >= 0:
        mj.mj_resetDataKeyframe(model, data, key_id)
    else:
        mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    return model, data


def launch_viewer(model, data, tracks, anchors_by_side, camera, args) -> None:
    import mujoco.viewer

    print("Launching MuJoCo viewer; close the window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = camera.type
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance = camera.distance
        viewer.cam.azimuth = camera.azimuth
        viewer.cam.elevation = camera.elevation
        while viewer.is_running():
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_tracks(viewer.user_scn, tracks, anchors_by_side, args)
            viewer.set_texts(
                (None, None, "EEF trajectory comparison", "\n".join(event_error_lines(tracks, anchors_by_side)))
            )
            viewer.sync()
            time.sleep(1.0 / 30.0)


def render_mp4(model, data, tracks, anchors_by_side, camera, args) -> None:
    cv, mj = replay.require_runtime()
    output = as_abs(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = mj.Renderer(model, height=args.height, width=args.width)
    frames = max(1, int(round(args.seconds * args.fps)))
    writer = cv.VideoWriter(
        str(output), cv.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"failed to open video writer: {output}")
    try:
        for index in range(frames):
            camera.azimuth = args.azimuth_start + args.orbit_degrees * index / max(frames - 1, 1)
            renderer.update_scene(data, camera=camera)
            if hasattr(renderer, "scene"):
                draw_tracks(renderer.scene, tracks, anchors_by_side, args)
            rgb = renderer.render()
            bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
            writer.write(
                draw_hud(
                    bgr,
                    tracks,
                    anchors_by_side,
                    f"EEF comparison {index + 1}/{frames}",
                )
            )
    finally:
        writer.release()
        renderer.close()
    print(f"Wrote {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eef", "--raw-eef", dest="eef", required=True)
    parser.add_argument("--corrected-eef", required=True)
    parser.add_argument("--realbot", "--real", dest="realbot", required=True)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--scene", default=str(replay.DEFAULT_SCENE))
    parser.add_argument("--out", default="outputs/real2sim/eef_raw_corrected_real_compare.mp4")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--anchors", nargs="*", type=int, default=None)
    parser.add_argument("--segment-start", type=int, default=None)
    parser.add_argument("--segment-end", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--orbit-degrees", type=float, default=360.0)
    parser.add_argument("--azimuth-start", type=float, default=130.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--camera-margin", type=float, default=1.6)
    parser.add_argument("--path-stride", type=int, default=5)
    parser.add_argument("--path-radius", type=float, default=0.0035)
    parser.add_argument("--pose-stride", type=int, default=90)
    parser.add_argument("--pose-axis-length", type=float, default=0.04)
    parser.add_argument("--pose-axis-radius", type=float, default=0.0013)
    parser.add_argument("--endpoint-radius", type=float, default=0.007)
    parser.add_argument("--anchor-radius", type=float, default=0.012)
    parser.add_argument("--anchor-link-radius", type=float, default=0.0012)
    parser.add_argument("--no-anchor-links", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (args.segment_start is None) != (args.segment_end is None):
        raise SystemExit("--segment-start and --segment-end must be supplied together")
    if args.path_stride <= 0 or args.pose_stride <= 0 or args.fps <= 0 or args.seconds <= 0:
        raise SystemExit("strides, fps, and seconds must be positive")
    tracks = load_tracks(args)
    anchors_by_side = resolve_anchors(tracks, args.anchors)
    if args.segment_start is not None:
        resolve_anchors(tracks, [args.segment_start, args.segment_end])

    print("[replay_traj_compare] loaded tracks:")
    for track in tracks:
        print(f"  {track.name}: {int(track.valid.sum())} points, {len(track.events)} events")
    print("[replay_traj_compare] corresponding event errors:")
    for line in event_error_lines(tracks, anchors_by_side):
        print(f"  {line}")
    if args.summary_only:
        return

    scene = as_abs(args.scene)
    if not scene.is_file():
        raise FileNotFoundError(f"MuJoCo scene not found: {scene}")
    replay.load_runtime(args.viewer)
    model, data = setup_scene(scene)
    camera = fit_camera(visible_points(tracks, args), args)
    if args.viewer:
        launch_viewer(model, data, tracks, anchors_by_side, camera, args)
    else:
        render_mp4(model, data, tracks, anchors_by_side, camera, args)


if __name__ == "__main__":
    main()
