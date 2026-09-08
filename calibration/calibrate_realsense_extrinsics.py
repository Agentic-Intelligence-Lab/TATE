#!/usr/bin/env python3
"""Calibrate the ARX head RealSense camera extrinsics for one arm base from AprilTags.

Method: PnP against desktop AprilTags whose top_left corner has a known 3D
position in the robot base frame (measured by touching the robot TCP to that
one corner -- see collect_apriltag_corners.py). Each tag's other 3 corners
are NOT measured; the tag is known to be a flat square of fixed side length
lying in a horizontal plane (same z as the measured corner), but its yaw
(rotation about +Z) on the table is unknown. So this script jointly solves
for the camera pose AND every tag's yaw at once, by minimizing reprojection
error against the tags' auto-detected pixel corners (cv2.aruco) -- a coarse
grid search over yaw candidates picks a good starting point (avoiding local
minima), then scipy.optimize.least_squares polishes camera pose + yaws
together. The square's chirality (which rotation direction is "top_left ->
top_right" vs "top_left -> bottom_left") is physically the same for every
tag lying face-up on the table under the same camera, so it is searched once
globally (2 options) rather than per tag.

Input JSON for --tag-corners-base (units: meters; the single point per tag
is top_left as seen by the camera when the tag is viewed right-side up --
matches collect_apriltag_corners.py's output):

    {
      "units": "meters",
      "tag_size_m": 0.05,
      "tags": {
        "1": [x, y, z],
        "2": [x, y, z]
      }
    }

Tag family: pass --tag-family with a short name -- 16h5, 25h9, 36h10, 36h11, or
41h12. 16h5/25h9/36h10/36h11 are detected via cv2.aruco (OpenCV ships
these built in). 41h12 is not shipped by cv2.aruco, so it's detected via
`pupil_apriltags` instead (pip install pupil-apriltags), which wraps the
official AprilRobotics C library. Full names (DICT_APRILTAG_36h11,
tagStandard41h12, ...) also work if you need them.

    --tag-family 36h11      # cv2.aruco
    --tag-family 41h12      # pupil_apriltags

Input images: either point at existing photo(s) with --images, or have the
script take the photo itself with --capture (RealSense by default; add more
--images to also mix in previously captured frames). Examples:

    # Use an existing photo
    python calibration/calibrate_realsense_extrinsics.py \\
        --images outputs/camera_extrinsics/captured/shot.png \\
        --arm left --tag-corners-base path/to/tag_corners_base_arx_left.json

    # Let the script capture a photo itself (ARX head D405, default 1280x720@30fps)
    python calibration/calibrate_realsense_extrinsics.py \\
        --capture --capture-count 3 \\
        --arm left --tag-corners-base path/to/tag_corners_base_arx_left.json

    # Lower resolution instead
    python calibration/calibrate_realsense_extrinsics.py \\
        --capture --capture-width 1280 --capture-height 720 --capture-fps 15 \\
        --arm left --tag-corners-base path/to/tag_corners_base_arx_left.json

    # Capture with a plain UVC/OpenCV camera instead of RealSense
    python calibration/calibrate_realsense_extrinsics.py \\
        --capture --capture-backend opencv --camera-index 1 \\
        --arm left --tag-corners-base path/to/tag_corners_base_arx_left.json

Intrinsics: RealSense capture reads the active color stream profile directly.
For existing images or the OpenCV capture backend, the script uses the ARX head
D405's saved factory intrinsics at 1280x720 from calibration/configs/. Pass
--intrinsics-json to explicitly select another camera or resolution. A mismatch
between the resolved intrinsics and the actual image size prints a [warn].

Input JSON for --intrinsics-json (RealSense pyrealsense2.intrinsics-style
fields also accepted: ppx/ppy in place of cx/cy, coeffs in place of
dist_coeffs; a top-level "by_resolution" map with one entry per resolution,
as produced by read_camera_intrinsics.py, is also accepted directly):

    {
      "width": 1280, "height": 720,
      "fx": 905.86, "fy": 905.42, "cx": 638.85, "cy": 361.16,
      "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]
    }

fx/fy/cx/cy/dist-coeffs can also be passed directly as CLI flags instead of
(or to override individual fields of) --intrinsics-json.

Output: T_cam_in_base -- the camera pose in the robot base frame -- which is
the requested camera extrinsic, plus its inverse T_base_in_cam, the solved
per-tag yaw angles, reprojection error diagnostics, and annotated debug
images.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from .update_yaml_from_result import DEFAULT_YAML, update_yaml_from_result
except ImportError:  # Direct execution: python calibration/calibrate_realsense_extrinsics.py
    from update_yaml_from_result import DEFAULT_YAML, update_yaml_from_result

REPO_ROOT = Path(__file__).resolve().parents[1]

ARX_HEAD_CAMERA_SERIAL = "409122273248"
ARX_HEAD_INTRINSICS_JSON = "calibration/configs/d405_head_1280x720_intrinsics.json"

PNP_METHOD_ATTRS = {
    "sqpnp": "SOLVEPNP_SQPNP",
    "epnp": "SOLVEPNP_EPNP",
    "iterative": "SOLVEPNP_ITERATIVE",
}


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def matrix_record(T: np.ndarray) -> dict:
    rot = R.from_matrix(T[:3, :3])
    return {
        "T": T.tolist(),
        "translation_m": T[:3, 3].tolist(),
        "quat_xyzw": rot.as_quat().tolist(),
        "rotvec": rot.as_rotvec().tolist(),
    }


# ---------------------------------------------------------------------------
# Intrinsics / tag geometry loading
# ---------------------------------------------------------------------------


def load_intrinsics(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    data: dict = {}
    source_desc = None
    if args.intrinsics_json:
        data = json.loads(as_abs(args.intrinsics_json).read_text(encoding="utf-8"))
        source_desc = args.intrinsics_json
    elif getattr(args, "captured_intrinsics", None):
        data = args.captured_intrinsics
        source_desc = getattr(args, "captured_intrinsics_source", "RealSense active color stream profile")
    elif args.fx is None and args.fy is None:
        data = json.loads(as_abs(ARX_HEAD_INTRINSICS_JSON).read_text(encoding="utf-8"))
        source_desc = ARX_HEAD_INTRINSICS_JSON

    by_res = data.get("by_resolution")
    if by_res:
        if args.resolution:
            key = args.resolution
            if key not in by_res:
                raise ValueError(f"resolution '{key}' not in {source_desc}'s by_resolution: {list(by_res)}")
        elif len(by_res) == 1:
            key = next(iter(by_res))
        else:
            raise ValueError(
                f"{source_desc} has multiple resolutions {list(by_res)}; pick one with --resolution WxH "
                "(auto-set from --capture-width/--capture-height when --capture is used, or from the "
                "first --images frame's actual pixel size otherwise)."
            )
        data = by_res[key]

    fx = args.fx if args.fx is not None else data.get("fx")
    fy = args.fy if args.fy is not None else data.get("fy")
    cx = args.cx if args.cx is not None else data.get("cx", data.get("ppx"))
    cy = args.cy if args.cy is not None else data.get("cy", data.get("ppy"))
    if fx is None or fy is None or cx is None or cy is None:
        raise ValueError(
            "Missing camera intrinsics: need fx, fy, cx (or ppx), cy (or ppy) "
            "via --intrinsics-json and/or --fx/--fy/--cx/--cy."
        )

    dist = args.dist_coeffs
    if dist is None:
        dist = data.get("dist_coeffs", data.get("coeffs"))
    if dist is None:
        dist = [0.0, 0.0, 0.0, 0.0, 0.0]

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_arr = np.array(dist, dtype=np.float64).reshape(-1, 1)
    meta = {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "dist_coeffs": dist_arr.flatten().tolist(),
        "width": data.get("width"),
        "height": data.get("height"),
        "source": source_desc,
        "source_json": args.intrinsics_json or (
            ARX_HEAD_INTRINSICS_JSON if not getattr(args, "captured_intrinsics", None) else None
        ),
    }
    return K, dist_arr, meta


def load_tag_top_left_base(path: str, tag_size_override: float | None) -> tuple[dict[int, np.ndarray], float, dict]:
    data = json.loads(as_abs(path).read_text(encoding="utf-8"))
    tags_raw = data.get("tags")
    if not tags_raw:
        raise ValueError(f"{path}: missing top-level 'tags' object")

    tag_size = tag_size_override if tag_size_override is not None else data.get("tag_size_m")
    if tag_size is None:
        raise ValueError(f"{path}: missing 'tag_size_m' (or pass --tag-size-m explicitly)")

    out: dict[int, np.ndarray] = {}
    for key, value in tags_raw.items():
        arr = np.array(value, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"tag '{key}': expected a single [x,y,z] top_left point (got shape {arr.shape})")
        out[int(key)] = arr

    meta = {
        "units": data.get("units", "meters"),
        "corner_captured": data.get("corner_captured", "top_left"),
        "tag_size_m": float(tag_size),
        "tag_ids": sorted(out.keys()),
    }
    return out, float(tag_size), meta


def corners_from_top_left(top_left: np.ndarray, size: float, yaw: float, handedness: float) -> np.ndarray:
    """Build [top_left, top_right, bottom_right, bottom_left] (base frame, planar in z)
    from the measured top_left corner, known side length, an unknown yaw (rotation of
    the "top_left -> top_right" edge about +Z), and a shared chirality sign."""
    right = size * np.array([np.cos(yaw), np.sin(yaw), 0.0])
    down = size * handedness * np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    top_right = top_left + right
    bottom_left = top_left + down
    bottom_right = top_right + down
    return np.stack([top_left, top_right, bottom_right, bottom_left], axis=0)


# ---------------------------------------------------------------------------
# AprilTag detection
# ---------------------------------------------------------------------------


def get_aruco_dictionary(family: str):
    candidates = [family, family.upper(), family.replace("h", "H"), family.replace("H", "h")]
    for name in candidates:
        if hasattr(cv2.aruco, name):
            return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    available = sorted(n for n in dir(cv2.aruco) if "APRILTAG" in n.upper())
    raise ValueError(f"Unknown tag family '{family}'. Available: {available}")


# Empirically determined for the current printed tag batch (verified by comparing
# annotate_apriltags.py's numbered corner overlay against which physical corner
# collect_apriltag_corners.py's operator actually touches): cv2.aruco's raw per-marker
# corner order for these tags is [bottom_right, bottom_left, top_left, top_right], NOT
# the textbook "clockwise from top_left" order docs describe. That means the tags'
# print/generation happened to place the algorithm's rotation-invariant first corner at
# what a human viewing the tag right-side-up would call the bottom-right -- i.e. this is
# a property of this specific print batch, not of cv2.aruco in general. This permutes
# raw index -> our [top_left, top_right, bottom_right, bottom_left] convention (index 0 =
# the corner the operator physically touches), matching what corners_from_top_left /
# solve_joint_pose_and_yaws assumes. If tags are reprinted
# or a different family/generator is used, re-verify with annotate_apriltags.py (its
# red-dot label = index 0 after this reorder) before trusting this constant again.
ARUCO_CORNER_REORDER = [2, 3, 0, 1]


def make_aruco_detect_fn(family: str):
    """cv2.aruco path -- only covers the families OpenCV bundles: 16h5/25h9/36h10/36h11."""
    dictionary = get_aruco_dictionary(family)
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        raw_detect = lambda gray: detector.detectMarkers(gray)
    else:
        raw_detect = lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    def detect(gray: np.ndarray) -> dict[int, np.ndarray]:
        corners, ids, _ = raw_detect(gray)
        result: dict[int, np.ndarray] = {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                raw = c.reshape(4, 2).astype(np.float64)
                result[int(i)] = raw[ARUCO_CORNER_REORDER]
        return result

    return detect


_PUPIL_FAMILY_ALIASES = {
    "41h12": "tagStandard41h12",
    "tag41h12": "tagStandard41h12",
    "standard41h12": "tagStandard41h12",
    "tagstandard41h12": "tagStandard41h12",
}

# Short names -> (backend, canonical name), so --tag-family takes the same kind of
# plain "36h11" / "41h12" value regardless of which detector backend actually
# handles it. cv2.aruco ships 16h5/25h9/36h10/36h11; everything else (41h12) falls
# back to pupil_apriltags. Full names (DICT_APRILTAG_36h11, tagStandard41h12, ...)
# still work directly too -- see make_detect_fn.
TAG_FAMILY_SHORTCUTS: dict[str, tuple[str, str]] = {
    "16h5": ("aruco", "DICT_APRILTAG_16h5"),
    "25h9": ("aruco", "DICT_APRILTAG_25h9"),
    "36h10": ("aruco", "DICT_APRILTAG_36h10"),
    "36h11": ("aruco", "DICT_APRILTAG_36h11"),
    "41h12": ("pupil", "tagStandard41h12"),
}


def make_pupil_apriltags_detect_fn(family: str):
    """Fallback path for AprilTag families OpenCV's cv2.aruco doesn't ship -- notably
    41h12 (default). Uses `pupil_apriltags`, a wrapper around the official AprilRobotics
    C library, which supports the full family set."""
    try:
        from pupil_apriltags import Detector as PupilDetector
    except ImportError as exc:
        raise ImportError(
            f"tag family '{family}' is not one of OpenCV's built-in DICT_APRILTAG_* families "
            "(16h5/25h9/36h10/36h11) -- cv2.aruco does not ship 41h12. Install the "
            "AprilRobotics-backed detector instead:\n\n    pip install pupil-apriltags\n"
        ) from exc

    pupil_family = _PUPIL_FAMILY_ALIASES.get(family.strip().lower(), family)
    detector = PupilDetector(families=pupil_family)

    def detect(gray: np.ndarray) -> dict[int, np.ndarray]:
        detections = detector.detect(gray, estimate_tag_pose=False)
        result: dict[int, np.ndarray] = {}
        for det in detections:
            # AprilTag reference lib order: counter-clockwise starting at the corner
            # nearest the tag's own (bottom-left-when-upright) origin, i.e.
            # [bottom_left, bottom_right, top_right, top_left] -- the reverse of
            # OpenCV's aruco order. Reverse it so index 0 is top_left, matching
            # collect_apriltag_corners.py's convention (and make_aruco_detect_fn above).
            result[int(det.tag_id)] = np.asarray(det.corners, dtype=np.float64)[::-1]
        return result

    return detect


def _detect_both_polarities(base_detect, gray: np.ndarray) -> dict[int, np.ndarray]:
    """Both cv2.aruco and the AprilTag reference library only look for one fixed
    polarity (dark border/bits on a light background) -- a tag printed as a
    black/white negative is invisible to a single detect() call (border extraction
    AND bit decoding are both polarity-dependent, not just "less reliable"). Try
    the normal image and its color-inverted version and merge, so mixed-polarity
    prints (some tags normal, some accidentally inverted) both work. Normal-polarity
    detections win on an id collision (shouldn't happen for a correctly-decoded tag)."""
    result = base_detect(gray)
    inverted = base_detect(cv2.bitwise_not(gray))
    for tag_id, corners in inverted.items():
        result.setdefault(tag_id, corners)
    return result


def make_detect_fn(family: str) -> tuple[object, str]:
    """Pick the detector backend for --tag-family. A short name (16h5/25h9/36h10/36h11/
    41h12) is looked up in TAG_FAMILY_SHORTCUTS; 36h10/36h11/25h9/16h5 route to cv2.aruco,
    41h12 to pupil_apriltags (cv2.aruco doesn't ship it). Full names (DICT_APRILTAG_36h11,
    tagStandard41h12, ...) also work directly, same routing rule (DICT_* -> aruco, else
    -> pupil_apriltags). The returned detect_fn tries both normal and inverted polarity
    (see _detect_both_polarities). Returns (detect_fn(gray) -> dict[tag_id, (4,2) corners
    in top_left/top_right/bottom_right/bottom_left order], backend_name)."""
    shortcut = TAG_FAMILY_SHORTCUTS.get(family.strip().lower())
    if shortcut is not None:
        backend, canonical = shortcut
    elif family.upper().startswith("DICT_"):
        backend, canonical = "aruco", family
    else:
        backend, canonical = "pupil", family

    if backend == "aruco":
        base = make_aruco_detect_fn(canonical)
        return (lambda gray: _detect_both_polarities(base, gray)), f"opencv.aruco:{canonical}"
    pupil_family = _PUPIL_FAMILY_ALIASES.get(canonical.strip().lower(), canonical)
    base = make_pupil_apriltags_detect_fn(canonical)
    return (lambda gray: _detect_both_polarities(base, gray)), f"pupil_apriltags:{pupil_family}"


# ---------------------------------------------------------------------------
# PnP solve + diagnostics
# ---------------------------------------------------------------------------


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    method: str,
    refine: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    attr = PNP_METHOD_ATTRS.get(method, PNP_METHOD_ATTRS["epnp"])
    flag = getattr(cv2, attr, None)
    used = method
    if flag is None:
        flag = cv2.SOLVEPNP_EPNP
        used = "epnp (fallback)"
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist, flags=flag)
    if not ok:
        raise RuntimeError("cv2.solvePnP failed to converge")
    if refine:
        rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, K, dist, rvec, tvec)
    return rvec, tvec, used


def solve_joint_pose_and_yaws(
    top_lefts: dict[int, np.ndarray],
    size: float,
    image_points_by_tag: dict[int, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    pnp_method: str,
    yaw_grid_step_deg: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, dict[int, float], float]:
    """Jointly solve camera pose (rvec, tvec) and each tag's yaw about +Z, given
    only each tag's top_left corner (base frame) and its fixed side length. Scales
    to any number of tags N (not just 2) -- see the two init strategies below.

    All tags lie in the same horizontal plane (same z), so the point set as a
    whole is exactly planar -- this makes the general 6-DOF pose ambiguous:
    reflecting the camera through the tags' plane (and correspondingly
    flipping yaw/chirality) reproduces an almost identical image with an
    equally low reprojection error, just with the camera on the wrong side of
    the table. We break that tie with the physical fact that this camera
    looks down at the table from above: candidates whose recovered camera
    z (base frame) is not above the tags' plane are rejected outright.

    Initial guess, N <= 2: 1-2 exact points can't constrain a 6-DOF pose by
    themselves (need >=3 non-collinear points in general), so fall back to a
    small combinatorial search over these tag(s)' own yaw x {+1,-1} chirality
    (cheap at N<=2: len(yaw_grid)^N x 2, i.e. <=1152 evaluations at the default
    15deg step -- this is the same search the old N-agnostic version of this
    function did for every N, which is what made N>=4 impractically slow).

    Initial guess, N >= 3: each tag's top_left corner is already an exact,
    unambiguous 3D<->2D correspondence (no yaw involved) -- so a rough camera
    pose can be solved directly from just the N top_left points via ordinary
    PnP (cv2.solvePnPGeneric with SQPNP, which handles any N>=3 and returns
    every candidate pose so the reflected-ambiguity twin can be filtered out
    same as above), with no yaw search needed at all for this step. Holding
    that pose fixed, each tag's yaw is then grid-searched independently
    (O(N x len(yaw_grid)), not O(len(yaw_grid)^N)) against its own 4-point
    reprojection error; the two global chirality options are compared by
    their summed error across all tags and the better one is kept (chirality
    is physically the same for every tag lying face-up under the same
    camera, so it's not per-tag).

    Either way, scipy.optimize.least_squares then polishes camera pose + all
    N yaws together from that initial guess -- this stage is smooth/gradient
    based and was always O(N), it's only the initial guess that used to be
    combinatorial in N.
    """
    from scipy.optimize import least_squares

    tag_ids = sorted(top_lefts.keys())
    n_tags = len(tag_ids)
    image_points = np.concatenate([image_points_by_tag[t] for t in tag_ids], axis=0)
    tag_plane_z = float(np.mean([top_lefts[t][2] for t in tag_ids]))

    def build_object_points(yaws: list[float] | np.ndarray, handedness: float) -> np.ndarray:
        return np.concatenate(
            [corners_from_top_left(top_lefts[t], size, y, handedness) for t, y in zip(tag_ids, yaws)], axis=0
        )

    def camera_position_base(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        R_mat, _ = cv2.Rodrigues(rvec)
        return (-R_mat.T @ tvec.reshape(3, 1)).flatten()

    yaw_grid = np.deg2rad(np.arange(0.0, 360.0, yaw_grid_step_deg))

    if n_tags <= 2:
        # A pose can't be pinned down from 1-2 exact points alone (6 DOF, need >=3
        # non-collinear points in general), so there's no cheaper bootstrap than
        # searching this/these tag(s)' own yaw directly. Still cheap: len(yaw_grid)^n_tags
        # x 2, i.e. <=1152 evaluations for n_tags<=2 at the default 15deg step.
        import itertools

        best = None  # (rmse, yaws, handedness, rvec, tvec)
        for handedness in (1.0, -1.0):
            for yaws in itertools.product(yaw_grid, repeat=n_tags):
                obj = build_object_points(yaws, handedness)
                try:
                    rvec, tvec, _ = solve_pnp(obj, image_points, K, dist, pnp_method, refine=False)
                except RuntimeError:
                    continue
                if camera_position_base(rvec, tvec)[2] <= tag_plane_z:
                    continue  # camera below/at the table plane: the reflected-ambiguity twin, reject
                _, _, rmse = reprojection_error(obj, image_points, K, dist, rvec, tvec)
                if best is None or rmse < best[0]:
                    best = (rmse, yaws, handedness, rvec, tvec)
        if best is None:
            raise RuntimeError(
                f"pose+yaw search over {n_tags} tag(s) found no solution with the camera above the "
                f"tags' plane (z > {tag_plane_z:.4f} m) -- check --tag-corners-base and the detected "
                "tag ids."
            )
        _, yaws0, handedness, rvec0, tvec0 = best
    else:
        # Rough pose from the N top_left correspondences alone (exact points, no yaw
        # needed) -- SQPNP handles any N>=3 (unlike EPNP, which needs N>=4).
        # solvePnPGeneric returns every candidate root so the reflected twin (camera
        # below the table) can be filtered out the same way as the N<=2 path.
        top_left_obj = np.stack([top_lefts[t] for t in tag_ids], axis=0)
        top_left_img = np.stack([image_points_by_tag[t][0] for t in tag_ids], axis=0)
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(top_left_obj, top_left_img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        best_stage1 = None  # (rmse, rvec, tvec)
        for rvec, tvec in zip(rvecs, tvecs):
            if camera_position_base(rvec, tvec)[2] <= tag_plane_z:
                continue
            _, _, rmse = reprojection_error(top_left_obj, top_left_img, K, dist, rvec, tvec)
            if best_stage1 is None or rmse < best_stage1[0]:
                best_stage1 = (rmse, rvec, tvec)
        if best_stage1 is None:
            raise RuntimeError(
                f"couldn't get an initial camera pose from the {n_tags} tags' top_left corners alone "
                "with the camera above the tags' plane -- check --tag-corners-base and the detected "
                "tag ids (tags nearly collinear or too few for a stable PnP solve can also cause this)."
            )
        _, rvec0, tvec0 = best_stage1

        # Per tag, per global chirality: best yaw against that tag's own 4-point error,
        # holding the stage-1 pose fixed. O(N x len(yaw_grid)), not O(len(yaw_grid)^N).
        handedness_totals: dict[float, float] = {}
        yaws_by_handedness: dict[float, list[float]] = {}
        for handedness in (1.0, -1.0):
            total_rmse = 0.0
            yaws_for_h = []
            for t in tag_ids:
                img_t = image_points_by_tag[t]
                best_tag = None  # (rmse, yaw)
                for yaw in yaw_grid:
                    obj_t = corners_from_top_left(top_lefts[t], size, yaw, handedness)
                    _, _, rmse = reprojection_error(obj_t, img_t, K, dist, rvec0, tvec0)
                    if best_tag is None or rmse < best_tag[0]:
                        best_tag = (rmse, yaw)
                total_rmse += best_tag[0]
                yaws_for_h.append(best_tag[1])
            handedness_totals[handedness] = total_rmse
            yaws_by_handedness[handedness] = yaws_for_h
        handedness = min(handedness_totals, key=handedness_totals.get)
        yaws0 = yaws_by_handedness[handedness]

    def residuals(params: np.ndarray) -> np.ndarray:
        rvec = params[0:3]
        tvec = params[3:6]
        yaws = params[6:]
        obj = build_object_points(yaws, handedness)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        return (proj.reshape(-1, 2) - image_points).ravel()

    x0 = np.concatenate([rvec0.flatten(), tvec0.flatten(), np.array(yaws0, dtype=np.float64)])
    result = least_squares(residuals, x0, method="lm")

    rvec = result.x[0:3].reshape(3, 1)
    tvec = result.x[3:6].reshape(3, 1)
    if camera_position_base(rvec, tvec)[2] <= tag_plane_z:
        raise RuntimeError(
            "local refinement drifted to the camera-below-the-table twin solution; "
            "try a finer --yaw-grid-step-deg"
        )
    yaws_final = result.x[6:]
    yaw_by_tag = {t: float(np.rad2deg(y) % 360.0) for t, y in zip(tag_ids, yaws_final)}
    return rvec, tvec, yaw_by_tag, handedness


def reprojection_error(
    object_points: np.ndarray, image_points: np.ndarray, K: np.ndarray, dist: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    per_point = np.linalg.norm(proj - image_points, axis=1)
    rmse = float(np.sqrt(np.mean(per_point**2)))
    return proj, per_point, rmse


def leave_one_out_tag_diagnostics(
    top_lefts: dict[int, np.ndarray],
    tag_ids: list[int],
    size: float,
    image_points_by_tag: dict[int, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    pnp_method: str,
    yaw_grid_step_deg: float,
) -> dict[int, float | None]:
    """For each tag, solve the camera pose from every OTHER tag alone, then measure how
    well the excluded tag's own 4 points fit that externally-determined pose (best yaw
    for just this one tag, holding the pose fixed). Requires >=3 tags total (so >=2
    remain per exclusion for an unambiguous pose -- see solve_joint_pose_and_yaws).

    This catches bad per-tag data (e.g. the physically touched corner not matching the
    detector's corner 0, or a mismeasured top_left) that per-point residuals from the
    FULL joint solve can miss: with N-1 good tags and 1 bad one, the shared least_squares
    fit doesn't reject the bad tag, it quietly drags the whole camera pose to partially
    accommodate it, smearing elevated error across every tag's own residuals instead of
    concentrating it on the bad one (confirmed empirically -- a single bad anchor among 4
    tags shifted the solved camera position by 300+mm while every tag's own per-point
    reprojection error looked similarly unremarkable). Leave-one-out sidesteps this by
    never letting a tag influence the pose used to judge it.

    Returns {tag_id: rmse_px} for tags whose leave-one-out solve succeeded, or
    {tag_id: None} when the remaining tags' pose solve itself failed (this can itself be
    a symptom of a *different* excluded tag being bad -- a corrupted tag sitting in the
    "rest" set for several other tags' checks can make those checks fail outright).
    """
    yaw_grid = np.deg2rad(np.arange(0.0, 360.0, yaw_grid_step_deg))
    result: dict[int, float | None] = {}
    for t in tag_ids:
        rest = [x for x in tag_ids if x != t]
        try:
            rvec_r, tvec_r, _, _ = solve_joint_pose_and_yaws(
                {x: top_lefts[x] for x in rest},
                size,
                {x: image_points_by_tag[x] for x in rest},
                K,
                dist,
                pnp_method,
                yaw_grid_step_deg,
            )
        except RuntimeError:
            result[t] = None
            continue
        img_t = image_points_by_tag[t]
        best_rmse = None
        for handedness in (1.0, -1.0):
            for yaw in yaw_grid:
                obj_t = corners_from_top_left(top_lefts[t], size, yaw, handedness)
                _, _, rmse = reprojection_error(obj_t, img_t, K, dist, rvec_r, tvec_r)
                if best_rmse is None or rmse < best_rmse:
                    best_rmse = rmse
        result[t] = best_rmse
    return result


def draw_debug_image(
    img: np.ndarray,
    detections: dict[int, np.ndarray],
    proj_by_tag: dict[int, np.ndarray],
) -> np.ndarray:
    vis = img.copy()
    for tag_id, corners in detections.items():
        pts = corners.astype(int)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        # Corner index labels (0=top_left .. 3=bottom_left by this script's convention) --
        # a quick visual sanity check that the detector backend's corner order matches
        # what collect_apriltag_corners.py measured, especially when switching tag family.
        for idx, (x, y) in enumerate(pts):
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(vis, str(idx), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(vis, f"id{tag_id} detected", (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for tag_id, proj in proj_by_tag.items():
        pts = proj.astype(int)
        cv2.polylines(vis, [pts], True, (0, 0, 255), 1)
        for x, y in pts:
            cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
    legend = "green=detected  red=yaw-solve reprojected"
    cv2.putText(vis, legend, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(vis, "digits=corner idx (0=top_left)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return vis


# ---------------------------------------------------------------------------
# Photo capture (--capture): take a fresh frame instead of/in addition to
# pointing --images at existing photo(s)
# ---------------------------------------------------------------------------


def capture_images(args: argparse.Namespace, out_dir: Path) -> list[Path]:
    capture_dir = out_dir / "captured"
    capture_dir.mkdir(parents=True, exist_ok=True)
    if args.capture_backend == "realsense":
        return _capture_realsense(args, capture_dir)
    return _capture_opencv(args, capture_dir)


def _list_realsense_color_profiles(rs) -> str:
    lines = []
    for dev in rs.context().query_devices():
        lines.append(f"  device {dev.get_info(rs.camera_info.serial_number)} ({dev.get_info(rs.camera_info.name)}):")
        seen = set()
        for sensor in dev.query_sensors():
            for p in sensor.get_stream_profiles():
                if p.stream_type() != rs.stream.color:
                    continue
                vp = p.as_video_stream_profile()
                key = (vp.width(), vp.height(), vp.fps())
                if key in seen:
                    continue
                seen.add(key)
        for w, h, fps in sorted(seen, key=lambda k: (-k[0] * k[1], -k[2])):
            lines.append(f"    {w}x{h}@{fps}")
    return "\n".join(lines) if lines else "  (no RealSense devices found)"


def _capture_realsense(args: argparse.Namespace, capture_dir: Path) -> list[Path]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ImportError(
            "--capture --capture-backend realsense needs pyrealsense2: pip install pyrealsense2 "
            "(or pass --capture-backend opencv for a plain UVC camera)"
        ) from exc

    pipeline = rs.pipeline()
    config = rs.config()
    if args.camera_serial != "auto":
        config.enable_device(args.camera_serial)
    config.enable_stream(rs.stream.color, args.capture_width, args.capture_height, rs.format.bgr8, args.capture_fps)

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise RuntimeError(
            f"RealSense couldn't start a {args.capture_width}x{args.capture_height}@{args.capture_fps} "
            f"color stream ({exc}). That resolution/fps combo likely isn't supported by this camera "
            f"-- supported color profiles:\n{_list_realsense_color_profiles(rs)}\n"
            "Pick a supported combo with --capture-width/--capture-height/--capture-fps."
        ) from exc
    paths: list[Path] = []
    try:
        serial = profile.get_device().get_info(rs.camera_info.serial_number)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        args.captured_intrinsics = {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "distortion_model": str(intr.model),
            "coeffs": [float(value) for value in intr.coeffs],
        }
        args.captured_intrinsics_source = f"RealSense active color stream profile (serial {serial})"
        print(
            f"[capture] RealSense {serial} @ {args.capture_width}x{args.capture_height}@{args.capture_fps} -- "
            f"warming up {args.capture_warmup_frames} frames ..."
        )
        print(
            f"[capture] intrinsics fx={intr.fx:.6f} fy={intr.fy:.6f} "
            f"cx={intr.ppx:.6f} cy={intr.ppy:.6f}"
        )
        for _ in range(args.capture_warmup_frames):
            pipeline.wait_for_frames(3000)
        for i in range(args.capture_count):
            frames = pipeline.wait_for_frames(3000)
            color = frames.get_color_frame()
            if not color:
                raise RuntimeError("RealSense pipeline returned no color frame")
            image = np.asanyarray(color.get_data())
            path = capture_dir / f"capture_{i:02d}.png"
            cv2.imwrite(str(path), image)
            print(f"[capture] wrote {path}")
            paths.append(path)
            if i + 1 < args.capture_count:
                time.sleep(args.capture_interval)
    finally:
        pipeline.stop()
    return paths


def _capture_opencv(args: argparse.Namespace, capture_dir: Path) -> list[Path]:
    from platform import system

    backend = cv2.CAP_DSHOW if system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera_index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera index {args.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
    cap.set(cv2.CAP_PROP_FPS, args.capture_fps)

    paths: list[Path] = []
    try:
        print(f"[capture] opencv camera {args.camera_index} -- warming up {args.capture_warmup_frames} frames ...")
        for _ in range(args.capture_warmup_frames):
            cap.read()
        for i in range(args.capture_count):
            ok, image = cap.read()
            if not ok or image is None:
                raise RuntimeError("camera returned no frame")
            path = capture_dir / f"capture_{i:02d}.png"
            cv2.imwrite(str(path), image)
            print(f"[capture] wrote {path}")
            paths.append(path)
            if i + 1 < args.capture_count:
                time.sleep(args.capture_interval)
    finally:
        cap.release()
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def calibrate(args: argparse.Namespace) -> None:
    out_dir = as_abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    if not args.no_debug_images:
        debug_dir.mkdir(parents=True, exist_ok=True)

    captured_paths: list[Path] = []
    if args.capture:
        captured_paths = capture_images(args, out_dir)
        if args.resolution is None:
            args.resolution = f"{args.capture_width}x{args.capture_height}"

    image_paths = [as_abs(p) for p in args.images] + captured_paths
    if not image_paths:
        raise ValueError("no input images: pass --images, --capture, or both")

    if args.resolution is None:
        # Not set by --capture above (plain --images run) -- infer from the first
        # image's actual pixel size so load_intrinsics() picks matching factory
        # intrinsics without the user having to spell out --resolution by hand.
        peek = cv2.imread(str(image_paths[0]))
        if peek is not None:
            h, w = peek.shape[:2]
            args.resolution = f"{w}x{h}"

    K, dist, intrinsics_meta = load_intrinsics(args)
    top_lefts, tag_size, tags_meta = load_tag_top_left_base(args.tag_corners_base, args.tag_size_m)
    detect_fn, detector_backend = make_detect_fn(args.tag_family)

    per_tag_pixels: dict[int, list[tuple[str, np.ndarray]]] = {}
    per_image_data: list[dict] = []

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"failed to read image: {path}")
        img_h, img_w = img.shape[:2]
        if intrinsics_meta.get("width") and intrinsics_meta.get("height"):
            if (intrinsics_meta["width"], intrinsics_meta["height"]) != (img_w, img_h):
                print(
                    f"[warn] {path.name}: image is {img_w}x{img_h} but intrinsics are for "
                    f"{intrinsics_meta['width']}x{intrinsics_meta['height']} -- fx/fy/cx/cy won't "
                    "match this frame's scale; pass --resolution or --intrinsics-json explicitly."
                )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = detect_fn(gray)
        for tag_id, corners in detections.items():
            if tag_id in top_lefts:
                per_tag_pixels.setdefault(tag_id, []).append((str(path), corners))
        per_image_data.append({"path": path, "img": img, "detections": detections})

    tags_used = sorted(per_tag_pixels.keys())
    tags_missing = sorted(set(top_lefts.keys()) - set(tags_used))
    if not tags_used:
        raise RuntimeError(f"No known tags (ids {sorted(top_lefts.keys())}) were detected in any input image.")
    if len(tags_used) < 2:
        print(
            f"[warn] only tag {tags_used[0]} detected; a single planar tag (4 points) is prone to "
            "pose ambiguity (near-zero reprojection error at the wrong camera pose) -- add more "
            "tags (--tag-corners-base / -n on collect_apriltag_corners.py) to fix this."
        )

    image_points_by_tag: dict[int, np.ndarray] = {}
    per_tag_pixel_std: dict[int, list[float]] = {}
    for tag_id in tags_used:
        entries = per_tag_pixels[tag_id]
        stacked = np.stack([c for _, c in entries], axis=0)  # (n, 4, 2)
        image_points_by_tag[tag_id] = stacked.mean(axis=0)
        per_tag_pixel_std[tag_id] = np.linalg.norm(stacked.std(axis=0), axis=1).tolist()

    used_top_lefts = {t: top_lefts[t] for t in tags_used}
    rvec, tvec, yaw_by_tag, handedness = solve_joint_pose_and_yaws(
        used_top_lefts, tag_size, image_points_by_tag, K, dist, args.pnp_method, args.yaw_grid_step_deg
    )
    method_used = args.pnp_method

    object_points = np.concatenate(
        [corners_from_top_left(used_top_lefts[t], tag_size, np.deg2rad(yaw_by_tag[t]), handedness) for t in tags_used],
        axis=0,
    )
    image_points = np.concatenate([image_points_by_tag[t] for t in tags_used], axis=0)
    _, per_point_err, rmse_px = reprojection_error(object_points, image_points, K, dist, rvec, tvec)

    # Diagnostic: leave-one-out consistency check, catches bad per-tag data (e.g. the
    # physically touched corner not matching the detector's corner 0) that the full
    # joint solve's own per-point residuals can miss -- see leave_one_out_tag_diagnostics'
    # docstring for why. Only meaningful with >=3 tags (need >=2 left per exclusion).
    loo_rmse_by_tag: dict[int, float | None] = {}
    suspect_tags: list[int] = []
    if len(tags_used) >= 3:
        loo_rmse_by_tag = leave_one_out_tag_diagnostics(
            used_top_lefts, tags_used, tag_size, image_points_by_tag, K, dist, args.pnp_method, args.yaw_grid_step_deg
        )
        successful = {t: v for t, v in loo_rmse_by_tag.items() if v is not None}
        failed = [t for t, v in loo_rmse_by_tag.items() if v is None]
        if successful:
            med = float(np.median(list(successful.values())))
            suspect_tags = [t for t, v in successful.items() if v > 3.0 * med and v > 3.0]
        if suspect_tags:
            print(
                f"[warn] tag(s) {suspect_tags}: leave-one-out check -- this tag's data doesn't fit "
                "the camera pose solved from every OTHER tag nearly as well as the other tags do "
                "(see suspect_anchor_tags / per-tag leave-one-out RMSE in the output JSON). Likely the "
                "physically touched corner doesn't match the detector's corner 0, or that tag's "
                "top_left was mismeasured. Re-check with annotate_apriltags.py's red-dot-labeled "
                "corner 0 and re-collect that tag with collect_apriltag_corners.py."
            )
        if failed:
            print(
                f"[warn] tag(s) {failed}: leave-one-out pose solve failed when this tag was held out "
                "-- the REMAINING tags' data (which does include a possibly-bad tag, just not this "
                "one) couldn't produce a valid camera-above-table pose. If a tag is also flagged above "
                "as suspect, that's the more likely culprit; if not, check whether the remaining tags "
                "are too close to collinear."
            )

    R_mat, _ = cv2.Rodrigues(rvec)
    T_base_in_cam = np.eye(4, dtype=np.float64)
    T_base_in_cam[:3, :3] = R_mat
    T_base_in_cam[:3, 3] = tvec.flatten()
    T_cam_in_base = np.linalg.inv(T_base_in_cam)

    # The solved tag yaw and chirality determine all four base-frame corners from the
    # physically measured top-left corner and the known tag side length.
    corner_labels = ["top_left", "top_right", "bottom_right", "bottom_left"]
    tag_corners_base: dict[int, dict[str, list[float]]] = {}
    for t in tags_used:
        corners = corners_from_top_left(used_top_lefts[t], tag_size, np.deg2rad(yaw_by_tag[t]), handedness)
        tag_corners_base[t] = {lbl: corners[i].tolist() for i, lbl in enumerate(corner_labels)}

    # Per-image diagnostics reproject the solved pose into each raw frame and, when a
    # frame alone has >=4 points, solve PnP standalone to check consistency (large deltas
    # usually mean the camera moved between shots or a frame's detection is bad).
    r_global = R.from_matrix(R_mat)
    per_image_diag = []
    for entry in per_image_data:
        path = entry["path"]
        detections = entry["detections"]
        known = {tid: c for tid, c in detections.items() if tid in used_top_lefts}
        if not known:
            per_image_diag.append({"image": str(path), "tags_detected": [], "reprojection_rmse_px": None})
            continue

        obj_here = np.concatenate(
            [
                corners_from_top_left(used_top_lefts[tid], tag_size, np.deg2rad(yaw_by_tag[tid]), handedness)
                for tid in sorted(known)
            ],
            axis=0,
        )
        img_here = np.concatenate([known[tid] for tid in sorted(known)], axis=0)
        proj_here, _, rmse_here = reprojection_error(obj_here, img_here, K, dist, rvec, tvec)

        diag = {
            "image": str(path),
            "tags_detected": sorted(known.keys()),
            "reprojection_rmse_px": rmse_here,
            "standalone_rotation_delta_deg": None,
            "standalone_translation_delta_mm": None,
        }
        if len(known) * 4 >= 4 and img_here.shape[0] >= 4:
            try:
                rvec_i, tvec_i, _ = solve_pnp(obj_here, img_here, K, dist, args.pnp_method)
                r_i = R.from_matrix(cv2.Rodrigues(rvec_i)[0])
                rot_delta_deg = float(np.degrees((r_global.inv() * r_i).magnitude()))
                trans_delta_mm = float(np.linalg.norm(tvec_i.flatten() - tvec.flatten()) * 1000.0)
                diag["standalone_rotation_delta_deg"] = rot_delta_deg
                diag["standalone_translation_delta_mm"] = trans_delta_mm
            except RuntimeError:
                pass
        per_image_diag.append(diag)

        if not args.no_debug_images:
            proj_by_tag = {}
            offset = 0
            for tid in sorted(known):
                proj_by_tag[tid] = proj_here[offset : offset + 4]
                offset += 4
            vis = draw_debug_image(entry["img"], detections, proj_by_tag)
            out_name = f"{path.stem}_reprojection.png"
            cv2.imwrite(str(debug_dir / out_name), vis)

    metadata = {
        "arm": args.arm,
        "images": [str(p) for p in image_paths],
        "tag_family": args.tag_family,
        "tag_detector_backend": detector_backend,
        "tag_corners_base_source": args.tag_corners_base,
        "tags_meta": tags_meta,
        "tag_size_m": tag_size,
        "solved_yaw_deg_by_tag": yaw_by_tag,
        "solved_chirality": handedness,
        "intrinsics": intrinsics_meta,
        "tags_used": tags_used,
        "tags_missing": tags_missing,
        "num_correspondences": int(object_points.shape[0]),
        "pnp_method_requested": args.pnp_method,
        "pnp_method_used": method_used,
        "reprojection_rmse_px": rmse_px,
        "per_point_reprojection_error_px": per_point_err.tolist(),
        "per_tag_pixel_detection_std_px": per_tag_pixel_std,
        "leave_one_out_rmse_px_by_tag": loo_rmse_by_tag,
        "suspect_tags": suspect_tags,
        "per_image_diagnostics": per_image_diag,
    }

    result = {
        "metadata": metadata,
        "T_cam_in_base": matrix_record(T_cam_in_base),
        "T_base_in_cam": matrix_record(T_base_in_cam),
        "camera_position_in_base_m": T_cam_in_base[:3, 3].tolist(),
        "tag_corners_base": {
            "description": "Each tag's 4 corners in the robot base frame (meters), computed from "
            "the measured top_left corner, tag size, and Method A's solved yaw and chirality.",
            "corner_order": corner_labels,
            "corners": {str(t): tag_corners_base[t] for t in tags_used},
        },
    }

    out_path = out_dir / f"camera_extrinsics_arx_{args.arm}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.no_update_yaml:
        update_yaml_from_result(args.calibration_yaml, out_path, args.arm)
        print(f"Updated {Path(args.calibration_yaml).resolve()} for {args.arm} arm")

    print(f"Wrote {out_path}")
    if not args.no_debug_images:
        print(f"Debug images in {debug_dir}")
    print(f"Tags used: {tags_used}  (missing/not detected: {tags_missing})")
    print(f"PnP method: {method_used}  correspondences: {object_points.shape[0]}")
    print()
    print("Method A (yaw-solve):")
    print(f"  Solved yaw per tag (deg): {yaw_by_tag}  chirality: {handedness:+.0f}")
    print(f"  Reprojection RMSE: {rmse_px:.3f} px")
    print(f"  Camera position in base frame (m): {T_cam_in_base[:3, 3].tolist()}")
    for diag in per_image_diag:
        if diag["standalone_rotation_delta_deg"] is not None:
            print(
                f"  {Path(diag['image']).name}: rmse={diag['reprojection_rmse_px']:.3f}px "
                f"standalone_delta=({diag['standalone_rotation_delta_deg']:.3f}deg, "
                f"{diag['standalone_translation_delta_mm']:.2f}mm)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate RealSense camera extrinsics (in robot base frame) from N AprilTags "
        "(N >= 1, more tags reduces reprojection error).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arm",
        required=True,
        choices=("left", "right"),
        help="Arm-base frame used by --tag-corners-base and destination YAML field.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=[],
        help="One or more existing RGB frames from the static camera. Optional if --capture is "
        "given (in which case captured frame(s) are used in addition to any --images); "
        "required otherwise.",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Take a fresh photo with the camera instead of (or in addition to) --images.",
    )
    parser.add_argument(
        "--capture-backend",
        choices=["realsense", "opencv"],
        default="realsense",
        help="Camera backend for --capture (default: realsense / pyrealsense2).",
    )
    parser.add_argument(
        "--camera-serial",
        default=ARX_HEAD_CAMERA_SERIAL,
        help=f"RealSense serial for --capture-backend realsense (default: ARX head D405 {ARX_HEAD_CAMERA_SERIAL}).",
    )
    parser.add_argument("--camera-index", type=int, default=1, help="OpenCV camera index for --capture-backend opencv.")
    parser.add_argument("--capture-width", type=int, default=1280)
    parser.add_argument("--capture-height", type=int, default=720)
    parser.add_argument(
        "--capture-fps",
        type=int,
        default=30,
        help="Default 30: ARX head D405 at 1280x720. If "
        "pipeline.start() fails, the error lists this device's supported width/height/fps combos.",
    )
    parser.add_argument("--capture-count", type=int, default=1, help="Number of frames to capture and average.")
    parser.add_argument("--capture-interval", type=float, default=0.5, help="Seconds between frames when capture-count > 1.")
    parser.add_argument(
        "--capture-warmup-frames",
        type=int,
        default=30,
        help="Frames to discard before capturing, so auto-exposure/white-balance settle.",
    )
    parser.add_argument(
        "--tag-corners-base",
        required=True,
        help="JSON with each tag's top_left corner position in the robot base frame (meters) "
        "and tag_size_m -- see collect_apriltag_corners.py.",
    )
    parser.add_argument(
        "--tag-size-m",
        type=float,
        default=None,
        help="Override the tag side length (meters) instead of reading tag_size_m from --tag-corners-base.",
    )
    parser.add_argument(
        "--yaw-grid-step-deg",
        type=float,
        default=15.0,
        help="Coarse grid resolution (degrees) for the initial per-tag yaw search before local refinement.",
    )
    parser.add_argument(
        "--tag-family",
        default="36h11",
        help="Tag family (default: 36h11). Short names: 16h5/25h9/36h10/36h11 (cv2.aruco) or "
        "41h12 (pupil_apriltags -- pip install pupil-apriltags; cv2.aruco doesn't ship it). "
        "Full names (DICT_APRILTAG_36h11, tagStandard41h12, ...) also accepted.",
    )
    parser.add_argument(
        "--intrinsics-json",
        default=None,
        help="JSON with fx/fy/cx(ppx)/cy(ppy)/dist_coeffs(coeffs). RealSense capture reads the "
        "active stream profile by default; other inputs fall back to the saved ARX head D405 1280x720 profile.",
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help="Resolution key (e.g. '1280x720') to select from the intrinsics source's "
        "by_resolution map (the ARX head D405 file by default, or "
        "--intrinsics-json's own map if given). Usually not needed: auto-set from "
        "--capture-width/--capture-height when --capture is used, otherwise from the first "
        "--images frame's actual pixel size.",
    )
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--dist-coeffs", nargs="*", type=float, default=None, help="OpenCV order: k1 k2 p1 p2 k3 ...")
    parser.add_argument("--pnp-method", choices=list(PNP_METHOD_ATTRS.keys()), default="sqpnp")
    parser.add_argument("--out", default="outputs/camera_extrinsics")
    parser.add_argument("--no-debug-images", action="store_true")
    parser.add_argument(
        "--calibration-yaml",
        default=str(DEFAULT_YAML),
        help="TATE RealSense YAML to update after a successful solve.",
    )
    parser.add_argument(
        "--no-update-yaml",
        action="store_true",
        help="Write the result JSON and diagnostics without modifying the calibration YAML.",
    )
    args = parser.parse_args()
    if not args.images and not args.capture:
        parser.error("pass --images (existing photo(s)), --capture (take a new photo), or both")
    calibrate(args)


if __name__ == "__main__":
    main()
