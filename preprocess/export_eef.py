"""Export dual-arm EEF targets from a cached WiLoR reconstruction."""

from __future__ import annotations

import argparse

from preprocess.Hand2Gripper import MODES
from preprocess.PipelineIO import export_eef_from_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", required=True, help="Sequence-level wilor_hands.json cache")
    parser.add_argument("--out", required=True, help="Output eef.json")
    parser.add_argument(
        "--camera-calibration",
        default="./cfg/preprocess/base/RealSenseD405.yaml",
        help="Camera intrinsics/extrinsics YAML",
    )
    parser.add_argument(
        "--eef-config",
        default="./cfg/preprocess/base/EEFExport.yaml",
        help="hand2gripper mode, grasp hysteresis, and TCP export configuration",
    )
    parser.add_argument(
        "--hand2gripper-mode", choices=sorted(MODES),
        default=None,
        help="Override hand2gripper.mode in --eef-config",
    )
    parser.add_argument(
        "--grasp-mode",
        choices=sorted(MODES),
        default=None,
        help="Override the gripper-state signal independently of EEF pose mode",
    )
    parser.add_argument("--grasp-close-ratio", type=float, default=None)
    parser.add_argument("--grasp-open-ratio", type=float, default=None)
    parser.add_argument("--grasp-min-frames", type=int, default=None)
    parser.add_argument(
        "--arm-mode", choices=("single_arm", "bimanual"), default=None
    )
    parser.add_argument(
        "--active-sides", nargs="+", choices=("left", "right"), default=None
    )
    args = parser.parse_args()
    payload = export_eef_from_cache(
        args.hands,
        args.camera_calibration,
        args.out,
        args.eef_config,
        grasp_close_ratio=args.grasp_close_ratio,
        grasp_open_ratio=args.grasp_open_ratio,
        grasp_min_frames=args.grasp_min_frames,
        hand2gripper_mode=args.hand2gripper_mode,
        grasp_mode=args.grasp_mode,
        arm_mode=args.arm_mode,
        active_sides=args.active_sides,
    )
    valid_r = sum(frame["hand_r"] is not None for frame in payload["frames"])
    valid_l = sum(frame["hand_l"] is not None for frame in payload["frames"])
    print(f"[Output] EEF JSON saved to: {args.out}")
    print(f"[Output] Valid right/left frames: {valid_r}/{valid_l} of {payload['total_frames']}")


if __name__ == "__main__":
    main()
