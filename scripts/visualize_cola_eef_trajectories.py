#!/usr/bin/env python3
"""Visualize real, uncorrected ego, and corrected ego TCP trajectories."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DEFAULT_ROOT = Path("/mnt/data/xule/TATE")
GROUPS = (
    ("Real robot", "real", "#222222", "-"),
    ("Ego before correction", "none", "#E76F51", "-"),
    ("Ego after position + rotation", "posrot", "#168AAD", "--"),
)


def _load_real(path: Path, side: str) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(
        [frame["arms"][side]["T_tcp_in_output_frame"] for frame in payload["frames"]],
        dtype=np.float64,
    )[:, :3, 3]


def _load_ego(path: Path, side: str) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "hand_l" if side == "left" else "hand_r"
    poses = [
        frame[key]["tcp_pose_eef_frame"]
        for frame in payload["frames"]
        if frame.get(key) is not None
    ]
    if len(poses) < 2:
        raise ValueError(f"{path} has fewer than two valid {side} poses")
    return np.asarray(poses, dtype=np.float64)[:, :3, 3]


def _resample(path: np.ndarray, count: int = 200) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(path))
    target = np.linspace(0.0, 1.0, count)
    return np.column_stack([np.interp(target, source, path[:, axis]) for axis in range(3)])


def _load_groups(
    root: Path,
    task: str,
    experiment: str,
    side: str,
    real_count: int,
    ego_count: int,
) -> dict[str, list[np.ndarray]]:
    real_files = sorted(
        glob.glob(str(root / f"outputs/arx_real_flange/{task}_arx/episode_*.json"))
    )[:real_count]
    variants = root / f"outputs/experiments/{experiment}/artifacts/variants"
    none_files = sorted(glob.glob(str(variants / "finger_center_hys085__none/episode_*/eef.json")))[:ego_count]
    posrot_files = sorted(
        glob.glob(str(variants / "finger_center_hys085__position_rotation/episode_*/eef.json"))
    )[:ego_count]
    expected = {
        "real": (real_files, real_count),
        "none": (none_files, ego_count),
        "posrot": (posrot_files, ego_count),
    }
    for name, (files, count) in expected.items():
        if len(files) != count:
            raise ValueError(f"expected {count} {name} episodes, found {len(files)}")
    return {
        "real": [_load_real(Path(path), side) for path in real_files],
        "none": [_load_ego(Path(path), side) for path in none_files],
        "posrot": [_load_ego(Path(path), side) for path in posrot_files],
    }


def _limits(groups: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate([path for paths in groups.values() for path in paths]) * 100.0
    lo, hi = points.min(axis=0), points.max(axis=0)
    padding = np.maximum((hi - lo) * 0.06, 0.8)
    return lo - padding, hi + padding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--task", choices=("cola", "drawer"), default="cola")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--real-count", type=int, default=20)
    parser.add_argument("--ego-count", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment or ("drawer_1" if args.task == "drawer" else args.task)

    trajectories = {
        side: _load_groups(
            args.root.resolve(), args.task, experiment, side, args.real_count, args.ego_count
        )
        for side in ("left", "right")
    }
    fig = plt.figure(figsize=(20, 11.5))
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.055, top=0.82, wspace=0.24, hspace=0.24)
    axes: dict[tuple[str, int], plt.Axes] = {}
    projections = ((0, 1, "XY projection"), (0, 2, "XZ projection"), (1, 2, "YZ projection"))

    for row, side in enumerate(("left", "right")):
        groups = trajectories[side]
        lo, hi = _limits(groups)
        ax3d = fig.add_subplot(2, 4, row * 4 + 1, projection="3d")
        axes[(side, 0)] = ax3d
        projection_axes = [fig.add_subplot(2, 4, row * 4 + index + 2) for index in range(3)]
        for index, axis in enumerate(projection_axes, start=1):
            axes[(side, index)] = axis

        for _, key, color, linestyle in GROUPS:
            paths_cm = [path * 100.0 for path in groups[key]]
            for path in paths_cm:
                ax3d.plot(*path.T, color=color, alpha=0.075, linewidth=0.55)
                for ax, (a, b, _) in zip(projection_axes, projections):
                    ax.plot(path[:, a], path[:, b], color=color, alpha=0.07, linewidth=0.5)

            mean_path = np.mean([_resample(path) for path in paths_cm], axis=0)
            ax3d.plot(*mean_path.T, color=color, linestyle=linestyle, linewidth=2.8)
            ax3d.scatter(*mean_path[0], color=color, marker="o", s=38, edgecolors="white", linewidths=0.6)
            ax3d.scatter(*mean_path[-1], color=color, marker="s", s=38, edgecolors="white", linewidths=0.6)
            for ax, (a, b, _) in zip(projection_axes, projections):
                ax.plot(mean_path[:, a], mean_path[:, b], color=color, linestyle=linestyle, linewidth=2.6)
                ax.scatter(mean_path[0, a], mean_path[0, b], color=color, marker="o", s=34,
                           edgecolors="white", linewidths=0.6, zorder=5)
                ax.scatter(mean_path[-1, a], mean_path[-1, b], color=color, marker="s", s=34,
                           edgecolors="white", linewidths=0.6, zorder=5)

        arm_name = "Left arm" if side == "left" else "Right arm"
        ax3d.set(
            title=f"{arm_name}: 3D TCP trajectories",
            xlabel="X (cm)", ylabel="Y (cm)", zlabel="Z (cm)",
            xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
        )
        ax3d.set_box_aspect(hi - lo)
        ax3d.view_init(elev=25, azim=-58)
        if side == "left":
            ax3d.text2D(
                0.02, 0.02,
                "Left position correction is absent:\nbefore/after paths coincide",
                transform=ax3d.transAxes, fontsize=9, color="#8A3E2A",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#E6B8A9", "alpha": 0.9},
            )
        for ax, (a, b, title) in zip(projection_axes, projections):
            ax.set(
                title=f"{arm_name}: {title}",
                xlabel=f"{'XYZ'[a]} (cm)", ylabel=f"{'XYZ'[b]} (cm)",
                xlim=(lo[a], hi[a]), ylim=(lo[b], hi[b]),
            )
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)

    legend = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2.8, label=label)
        for label, _, color, linestyle in GROUPS
    ]
    legend += [
        Line2D([0], [0], color="#777777", alpha=0.35, linewidth=0.8, label="Individual episode"),
        Line2D([0], [0], marker="o", color="#777777", linestyle="", label="Mean start"),
        Line2D([0], [0], marker="s", color="#777777", linestyle="", label="Mean end"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.885))
    fig.suptitle(
        f"{args.task.capitalize()} task: real robot vs. ego EEF trajectories before and after correction\n"
        f"{args.real_count} real episodes and {args.ego_count} ego episodes; thin lines are episodes, "
        "thick lines are normalized-time means; per-arm zero-flange frames",
        fontsize=15, y=0.985,
    )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    summary = {
        side: {key: len(paths) for key, paths in groups.items()}
        for side, groups in trajectories.items()
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"png": str(output), "pdf": str(output.with_suffix('.pdf')), "episodes": summary}, indent=2))


if __name__ == "__main__":
    main()
