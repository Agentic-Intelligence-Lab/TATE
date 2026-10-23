#!/usr/bin/env python3
"""Plot the hanging_1016 episode 34 correction overshoot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "evaluation/events.py").is_file())))
from evaluation.events import extract_events
from evaluation.loaders import load_eef_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/data/xule/TATE"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    base = args.root / "outputs/experiments/hanging_1016/artifacts/variants"
    paths = {
        "raw": base / "finger_center_hys085__none/episode_000034/eef.json",
        "corrected": base / "finger_center_hys085__position_rotation/episode_000034/eef.json",
    }
    trajectories = {k: load_eef_json(v, episode_id="34") for k, v in paths.items()}
    poses = {k: t.sides["right"].pose_xyzw[:, :3] * 100 for k, t in trajectories.items()}
    events = extract_events(trajectories["raw"], "right")
    event_frames = [e.frame_index for e in events]
    anchors = [0, *event_frames[:3], len(poses["raw"])-1]
    displacement = np.linalg.norm(poses["corrected"] - poses["raw"], axis=1) * 10
    max_frame = int(np.argmax(displacement))

    fig = plt.figure(figsize=(17, 8.5), facecolor="white")
    grid = fig.add_gridspec(2, 3, left=.06, right=.98, bottom=.09, top=.84, wspace=.27, hspace=.34)
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    projections = [(0,1,"XY / top view"), (0,2,"XZ / side view"), (1,2,"YZ / front view")]
    axes = [fig.add_subplot(grid[0,1]), fig.add_subplot(grid[0,2]), fig.add_subplot(grid[1,0])]
    colors = {"raw":"#dd7844", "corrected":"#087f9b"}
    for key in ["raw", "corrected"]:
        p = poses[key]
        ax3.plot(*p.T, color=colors[key], lw=1.5, label=key.capitalize())
        for ax, (a,b,_) in zip(axes, projections):
            ax.plot(p[:,a], p[:,b], color=colors[key], lw=1.5, label=key.capitalize())
    for i, frame in enumerate(anchors):
        p = poses["corrected"][frame]
        ax3.scatter(*p, color="#242424", s=22, marker="o", zorder=5)
        for ax, (a,b,_) in zip(axes, projections):
            ax.scatter(p[a], p[b], color="#242424", s=20, zorder=5)
    ax3.scatter(*poses["corrected"][max_frame], color="#c23628", s=50, marker="x")
    ax3.set(xlabel="X / forward (cm)", ylabel="Y / left (cm)", zlabel="Z / up (cm)")
    ax3.set_title("3D trajectory", loc="left"); ax3.legend(frameon=False)
    ax3.view_init(elev=24, azim=-58)
    for ax, (a,b,title) in zip(axes, projections):
        ax.set(xlabel=f"{'XYZ'[a]} (cm)", ylabel=f"{'XYZ'[b]} (cm)", title=title)
        ax.grid(alpha=.18); ax.spines[["top","right"]].set_visible(False)
        ax.set_aspect("equal", adjustable="box")

    axd = fig.add_subplot(grid[1,1:])
    axd.plot(displacement, color="#087f9b", lw=1.7)
    for frame in anchors:
        axd.axvline(frame, color="#555", lw=.8, ls="--", alpha=.55)
    axd.scatter(max_frame, displacement[max_frame], color="#c23628", s=42, zorder=5)
    axd.annotate(f"max {displacement[max_frame]:.1f} mm @ frame {max_frame}",
                 (max_frame, displacement[max_frame]), xytext=(8,-22), textcoords="offset points")
    axd.set(xlabel="Source frame", ylabel="Correction displacement (mm)",
            title="Min-bending displacement; dashed lines are start, first 3 events, end")
    axd.grid(alpha=.18); axd.spines[["top","right"]].set_visible(False)
    fig.suptitle("Hanging 1016 / source episode 34: correction overshoot", fontsize=17, weight="bold", y=.97)
    fig.text(.5, .90,
             "Detected events: open-close-open-close-open; correction anchors the first three events",
             ha="center", fontsize=10)
    fig.text(.5, .025, "Black circles: correction anchors | red ×: maximum displacement", ha="center", fontsize=9)
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / f"hanging_episode34_overshoot.{ext}", dpi=180)
    plt.close(fig)

    audit = {
        "source_episode_id": 34,
        "detected_events": [{"frame": e.frame_index, "transition": e.transition} for e in events],
        "anchor_frames": anchors,
        "anchor_displacement_mm": displacement[anchors].tolist(),
        "max_anchor_displacement_mm": float(displacement[anchors].max()),
        "max_displacement_mm": float(displacement[max_frame]),
        "max_displacement_frame": max_frame,
        "overshoot_ratio": float(displacement[max_frame] / displacement[anchors].max()),
        "raw_min_cm": poses["raw"].min(0).tolist(), "raw_max_cm": poses["raw"].max(0).tolist(),
        "corrected_min_cm": poses["corrected"].min(0).tolist(),
        "corrected_max_cm": poses["corrected"].max(0).tolist(),
    }
    (args.output / "hanging_episode34_overshoot.json").write_text(json.dumps(audit, indent=2)+"\n")


if __name__ == "__main__":
    main()
