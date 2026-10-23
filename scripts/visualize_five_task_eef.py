#!/usr/bin/env python3
"""Plot measured real TCP and exported ego TCP labels, with source-ID auditing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pyarrow.parquet as pq

TASKS = [
    ("cube", "stack_redcube_v1", "stack_redcube_v1_1"),
    ("hanging", "hanging", "hanging_1"),
    ("bowl", "bowl", "bowl_1"),
    ("cola", "cola", "cola_1012"),
    ("drawer", "drawer", "drawer_1012"),
]
GROUPS = [("real", "Real robot", "#303641", "-"),
          ("before", "Ego before", "#dd7844", "-"),
          ("after", "Ego after", "#087f9b", "--")]
LABELS = ["X / forward (cm)", "Y / left (cm)", "Z / up (cm)"]
PROJECTIONS = [(0, 1), (0, 2), (1, 2)]


def read_ego(root):
    info = json.loads((root / "meta/info.json").read_text())
    assert info["eef_coordinate_convention"]["pose_semantics"] == "arx_tcp"
    sides = info["tate"]["active_sides"]
    cols = ["episode_index", "tate.source_episode_index", "tate.source_frame_index"]
    for side in sides:
        assert info["features"][f"tate.eef.{side}.pose"]["frame"] == f"{side}_flange_zero"
        cols += [f"tate.eef.{side}.pose", f"tate.eef.{side}.valid"]
    rows = []
    for file in sorted((root / "data").rglob("*.parquet")):
        rows.extend(pq.read_table(file, columns=cols).to_pylist())
    assert len(rows) == info["total_frames"]
    assert len({r['episode_index'] for r in rows}) == info['total_episodes']
    episodes = {}
    for row in rows:
        episodes.setdefault(int(row["tate.source_episode_index"]), []).append(row)
    result = {}
    for eid, records in sorted(episodes.items()):
        records.sort(key=lambda r: r["tate.source_frame_index"])
        indices = np.array([r["tate.source_frame_index"] for r in records], dtype=int)
        assert len(np.unique(indices)) == len(indices), (root, eid, "duplicate source frame")
        poses = {}
        for side in sides:
            p = np.array([r[f"tate.eef.{side}.pose"][:3] for r in records], dtype=float) * 100
            valid = np.array([r[f"tate.eef.{side}.valid"] for r in records], dtype=bool)
            valid &= np.isfinite(p).all(axis=1)
            p[~valid] = np.nan
            poses[side] = p
        result[eid] = {"frames": indices, "poses": poses,
                       "export_ids": sorted({int(r["episode_index"]) for r in records})}
    return result, info


def read_real(root, sides):
    result = {}
    for file in sorted(root.glob("episode_*.json")):
        data = json.loads(file.read_text())
        eid = int(data["source"]["episode_index"])
        assert eid not in result
        poses = {}
        for side in sides:
            assert data["coordinate_convention"]["output_frames"][side] == f"{side}_flange_zero"
            poses[side] = np.array([f["arms"][side]["T_tcp_in_output_frame"]
                                    for f in data["frames"]], dtype=float)[:, :3, 3] * 100
        result[eid] = {"frames": np.array([f["frame_index"] for f in data["frames"]]),
                       "poses": poses, "file": str(file)}
    assert result, root
    return result


def points(episodes, side):
    p = np.concatenate([e["poses"][side] for e in episodes.values()])
    return p[np.isfinite(p).all(axis=1)]


def matched_groups(groups, sides):
    matched = {"real": groups["real"], "before": {}, "after": {}}
    for eid in sorted(groups["before"].keys() & groups["after"].keys()):
        b, a = groups["before"][eid], groups["after"][eid]
        common, bi, ai = np.intersect1d(b["frames"], a["frames"], return_indices=True)
        for key, e, idx in [("before", b, bi), ("after", a, ai)]:
            poses = {}
            for side in sides:
                valid = np.isfinite(b["poses"][side][bi]).all(1) & np.isfinite(a["poses"][side][ai]).all(1)
                p = e["poses"][side][idx].copy()
                p[~valid] = np.nan
                poses[side] = p
            matched[key][eid] = {"frames": common, "poses": poses}
    return matched


def path_with_gaps(episode, side):
    p = episode["poses"][side]
    gaps = np.where(np.diff(episode["frames"]) != 1)[0] + 1
    return np.insert(p, gaps, np.nan, axis=0)


def draw_row(fig, grid, row, task, side, groups, selection, mode):
    ax3 = fig.add_subplot(grid[row, 0], projection="3d")
    axes = [fig.add_subplot(grid[row, i]) for i in range(1, 4)]
    all_points = {key: points(episodes, side) for key, episodes in groups.items()}
    if mode == "single":
        visible = np.concatenate([groups[k][selection[k]]["poses"][side] for k in groups])
        visible = visible[np.isfinite(visible).all(1)]
    else:
        visible = np.concatenate(list(all_points.values()))
    lo, hi = visible.min(0), visible.max(0)
    padding = np.maximum((hi - lo) * .06, .8)
    lo, hi = lo - padding, hi + padding
    for key, label, color, style in GROUPS:
        if mode == "single":
            p = path_with_gaps(groups[key][selection[key]], side)
            valid = p[np.isfinite(p).all(1)]
            ax3.plot(*p.T, color=color, ls=style, lw=1.65, alpha=.9)
            for index, marker in [(0, "o"), (-1, "s")]:
                ax3.scatter(*valid[index], s=35, marker=marker, color=color, edgecolors="white", linewidths=.6)
            for ax, (a, b) in zip(axes, PROJECTIONS):
                ax.plot(p[:, a], p[:, b], color=color, ls=style, lw=1.6, alpha=.9)
                for index, marker in [(0, "o"), (-1, "s")]:
                    ax.scatter(valid[index, a], valid[index, b], s=30, marker=marker,
                               color=color, edgecolors="white", linewidths=.5, zorder=5)
        else:
            p = all_points[key]
            ax3.scatter(*p.T, color=color, s=.65, alpha=.13, linewidths=0,
                        depthshade=False, rasterized=True)
            for dim, ax in enumerate(axes):
                bins = np.linspace(lo[dim], hi[dim], 61)
                hist, edges = np.histogram(p[:, dim], bins=bins, density=True)
                ax.stairs(hist, edges, color=color, lw=1.5, ls=style)
                ax.stairs(hist, edges, color=color, fill=True, alpha=.07, lw=0)
    arm = side.capitalize()
    if mode == "single":
        detail = f"real {selection['real']} | ego source {selection['before']}"
    else:
        detail = "episodes R/B/A: " + "/".join(str(len(groups[k])) for k, *_ in GROUPS)
    ax3.set_title(f"{task.capitalize()} / {arm}\n{detail}", fontsize=11, loc="left", pad=8)
    ax3.set(xlabel=LABELS[0], ylabel=LABELS[1], zlabel=LABELS[2],
            xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
    ax3.set_box_aspect(hi - lo)
    ax3.view_init(elev=24, azim=-58)
    ax3.tick_params(labelsize=8, pad=0)
    for dim, ax in enumerate(axes):
        if mode == "single":
            a, b = PROJECTIONS[dim]
            ax.set(xlabel=LABELS[a], ylabel=LABELS[b], xlim=(lo[a], hi[a]), ylim=(lo[b], hi[b]))
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(["XY / top view", "XZ / side view", "YZ / front view"][dim], fontsize=10)
        else:
            ax.set(xlabel=LABELS[dim], ylabel="Probability density (1/cm)", xlim=(lo[dim], hi[dim]))
            ax.set_ylim(bottom=0)
            ax.set_title(f"{'XYZ'[dim]} distribution / all valid frames", fontsize=10)
            if dim == 0:
                counts = "  ".join(f"{k[0].upper()}: {len(all_points[k]):,}" for k, *_ in GROUPS)
                ax.text(.02, .98, f"Frames {counts}", transform=ax.transAxes, va="top", fontsize=8)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)


def render(tasks, output, mode, matched=False):
    rows = [(t, s) for t in tasks for s in t["sides"]]
    height = 3.65 * len(rows) + 1.75
    fig = plt.figure(figsize=(19, height), facecolor="white")
    grid = fig.add_gridspec(len(rows), 4, left=.045, right=.982, bottom=.9/height,
                           top=1 - 1.35/height, wspace=.32, hspace=.48)
    for row, (t, side) in enumerate(rows):
        groups = t["matched"] if matched else t["groups"]
        draw_row(fig, grid, row, t["name"], side, groups, t["selection"], mode)
    title = "Single-episode TCP trajectories" if mode == "single" else "Full-dataset TCP position distributions"
    if matched:
        title = "TCP distributions / matched ego source frames"
    fig.suptitle(title, fontsize=19, y=1 - .12/height, weight="bold")
    subtitle = "X forward / Y left / Z up | centimeters | each arm in its own zero-flange frame"
    fig.text(.5, 1 - .55/height, subtitle, ha="center", fontsize=11)
    handles = [Line2D([0], [0], color=c, ls=s, lw=2, label=l) for _, l, c, s in GROUPS]
    if mode == "single":
        handles += [Line2D([0], [0], color="#666666", marker=m, ls="", label=l)
                    for m, l in [("o", "Start"), ("s", "End")]]
        note = "Smallest eligible source ID; before/after share source episode. Real sample is independent; no temporal alignment."
    else:
        note = "All valid frames; no subsampling or outlier trimming. Each R/B/A density integrates to 1; longer episodes carry more weight."
        if matched:
            note = "Ego before/after use identical source frames and joint validity per arm; real reference remains the full dataset."
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(.5, 1 - .67/height), fontsize=11)
    fig.text(.5, .10/height, note, ha="center", fontsize=9, color="#555555")
    for ext in ["png", "pdf"]:
        fig.savefig(output.with_suffix('.' + ext), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/data/xule/TATE"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    tasks, audit = [], {}
    for name, real_name, experiment in TASKS:
        base = args.root / "outputs/experiments" / experiment / "datasets"
        before_path = base / "finger_center_hys085__none"
        after_path = base / "finger_center_hys085__position_rotation"
        before, bi = read_ego(before_path)
        after, ai = read_ego(after_path)
        sides = bi["tate"]["active_sides"]
        assert sides == ai["tate"]["active_sides"]
        real_path = args.root / "outputs/arx_real_flange" / (real_name + "_arx")
        real = read_real(real_path, sides)
        groups = dict(real=real, before=before, after=after)
        eligible = lambda e: all(np.isfinite(e["poses"][s]).all(1).sum() >= 2 for s in sides)
        ego_id = min(i for i in before.keys() & after.keys() if eligible(before[i]) and eligible(after[i]))
        selection = dict(real=min(i for i in real if eligible(real[i])), before=ego_id, after=ego_id)
        matched = matched_groups(groups, sides)
        task = dict(name=name, sides=sides, groups=groups, matched=matched, selection=selection)
        tasks.append(task)
        audit[name] = dict(paths=dict(real=str(real_path), before=str(before_path), after=str(after_path)),
                           sides=sides, selected_source_ids=selection,
                           selected_export_ids={k: groups[k][ego_id]["export_ids"] for k in ["before", "after"]},
                           missing_after_source_ids=sorted(before.keys() - after.keys()),
                           added_after_source_ids=sorted(after.keys() - before.keys()), groups={})
        for key, episodes in groups.items():
            audit[name]["groups"][key] = dict(episodes=len(episodes), source_ids=sorted(episodes), arms={})
            for side in sides:
                p = points(episodes, side)
                audit[name]["groups"][key]["arms"][side] = dict(valid_frames=len(p),
                    invalid_frames=sum(len(e["frames"]) for e in episodes.values())-len(p),
                    min_cm=p.min(0).tolist(), max_cm=p.max(0).tolist(), mean_cm=p.mean(0).tolist())
        audit[name]["matched_valid_frames"] = {s: len(points(matched["before"], s)) for s in sides}
        for side in sides:
            assert len(points(matched['before'], side)) == len(points(matched['after'], side))
        print(json.dumps({name: {"episodes": {k: len(v) for k,v in groups.items()}, "selection": selection}}), flush=True)
        render([task], args.output / f"{name}_single", "single")
        render([task], args.output / f"{name}_distribution", "distribution")
    render(tasks, args.output / "01_single_episodes", "single")
    render(tasks, args.output / "02_full_distributions", "distribution")
    render(tasks, args.output / "03_matched_distributions", "distribution", matched=True)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
