#!/usr/bin/env python3
"""Compare the 1016 bowl/hanging collections with their previous experiments."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


TASKS = {
    "bowl": {"old": "bowl_1", "new": "bowl_1016", "real": "bowl_arx"},
    "hanging": {"old": "hanging_1", "new": "hanging_1016", "real": "hanging_arx"},
}
VARIANTS = {
    "before": "finger_center_hys085__none",
    "after": "finger_center_hys085__position_rotation",
}
METRICS = ["rho_pos", "rho_rot", "rho_se3", "D_pos_mm", "D_rot_deg"]
GROUPS = [
    ("real", "Real robot", "#303641", "-"),
    ("old", "Previous corrected", "#dd7844", "-"),
    ("new", "Recollected corrected", "#087f9b", "--"),
]
AXIS_LABELS = ["X / forward (cm)", "Y / left (cm)", "Z / up (cm)"]


def load_plot_helpers(root: Path):
    path = root / "outputs/five_task_eef_visualization_20261012/visualize_five_task_eef.py"
    spec = importlib.util.spec_from_file_location("tate_plot_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.read_ego, module.read_real, module.points


def headline(summary_path: Path):
    data = json.loads(summary_path.read_text())
    metrics = data["metrics"]
    return ({key: metrics["bimanual_active_side_mean"][key] for key in METRICS},
            {side: block["n_ego_episodes"] for side, block in metrics["per_side"].items()})


def collect_metrics(root: Path, output: Path):
    old = json.loads((root / "outputs/five_task_correction_eval_20261012/metrics.json").read_text())
    old_lookup = {(row["task"], row["variant"]): row for row in old["task_metrics"]}
    rows = []
    counts = {}
    for task in TASKS:
        counts[task] = {}
        for generation in ["old", "new"]:
            for stage, variant in VARIANTS.items():
                if generation == "old":
                    values = {key: old_lookup[(task, stage)][key] for key in METRICS}
                    n = old["counts"][task]["ego_per_side"][stage]
                else:
                    path = output / task / "evaluation" / variant / "summary.json"
                    values, n = headline(path)
                counts[task][f"{generation}_{stage}"] = n
                rows.append({"task": task, "generation": generation, "stage": stage, **values})
    lookup = {(r["task"], r["generation"], r["stage"]): r for r in rows}
    changes = []
    for task in TASKS:
        for stage in ["before", "after"]:
            a, b = lookup[(task, "old", stage)], lookup[(task, "new", stage)]
            changes.append({"task": task, "comparison": f"new_vs_old_{stage}",
                            **{key: (b[key] / a[key] - 1) * 100 for key in METRICS}})
        for generation in ["old", "new"]:
            a, b = lookup[(task, generation, "before")], lookup[(task, generation, "after")]
            changes.append({"task": task, "comparison": f"{generation}_after_vs_before",
                            **{key: (b[key] / a[key] - 1) * 100 for key in METRICS}})
    for name, values in [("metrics_comparison.csv", rows), ("percent_changes.csv", changes)]:
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    payload = {"metrics": rows, "percent_changes": changes, "evaluated_ego_counts": counts}
    (output / "comparison_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return rows, payload


def plot_metrics(rows, output: Path):
    lookup = {(r["task"], r["generation"], r["stage"]): r for r in rows}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), constrained_layout=True)
    x = np.arange(len(METRICS)); width = .36
    labels = ["rho_pos", "rho_rot", "rho_se3", "D_pos (mm)", "D_rot (deg)"]
    for ax, task in zip(axes, TASKS):
        old = lookup[(task, "old", "after")]
        new = lookup[(task, "new", "after")]
        reductions = np.array([(1 - new[k] / old[k]) * 100 for k in METRICS])
        bars = ax.bar(x, reductions, width=.62, color="#087f9b")
        for i, (bar, key) in enumerate(zip(bars, METRICS)):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.2,
                    f"{old[key]:.2f} → {new[key]:.2f}\n{reductions[i]:.1f}%",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, max(60, reductions.max()+10))
        ax.set_ylabel("Reduction from previous (%)")
        ax.set_title(task.capitalize(), loc="left", weight="bold")
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Recollected vs previous corrected datasets (lower is better)", fontsize=16, weight="bold")
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"metrics_new_vs_previous.{ext}", dpi=180)
    plt.close(fig)


def plot_distributions(root: Path, output: Path):
    read_ego, read_real, points = load_plot_helpers(root)
    fig = plt.figure(figsize=(18, 8.3), facecolor="white")
    grid = fig.add_gridspec(2, 4, left=.045, right=.985, bottom=.10, top=.84,
                           wspace=.30, hspace=.40)
    audit = {}
    for row, (task, cfg) in enumerate(TASKS.items()):
        datasets = {}
        infos = {}
        for generation in ["old", "new"]:
            path = root / "outputs/experiments" / cfg[generation] / "datasets" / VARIANTS["after"]
            datasets[generation], infos[generation] = read_ego(path)
        side = infos["old"]["tate"]["active_sides"][0]
        datasets["real"] = read_real(root / "outputs/arx_real_flange" / cfg["real"], [side])
        clouds = {key: points(value, side) for key, value in datasets.items()}
        all_points = np.concatenate(list(clouds.values()))
        lo, hi = all_points.min(0), all_points.max(0)
        pad = np.maximum((hi-lo)*.05, .8); lo -= pad; hi += pad
        ax3 = fig.add_subplot(grid[row, 0], projection="3d")
        axes = [fig.add_subplot(grid[row, i]) for i in range(1, 4)]
        for key, label, color, style in GROUPS:
            p = clouds[key]
            ax3.scatter(*p.T, s=.55, alpha=.12, linewidths=0, color=color,
                        depthshade=False, rasterized=True)
            for dim, ax in enumerate(axes):
                bins = np.linspace(lo[dim], hi[dim], 61)
                hist, edges = np.histogram(p[:, dim], bins=bins, density=True)
                ax.stairs(hist, edges, color=color, lw=1.55, ls=style)
        ax3.set_title(f"{task.capitalize()} / right\nEpisodes R/O/N: " +
                      "/".join(str(len(datasets[k])) for k in ["real", "old", "new"]), loc="left")
        ax3.set(xlabel=AXIS_LABELS[0], ylabel=AXIS_LABELS[1], zlabel=AXIS_LABELS[2],
                xlim=(lo[0],hi[0]), ylim=(lo[1],hi[1]), zlim=(lo[2],hi[2]))
        ax3.set_box_aspect(hi-lo); ax3.view_init(elev=24, azim=-58)
        for dim, ax in enumerate(axes):
            ax.set(xlabel=AXIS_LABELS[dim], ylabel="Probability density", xlim=(lo[dim],hi[dim]))
            ax.set_title(f"{'XYZ'[dim]} distribution")
            ax.grid(alpha=.18); ax.spines[["top", "right"]].set_visible(False)
        real_lo, real_hi = clouds["real"].min(0), clouds["real"].max(0)
        audit[task] = {key: {"episodes": len(datasets[key]), "valid_frames": len(clouds[key]),
                             "mean_cm": clouds[key].mean(0).tolist(),
                             "min_cm": clouds[key].min(0).tolist(),
                             "max_cm": clouds[key].max(0).tolist(),
                             "frames_outside_real_axis_envelope": int(np.any(
                                 (clouds[key] < real_lo) | (clouds[key] > real_hi), axis=1).sum()),
                             "fraction_outside_real_axis_envelope": float(np.any(
                                 (clouds[key] < real_lo) | (clouds[key] > real_hi), axis=1).mean())}
                       for key in datasets}
        for key in ["old", "new"]:
            outside_ids = []
            extrema = {}
            for eid, episode in datasets[key].items():
                p = episode["poses"][side]
                valid = np.isfinite(p).all(1)
                if np.any(np.any((p[valid] < real_lo) | (p[valid] > real_hi), axis=1)):
                    outside_ids.append(eid)
            for dim, axis in enumerate("xyz"):
                candidates = []
                for eid, episode in datasets[key].items():
                    p = episode["poses"][side]
                    valid_indices = np.flatnonzero(np.isfinite(p).all(1))
                    if len(valid_indices):
                        for kind, local in [("min", valid_indices[np.argmin(p[valid_indices, dim])]),
                                            ("max", valid_indices[np.argmax(p[valid_indices, dim])])]:
                            candidates.append((kind, float(p[local, dim]), int(eid),
                                               int(episode["frames"][local])))
                extrema[axis] = {kind: {"value_cm": value, "source_episode_id": eid,
                                        "source_frame_index": frame}
                                 for kind, value, eid, frame in
                                 [min((x for x in candidates if x[0] == "min"), key=lambda x:x[1]),
                                  max((x for x in candidates if x[0] == "max"), key=lambda x:x[1])]}
            audit[task][key]["episodes_outside_real_axis_envelope"] = len(outside_ids)
            audit[task][key]["outside_source_episode_ids"] = sorted(outside_ids)
            audit[task][key]["extrema"] = extrema
    handles = [Line2D([0],[0],color=c,ls=s,lw=2,label=l) for _,l,c,s in GROUPS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5,.91))
    fig.suptitle("Corrected TCP position distributions", fontsize=17, weight="bold", y=.98)
    fig.text(.5, .035, "All valid frames; X forward / Y left / Z up; centimeters; no trimming or subsampling",
             ha="center", fontsize=9, color="#555")
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"corrected_position_distributions.{ext}", dpi=180)
    plt.close(fig)
    (output / "distribution_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/data/xule/TATE"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows, _ = collect_metrics(args.root, args.output)
    plot_metrics(rows, args.output)
    plot_distributions(args.root, args.output)


if __name__ == "__main__":
    main()
