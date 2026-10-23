"""Read-only anchor audit; run from the repository root with numpy/scipy installed.

This is a distribution diagnostic, not a per-episode correction evaluation.
All available healthy ego episodes are used transductively. Real references are
split into 20 calibration episodes and a disjoint evaluation set, repeatedly.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.transform import Rotation


def read_episode(path, kind):
    before = path.stat()
    data = json.loads(path.read_text())
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("file changed during read")
    frames = data["frames"]
    if len(frames) != data["total_frames"]:
        raise ValueError("incomplete frame count")
    events, poses, previous = [], [], None
    valid_count = 0
    frame_names = set()
    for i, frame in enumerate(frames):
        arm = frame.get("arms", {}).get("right") if kind == "real" else frame.get("hand_r")
        if not arm or not arm.get("valid", True):
            continue
        pose = np.asarray(arm["T_tcp_in_output_frame" if kind == "real" else "tcp_pose_eef_frame"])
        state = int(arm["gripper"]["binary"] if kind == "real" else arm["grasp_state"])
        frame_names.add(arm.get("output_frame" if kind == "real" else "eef_frame"))
        if state not in (0, 1) or pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("invalid state/pose")
        time = float(frame.get("timestamp_s", i / data["fps"]))
        valid_count += 1
        poses.append(pose)
        if previous is not None and previous[0] != state:
            events.append(dict(frame=i, time_s=time, transition="close" if state else "open",
                               ambiguous=i != previous[1] + 1 and time - previous[2] > .25,
                               position_m=pose[:3, 3].tolist(), rotation=pose[:3, :3].tolist()))
        previous = state, i, time
    return dict(id=path.stem if kind == "real" else path.parent.name,
                path=str(path), frames=len(frames), valid_fraction=valid_count / len(frames),
                camera_origin_m=(np.asarray(data["arm_transforms"]["T_cam_in_output_frame"]["right"])[:3, 3].tolist()
                                 if kind == "ego" else None),
                frame_names=sorted(frame_names), events=events,
                start_m=poses[0][:3, 3].tolist() if poses else None,
                end_m=poses[-1][:3, 3].tolist() if poses else None)


def summary(records):
    result = dict(count=len(records), sequences=dict(Counter(
        ",".join(e["transition"] for e in r["events"]) for r in records)),
        mean_valid_fraction=float(np.mean([r["valid_fraction"] for r in records])) if records else None)
    healthy = [r for r in records if [e["transition"] for e in r["events"]] == ["open", "close", "open"]
               and not any(e["ambiguous"] for e in r["events"])]
    result["healthy_count"] = len(healthy)
    result["anchors"] = {}
    for i, label in enumerate(["pregrasp_open", "grasp_close", "release_open"]):
        if len(healthy) < 2:
            continue
        x = np.array([r["events"][i]["position_m"] for r in healthy])
        rotations = Rotation.from_matrix(np.array([r["events"][i]["rotation"] for r in healthy]))
        mean = rotations.mean()
        result["anchors"][label] = dict(mean_m=x.mean(0).tolist(), std_m=x.std(0, ddof=1).tolist(),
            range_m=np.ptp(x, axis=0).tolist(), mean_rotation=mean.as_matrix().tolist(),
            mean_orientation_spread_deg=float(np.degrees((mean.inv()*rotations).magnitude()).mean()))
    return result, healthy


def swd(x, y, directions):
    return float(np.mean([wasserstein_distance(x @ v, y @ v) for v in directions]))


def baseline_probe(real, ego, repeats):
    if len(real) <= 20 or len(ego) < 5:
        return {"skipped": "requires >20 healthy real and >=5 healthy ego"}
    r = np.array([[e["position_m"] for e in v["events"][1:]] for v in real])
    x = np.array([[e["position_m"] for e in v["events"][1:]] for v in ego])
    origins = np.array([v["camera_origin_m"] for v in ego])[:, None, :]
    rng = np.random.default_rng(947)
    directions = rng.normal(size=(256, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    results = []
    for seed in range(repeats):
        order = np.random.default_rng(seed).permutation(len(r))
        train, test = r[order[:20]], r[order[20:]]
        delta = train.mean(0) - x.mean(0)
        samples = {
            "raw": x,
            "anchor_mean": np.broadcast_to(train.mean(0), x.shape),
            "anchor_translation": x + delta,
            "release_shared_translation": x + delta[1],
        }
        # Strongly restricted depth/height baseline: preserve each source XY.
        z = x.copy()
        z[:, :, 2] += delta[:, 2]
        samples["anchor_z_translation"] = z
        rays = x - origins
        if np.all(np.abs(rays[:, :, 2]) > 1e-6):
            scales = (train.mean(0)[None, :, 2] - origins[:, :, 2]) / rays[:, :, 2]
            if np.all(scales > 0):
                samples["ray_to_real_anchor_height"] = origins + rays * scales[:, :, None]
        entry = dict(seed=seed, calibration_ids=[real[i]["id"] for i in order[:20]], methods={})
        for name, y in samples.items():
            entry["methods"][name] = dict(
                grasp_swd_m=swd(y[:, 0], test[:, 0], directions),
                release_swd_m=swd(y[:, 1], test[:, 1], directions),
                grasp_xy_spread_m=float(np.linalg.norm(y[:, 0, :2].std(0))),
                mean_anchor_displacement_m=float(np.linalg.norm(y-x, axis=2).mean()))
        entry["real20_vs_real_holdout"] = dict(grasp_swd_m=swd(train[:, 0], test[:, 0], directions),
                                               release_swd_m=swd(train[:, 1], test[:, 1], directions))
        results.append(entry)
    aggregate = {}
    for name in results[0]["methods"]:
        aggregate[name] = {metric: dict(mean=float(np.mean(values)), std=float(np.std(values)))
            for metric in results[0]["methods"][name]
            for values in [[row["methods"][name][metric] for row in results]]}
    return dict(protocol="20 real calibration; remaining healthy real held out; all healthy ego transductive; no scene ground truth",
                repeats=repeats, aggregate=aggregate, splits=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--ego-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(), errors=[], records={}, summary={})
    healthy = {}
    for kind, paths in [("real", sorted(args.real_root.glob("episode_*.json"))),
                        ("ego", sorted(args.ego_root.glob("episode_*/eef_raw.json")))]:
        records = []
        for path in paths:
            try:
                records.append(read_episode(path, kind))
            except (ValueError, KeyError, OSError) as exc:
                output["errors"].append(dict(path=str(path), error=str(exc)))
        output["records"][kind] = records
        output["summary"][kind], healthy[kind] = summary(records)
    names = {tuple(r["frame_names"]) for group in healthy.values() for r in group}
    if len(names) != 1 or next(iter(names), ()) != ("right_flange_zero",):
        raise ValueError(f"mismatched or unexpected coordinate frame names: {names}")
    output["probe"] = baseline_probe(healthy["real"], healthy["ego"], args.repeats)
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
