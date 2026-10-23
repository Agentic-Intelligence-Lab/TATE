"""Read-only remote dataset extraction. Gzipped JSON goes to stdout.

Extracts compact TCP trajectories and independent initial-frame color landmarks.
No source EEF or videos are modified. Red/blue segmentation is specific to this
red-cube experiment, not a general object pose estimator.
"""
import gzip
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def landmark(image, color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if color == "red":
        mask = (((h > 164) | (h < 5)) & (s > 85) & (v > 65)).astype("uint8")
    else:
        mask = ((h > 85) & (h < 115) & (s > 90) & (v > 65)).astype("uint8")
    height, width = mask.shape
    mask[:int(height*.25)] = 0
    mask[int(height*.8):] = 0
    mask[:, :int(width*.15)] = 0
    mask[:, int(width*.85):] = 0
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 100]
    if not candidates:
        return None
    i = max(candidates, key=lambda j: stats[j, cv2.CC_STAT_AREA])
    return dict(uv=centers[i].tolist(), area=int(stats[i, cv2.CC_STAT_AREA]),
                bbox=stats[i, :4].tolist())


def run():
    root = Path("/mnt/data/xule")
    result = dict(records={}, errors=[], extraction="initial frames 0,5,10; median landmark; raw TCP poses")
    for kind in ["real", "ego"]:
        folder = (root / "TATE/outputs/arx_real_flange/redcube_random_arx" if kind == "real" else
                  root / "TATE/outputs/experiments/redcube_random/artifacts/eef/finger_center_hys085")
        paths = sorted(folder.glob("episode_*.json" if kind == "real" else "episode_*/eef_raw.json"))
        records = []
        for path in paths:
            raw = path.read_bytes()
            d = json.loads(raw)
            episode = int((path.stem if kind == "real" else path.parent.name).split("_")[-1])
            p, r, t, valid, grip, events = [], [], [], [], [], []
            previous = None
            for i, f in enumerate(d["frames"]):
                a = f.get("arms", {}).get("right") if kind == "real" else f.get("hand_r")
                good = a is not None and a.get("valid", True)
                pose = np.array(a["T_tcp_in_output_frame" if kind == "real" else "tcp_pose_eef_frame"]) if good else np.eye(4)
                state = int(a["gripper"]["binary"] if kind == "real" else a["grasp_state"]) if good else -1
                timestamp = float(f.get("timestamp_s", i/d["fps"]))
                p.append(pose[:3, 3].tolist()); r.append(pose[:3, :3].tolist())
                t.append(timestamp); valid.append(good); grip.append(state)
                if good:
                    if previous is not None and previous[0] != state:
                        events.append(dict(frame=i, transition="close" if state else "open",
                            ambiguous=i!=previous[1]+1 and timestamp-previous[2]>.25))
                    previous = state, i, timestamp
            video = root / f"Data_TATE/redcube_random_{'arx' if kind=='real' else 'ego'}/videos/observation.images.head/chunk-000/file-{episode:03d}.mp4"
            cap = cv2.VideoCapture(str(video))
            landmarks = {"red": [], "blue": []}
            for index in [0, 5, 10]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"cannot read {video}:{index}")
                for color in landmarks:
                    l = landmark(image, color)
                    if l is not None:
                        landmarks[color].append(l)
            cap.release()
            lm = {}
            for color, values in landmarks.items():
                if values:
                    uv = np.array([v["uv"] for v in values])
                    lm[color] = dict(uv=np.median(uv, axis=0).tolist(),
                        jitter_px=float(np.linalg.norm(uv-np.median(uv,axis=0),axis=1).max()),
                        observations=len(values), area=float(np.median([v["area"] for v in values])))
                else:
                    lm[color] = None
            record = dict(id=episode, path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                video=str(video), p=p, r=r, t=t, valid=valid, grip=grip, events=events,
                landmarks=lm, image_size=[image.shape[1], image.shape[0]])
            if kind == "ego":
                record["camera"] = dict(k=d["k"], distortion=d["d"],
                    transform=d["arm_transforms"]["T_cam_in_output_frame"]["right"])
            records.append(record)
        result["records"][kind] = records
    sys.stdout.buffer.write(gzip.compress(json.dumps(result, allow_nan=False).encode()))


if __name__ == "__main__":
    run()
