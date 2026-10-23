"""Materialize experimental corrections beside (never over) source EEF files."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    payload=json.load(gzip.open(args.payload))
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment output: {args.out}")
    args.out.mkdir(parents=True)
    sys.path.insert(0,"/mnt/data/xule/TATE")
    from evaluation.loaders import load_eef_json
    manifest=dict(schema="tate.experimental_scene_correction_export",schema_version=1,
                  exclusions=payload["exclusions"],methods={},policy_training=False)
    for method,rows in payload["methods"].items():
        records=[]
        for row in rows:
            source=Path(row["source"])
            raw=source.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row["source_sha256"]:
                raise ValueError(f"source changed: {source}")
            data=json.loads(raw)
            if len(data["frames"])!=len(row["p"]): raise ValueError("frame count mismatch")
            inv=np.linalg.inv(np.asarray(data["arm_transforms"]["T_cam_in_output_frame"]["right"]))
            for i,frame in enumerate(data["frames"]):
                hand=frame.get("hand_r")
                if hand is None or not hand.get("valid",True): continue
                old=deepcopy(hand["tcp_pose_eef_frame"])
                pose=np.eye(4);pose[:3,:3]=row["r"][i];pose[:3,3]=row["p"][i]
                hand["tcp_pose_eef_frame_before_scene_transport"]=old
                hand["tcp_pose_eef_frame"]=pose.tolist()
                hand["eef_pose_world"]=pose.tolist()
                hand["eef_translation_world"]=pose[:3,3].tolist()
                hand["tcp_pose_cam"]=(inv@pose).tolist()
                hand["eef_pose_cam"]=(inv@pose).tolist()
                hand["eef_linear_velocity_world"]=row["v"][i]
                hand["eef_angular_velocity_world"]=row["w"][i]
            data["correction"]=dict(schema="tate.experimental_scene_transport",schema_version=1,
                method=method,real_calibration_episode_ids=[str(x) for x in payload["calibration"]],
                ego_calibration_episode_ids=[str(x) for x in payload["ego_calibration"]],
                source_sha256=row["source_sha256"],phase_rule="start/open/close/release/end",
                scene_rule=payload.get('scene_rule','initial red minus blue centroid'),policy_training=False)
            data['correction']['experiment_family']=payload.get('experiment_family','scene_transport')
            data['correction']['application_mode']=row.get('mode',method)
            if 'pair_id' in row:data['correction']['pair_id']=row['pair_id']
            if 'model_sha256' in payload:data['correction']['model_sha256']=payload['model_sha256']
            if payload.get('experiment_family')=='paired_calibration':
                data['correction']['schema']='tate.experimental_paired_calibration'
            if payload.get('experiment_family')=='unpaired_scene_bridge':
                data['correction']['schema']='tate.experimental_unpaired_scene_bridge'
                data['correction']['cross_domain_pairing_used']=False
            if row.get('mode')=='global_affine_fallback':
                data['correction']['phase_rule']='event-independent fallback'
            data.setdefault("kinematic_limits",{})["enforced_after_correction"]=False
            target=args.out/method/f"episode_{row['id']:06d}"/"eef_corrected.json"
            target.parent.mkdir(parents=True)
            tmp=target.with_suffix(".tmp")
            tmp.write_text(json.dumps(data,separators=(",",":"),allow_nan=False))
            load_eef_json(tmp)
            tmp.replace(target)
            records.append(dict(id=row["id"],source=str(source),output=str(target),
                                frames=len(data["frames"]),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        manifest["methods"][method]=records
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:len(v) for k,v in manifest["methods"].items()}))


if __name__=="__main__":main()
