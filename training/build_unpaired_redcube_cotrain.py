#!/usr/bin/env python3
"""Build six EEF cotrain datasets from audited raw/corrected ego JSON and real FK.

Reuses the established ARX cotrain writer, masks, video slicing and t -> t+1
semantics. No IK round trip: ego labels are exactly the selected EEF export.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from training.build_arx_cotrain_datasets import (
    TaskSpec,DatasetWriter,ArxForwardKinematics,DEFAULT_SCENE,DEFAULT_CALIBRATION,
    VIDEO_KEYS,EEF_DIM,CANONICAL_ARM,eef_mask,apply_arm_mask,joints_to_eef,
    continuous_quaternions,contiguous_runs,source_video,read_json,write_json,
)


def read_ego(path):
    payload=read_json(path);frames=payload['frames'];n=len(frames)
    state=np.tile(np.r_[CANONICAL_ARM,CANONICAL_ARM],(n,1));valid=np.zeros(n,bool)
    for i,f in enumerate(frames):
        h=f.get('hand_r')
        if h is None or not h.get('valid',True):continue
        pose=np.asarray(h['tcp_pose_eef_frame'],float)
        if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError(path)
        state[i,8:11]=pose[:3,3];state[i,11:15]=Rotation.from_matrix(pose[:3,:3]).as_quat()
        state[i,15]=int(h['grasp_state']);valid[i]=True
    state[:,11:15]=continuous_quaternions(state[:,11:15])
    return state,valid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/mnt/data/xule/Data_TATE_0930/stack_redcube_arx_ego_1'))
    p.add_argument('--study-root',type=Path,default=REPO/'outputs/unpaired_redcube_20261011')
    p.add_argument('--pairing',type=Path,default=REPO/'outputs/paired_redcube_20261010/pairing_verified.json')
    p.add_argument('--output-root',type=Path,default=REPO/'outputs/lerobot')
    args=p.parse_args();audit=args.study_root/'cotrain_build';audit.mkdir(exist_ok=True)
    if (audit/'complete.json').exists():raise FileExistsError('Build already complete; do not overwrite')
    info=read_json(args.source/'meta/info.json')
    if int(info['fps'])!=30:raise ValueError('Unexpected FPS')
    pairs=read_json(args.pairing)['pairs'];all_real={r['real_episode'] for r in pairs};all_ego={r['ego_episode'] for r in pairs}
    manifest=read_json(args.study_root/'corrected/manifest.json')['methods']['unpaired_bridge']
    by_id={r['id']:r for r in manifest};assert set(by_id)==all_ego and len(all_ego)==100
    meta=read_json(args.study_root/'20261011/export_metadata.json');cal=list(meta['real_episode_ids'])
    assert len(cal)==len(set(cal))==20 and set(cal)<=all_real
    rest=sorted(all_real-set(cal));np.random.default_rng(20261011).shuffle(rest)
    real_order=cal+rest
    selections={n:sorted(real_order[:n]) for n in [20,40,60]}
    specs={variant:TaskSpec('redcube_random','unpaired_redcube_20261011',variant,
        'pick the red cube and place it on the blue plate',('right',),args.source,args.source) for variant in ['none','unpaired_bridge']}
    names={(variant,n):f'arx_eef_redcube_random_cotrain_e100r{n}_{variant}_nodropout_train' for variant in specs for n in selections}
    for name in names.values():
        if (args.output_root/'local'/name).exists():raise FileExistsError(name)
    plan=dict(real_ids=selections,ego_ids=sorted(all_ego),calibration_real_ids=sorted(cal),
        source=str(args.source),state_action='16D xyz+xyzw+gripper, next frame action',
        gripper='0=open,1=closed',single_train_split=True,camera_dropout=False,
        ego_image_mask=[True,False,False],real_image_mask=[True,True,True],
        correction_model='unpaired SceneBridge primary seed 20261011; always 20 real calibration trajectories',
        ego_pose_source='direct EEF JSON, no IK/FK round-trip',datasets=list(names.values()))
    write_json(audit/'plan.json',plan)
    prepared={};hashes=[];mask=eef_mask(('right',))
    for episode in sorted(all_ego):
        row=by_id[episode];paths={'none':Path(row['source']),'unpaired_bridge':REPO/row['output']}
        table=pq.read_table(args.source/f'data/chunk-000/file-{episode:03d}.parquet')
        frame_ids=np.asarray(table['frame_index'].to_pylist(),np.int64)
        assert np.array_equal(frame_ids,np.arange(len(table)))
        for key in VIDEO_KEYS:source_video(args.source,key,episode)
        states={};valids={}
        for variant,path in paths.items():
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if variant=='unpaired_bridge':assert digest==row['sha256']
            else:
                corrected=read_json(paths['unpaired_bridge'])
                assert digest==corrected['correction']['source_sha256']
            state,valid=read_ego(path);assert len(state)==len(table)
            states[variant]=state;valids[variant]=valid
            hashes.append(dict(episode=episode,variant=variant,path=str(path),sha256=digest))
        assert np.array_equal(valids['none'],valids['unpaired_bridge'])
        assert np.array_equal(states['none'][:,15],states['unpaired_bridge'][:,15])
        selected=np.flatnonzero(valids['none'][:-1]&valids['none'][1:]);assert len(selected)>0
        for variant in specs:prepared[variant,episode]=(states[variant],contiguous_runs(selected),frame_ids)
    fk=ArxForwardKinematics(DEFAULT_SCENE,DEFAULT_CALIBRATION)
    for episode in selections[60]:
        table=pq.read_table(args.source/f'data/chunk-000/file-{episode:03d}.parquet')
        for key in VIDEO_KEYS:source_video(args.source,key,episode)
        frame_ids=np.asarray(table['frame_index'].to_pylist(),np.int64)
        state=apply_arm_mask(joints_to_eef(np.asarray(table['observation.state'].to_pylist(),np.float32),fk),mask)
        prepared['real',episode]=(state,[np.arange(len(table)-1)],frame_ids)
    write_json(audit/'eef_source_hashes.json',hashes)
    results=[]
    for variant,spec in specs.items():
        for n,real_ids in selections.items():
            root=args.output_root/'local'/names[variant,n]
            writer=DatasetWriter(root,info,spec,'train','nodropout','head',sorted(all_ego),real_ids,mask,
                'single train; nested real sets start with the 20 correction calibration references',False)
            for domain,episodes in [('ego',sorted(all_ego)),('real',real_ids)]:
                for episode in episodes:
                    state,runs,frame_ids=prepared[variant if domain=='ego' else 'real',episode]
                    for run in runs:
                        image_mask=np.tile([True,False,False] if domain=='ego' else [True,True,True],(len(run),1))
                        writer.add(source=domain,source_root=args.source,source_episode=episode,selected=run,source_length=len(state),
                            state=state[run],action=state[run+1],source_frame_index=frame_ids[run],image_mask=image_mask)
            result=writer.finish();result.update(variant=variant,real_demonstrations=n,ego_demonstrations=100)
            provenance=read_json(root/'cotrain_provenance.json')
            provenance.update(ego_variant=variant,correction=str(args.study_root/'20261011/model.json') if variant!='none' else None,
                eef_manifest=str(args.study_root/'corrected/manifest.json'),ego_label_source='direct EEF JSON; no IK round trip',
                correction_calibration_real_ids=sorted(cal) if variant!='none' else [],
                expected_source_demonstrations={'ego':100,'real':n},build_plan=str(audit/'plan.json'))
            write_json(root/'cotrain_provenance.json',provenance)
            results.append(result);write_json(audit/'progress.json',results);print(json.dumps(result),flush=True)
    write_json(audit/'complete.json',dict(results=results,plan=plan))

if __name__=='__main__':main()
