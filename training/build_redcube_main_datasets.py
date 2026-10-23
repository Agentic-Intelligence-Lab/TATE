#!/usr/bin/env python3
"""Build ten main-experiment EEF datasets from audited raw/corrected ego JSON and real FK.

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


def exact_video_segment(source,destination,*,start_frame,frame_count):
    """Keep every selected source frame and assign the dataset's 30 Hz clock."""
    import os,subprocess
    import cv2
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(f'.{destination.stem}.tmp-{os.getpid()}.mp4')
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(source),
        '-vf',f'trim=start_frame={start_frame}:end_frame={start_frame+frame_count},setpts=N/(30*TB)',
        '-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p',
        '-r','30','-fps_mode','cfr','-movflags','+faststart',str(temporary)]
    try:
        subprocess.run(command,check=True)
        cap=cv2.VideoCapture(str(temporary));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        if count!=frame_count or abs(fps-30)>1e-6:raise RuntimeError((str(source),count,frame_count,fps))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/mnt/data/xule/Data_TATE_0930/stack_redcube_arx_ego_1'))
    p.add_argument('--study-root',type=Path,default=REPO/'outputs/unpaired_redcube_20261011')
    p.add_argument('--pairing',type=Path,default=REPO/'outputs/paired_redcube_20261010/pairing_verified.json')
    p.add_argument('--output-root',type=Path,default=REPO/'output/lerobot')
    p.add_argument('--seed',type=int,default=20261016)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();audit=args.output_root/'main_redcube_audit';audit.mkdir(parents=True,exist_ok=True)
    if (audit/'complete.json').exists():raise FileExistsError('Build already complete; do not overwrite')
    import os,shutil
    if shutil.which('ffmpeg') is None:
        import imageio_ffmpeg
        binary=Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        tools_dir=audit/'bin';tools_dir.mkdir(exist_ok=True)
        link=tools_dir/'ffmpeg'
        if not link.exists():link.symlink_to(binary)
        os.environ['PATH']=str(tools_dir)+os.pathsep+os.environ.get('PATH','')
    info=read_json(args.source/'meta/info.json')
    if int(info['fps'])!=30:raise ValueError('Unexpected FPS')
    pairs=read_json(args.pairing)['pairs'];all_real={r['real_episode'] for r in pairs};all_ego={r['ego_episode'] for r in pairs}
    manifest=read_json(args.study_root/'corrected/manifest.json')['methods']['unpaired_bridge']
    by_id={r['id']:r for r in manifest};assert set(by_id)==all_ego and len(all_ego)==100
    meta=read_json(args.study_root/'20261011/export_metadata.json');cal=list(meta['real_episode_ids'])
    assert len(cal)==len(set(cal))==20 and set(cal)<=all_real
    rest=sorted(all_real-set(cal));np.random.default_rng(args.seed).shuffle(rest)
    real_order=cal+rest
    selections={n:sorted(real_order[:n]) for n in [20,40,70,100]}
    # Stratify ego using its own observed scene features, never paired targets.
    scene=np.load(args.study_root/'20261011/training.npz')['source_scene']
    ego_source_ids=meta['ego_episode_ids']
    assert scene.shape==(100,2) and set(ego_source_ids)==all_ego
    edges=[np.quantile(scene[:,j],[1/3,2/3]) for j in range(2)]
    bins={};rng=np.random.default_rng(args.seed)
    for ep,point in zip(ego_source_ids,scene):
        cell=tuple(int(np.searchsorted(edges[j],point[j],side='right')) for j in range(2))
        bins.setdefault(cell,[]).append(ep)
    for ids in bins.values():rng.shuffle(ids)
    sizes={key:len(ids) for key,ids in bins.items()};taken={key:0 for key in bins};ego_order=[]
    tie_order=list(sorted(bins));rng.shuffle(tie_order)
    for step in range(100):
        cell=max((key for key in tie_order if bins[key]),key=lambda key:(step+1)*sizes[key]/100-taken[key])
        ego_order.append(bins[cell].pop());taken[cell]+=1
    ego_selections={n:sorted(ego_order[:n]) for n in [20,50,100]}
    specs={variant:TaskSpec('redcube_random','unpaired_redcube_20261011',variant,
        'pick the red cube and place it on the blue plate',('right',),args.source,args.source)
        for variant in ['none','unpaired_bridge','real_only']}
    configs=[('real_only',0,n) for n in [20,40,70,100]]+[(v,e,20) for e in [20,50,100] for v in ['none','unpaired_bridge']]
    names={(v,e,r):f'arx_eef_redcube_random_main_{"r"+str(r) if not e else "e"+str(e)+"r"+str(r)+"_"+("raw" if v=="none" else "tate")}_nodropout_train' for v,e,r in configs}
    if not args.resume:
        for name in names.values():
            if (args.output_root/'local'/name).exists():raise FileExistsError(name)
    plan=dict(real_ids=selections,ego_ids=ego_selections,calibration_real_ids=sorted(cal),
        selection_seed=args.seed,ego_selection='nested proportional 3x3 marginal-quantile scene strata, seeded within-cell shuffle',
        ego_scene_bin_edges=[edge.tolist() for edge in edges],ego_order=ego_order,real_order=real_order,
        ego_scene_by_id={str(ep):point.tolist() for ep,point in zip(ego_source_ids,scene)},
        real_selection='20 fixed calibration references, then seeded shuffle of remaining real demonstrations',
        source=str(args.source),state_action='16D xyz+xyzw+gripper, next frame action',
        gripper='0=open,1=closed',single_train_split=True,camera_dropout=False,
        ego_image_mask=[True,False,False],real_image_mask=[True,True,True],
        correction_model='unpaired SceneBridge primary seed 20261011; 20 real and ALL 100 ego used in correction fitting',
        ego_pose_source='direct EEF JSON, no IK/FK round-trip',datasets=list(names.values()))
    if args.resume:
        assert read_json(audit/'plan.json')==json.loads(json.dumps(plan)), 'Resume plan changed'
    write_json(audit/'plan.json',plan)
    import training.build_arx_cotrain_datasets as cotrain
    cotrain.materialize_video_segment=exact_video_segment
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
        for variant in ['none','unpaired_bridge']:prepared[variant,episode]=(states[variant],contiguous_runs(selected),frame_ids)
    fk=ArxForwardKinematics(DEFAULT_SCENE,DEFAULT_CALIBRATION)
    for episode in selections[100]:
        table=pq.read_table(args.source/f'data/chunk-000/file-{episode:03d}.parquet')
        for key in VIDEO_KEYS:source_video(args.source,key,episode)
        frame_ids=np.asarray(table['frame_index'].to_pylist(),np.int64)
        state=apply_arm_mask(joints_to_eef(np.asarray(table['observation.state'].to_pylist(),np.float32),fk),mask)
        prepared['real',episode]=(state,[np.arange(len(table)-1)],frame_ids)
    write_json(audit/'eef_source_hashes.json',hashes)
    results=read_json(audit/'progress.json') if args.resume and (audit/'progress.json').exists() else []
    completed={row['dataset'] for row in results}
    for variant,e,n in configs:
        real_ids=selections[n];ego_ids=ego_selections[e] if e else [];spec=specs[variant]
        root=args.output_root/'local'/names[variant,e,n]
        if str(root) in completed:
            prov=read_json(root/'cotrain_provenance.json')
            assert prov['expected_source_demonstrations']=={'ego':e,'real':n}
            assert prov['ego_selected_episode_ids']==ego_ids and prov['real_selected_episode_ids']==real_ids
            continue
        if root.exists():
            if not args.resume:raise FileExistsError(root)
            from datetime import datetime,timezone
            archived=audit/'incomplete'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')/root.name
            archived.parent.mkdir(parents=True,exist_ok=True)
            root.rename(archived)
        writer=DatasetWriter(root,info,spec,'train','nodropout','head',ego_ids,real_ids,mask,
            'single train; nested sets; real includes the same 20 correction references',False)
        for domain,episodes in [('ego',ego_ids),('real',real_ids)]:
            for episode in episodes:
                state,runs,frame_ids=prepared[variant if domain=='ego' else 'real',episode]
                for run in runs:
                    image_mask=np.tile([True,False,False] if domain=='ego' else [True,True,True],(len(run),1))
                    writer.add(source=domain,source_root=args.source,source_episode=episode,selected=run,source_length=len(state),
                        state=state[run],action=state[run+1],source_frame_index=frame_ids[run],image_mask=image_mask)
        result=writer.finish();result.update(variant=variant,real_demonstrations=n,ego_demonstrations=e)
        provenance=read_json(root/'cotrain_provenance.json')
        provenance.update(ego_variant=variant,correction=str(args.study_root/'20261011/model.json') if variant=='unpaired_bridge' else None,
            eef_manifest=str(args.study_root/'corrected/manifest.json') if e else None,
            ego_label_source='direct EEF JSON; no IK round trip' if e else None,
            correction_calibration_real_ids=sorted(cal) if variant=='unpaired_bridge' else [],
            correction_fitting_ego_ids=sorted(all_ego) if variant=='unpaired_bridge' else [],
            expected_source_demonstrations={'ego':e,'real':n},build_plan=str(audit/'plan.json'))
        write_json(root/'cotrain_provenance.json',provenance)
        results.append(result);write_json(audit/'progress.json',results);print(json.dumps(result),flush=True)
    write_json(audit/'complete.json',dict(results=results,plan=plan))

if __name__=='__main__':main()
