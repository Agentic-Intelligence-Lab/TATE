#!/usr/bin/env python3
"""Read-only bowl anchor audit and diagnostic interpolation counterfactual."""
import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mnt/data/xule/TATE'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    from evaluation.loaders import load_eef_json, load_real_fk_json
    from evaluation.events import extract_events
    from evaluation.run_eval import _distance_components, _merge, DEFAULT_CONFIG
    from correction.methods.anchors import propagate_anchor_displacements
    from correction.anchor_ids import resolve_anchor_frame
    import yaml

    out = args.output; out.mkdir(parents=True, exist_ok=True)
    exp = args.root/'outputs/experiments/bowl_1'
    artifact = json.loads((args.root/'outputs/corrections/bowl/xyz_mean_target_min_bending_all_anchors.json').read_text())
    ids = ['start','0','1','2','end']
    anchors = artifact['sides']['right']['anchors']
    targets = np.array([anchors[k]['position_mean_m'] for k in ids])
    cal_ids = anchors['0']['episode_ids']
    real = {str(i): load_real_fk_json(args.root/f'outputs/arx_real_flange/bowl_arx/episode_{i:06d}.json') for i in range(100)}
    real_positions = {}; real_frames = {}
    for eid, t in real.items():
        events = extract_events(t,'right')
        if [e.transition for e in events] != ['open','close','open']: continue
        frames = [resolve_anchor_frame(t,'right',events,k) for k in ids]
        real_frames[eid] = frames
        real_positions[eid] = t.sides['right'].position[frames]
    np.testing.assert_allclose(np.mean([real_positions[i] for i in cal_ids],axis=0), targets, atol=1e-12)
    heldout = ['10','11','12','13','14','15','16','17','19']
    heldout_points = np.array([real_positions[i] for i in heldout])
    before = {}; after = {}; records = []; per_episode = []; sample = None
    parquet = {}
    for f in sorted((exp/'datasets/finger_center_hys085__position_rotation/data').rglob('*.parquet')):
        cols=['tate.source_episode_index','tate.source_frame_index','tate.eef.right.pose','tate.eef.right.valid']
        for row in pq.read_table(f,columns=cols).to_pylist():
            parquet[(int(row['tate.source_episode_index']), int(row['tate.source_frame_index']))] = row
    for p in sorted((exp/'artifacts/variants/finger_center_hys085__position_rotation').glob('episode_*/eef.json')):
        eid = int(p.parent.name.split('_')[-1])
        if not any(key[0]==eid for key in parquet): continue
        bpath = exp/f'artifacts/variants/finger_center_hys085__none/episode_{eid:06d}/eef.json'
        b = load_eef_json(bpath); a = load_eef_json(p)
        data = json.loads(p.read_text()); md = data['metadata']['real_anchor_correction']['applied']['right']
        frames = md['anchor_frames']; assert md['anchor_ids']==ids
        be = extract_events(b,'right'); ae = extract_events(a,'right')
        assert [(e.frame_index,e.transition) for e in be]==[(e.frame_index,e.transition) for e in ae]
        assert frames==[resolve_anchor_frame(a,'right',ae,k) for k in ids]
        disp = np.array(md['target_positions_m'])-b.sides['right'].position[frames]
        mb,_,_ = propagate_anchor_displacements(a.n,frames,disp,method='min_bending')
        linear,_,_ = propagate_anchor_displacements(a.n,frames,disp,method='linear')
        valid = a.sides['right'].valid & b.sides['right'].valid
        np.testing.assert_allclose((a.sides['right'].position-b.sides['right'].position)[valid], mb[valid], atol=1e-10)
        for j,k in enumerate(ids):
            frame=frames[j]; bp=b.sides['right'].position[frame]; ap=a.sides['right'].position[frame]
            row=parquet[(eid,frame)]; assert row['tate.eef.right.valid']
            pp=np.array(row['tate.eef.right.pose'][:3])
            target_q=Rotation.from_quat(anchors[k]['orientation_mean_quat_xyzw'])
            rec=dict(ego_source_id=eid,anchor=k,frame=frame,
                before_target_mm=float(np.linalg.norm(bp-targets[j])*1000),
                after_target_mm=float(np.linalg.norm(ap-targets[j])*1000),
                parquet_target_mm=float(np.linalg.norm(pp-targets[j])*1000),
                after_real0_mm=float(np.linalg.norm(ap-real_positions['0'][j])*1000),
                after_heldout_anchor_mean_mm=float(np.linalg.norm(ap-heldout_points[:,j].mean(0))*1000),
                after_heldout_anchor_mean_distance_mm=float(np.linalg.norm(ap-heldout_points[:,j],axis=1).mean()*1000),
                before_target_rotation_deg=float(np.degrees((target_q.inv()*Rotation.from_quat(b.sides['right'].quaternion_xyzw[frame])).magnitude())),
                after_target_rotation_deg=float(np.degrees((target_q.inv()*Rotation.from_quat(a.sides['right'].quaternion_xyzw[frame])).magnitude())))
            records.append(rec)
        norms=np.linalg.norm(mb,axis=1); controls=np.linalg.norm(disp,axis=1)
        per_episode.append(dict(ego_source_id=eid,event_frames=frames[1:-1],
            open_close_seconds=(frames[2]-frames[1])/30,
            max_displacement_mm=float(norms.max()*1000),
            max_anchor_displacement_mm=float(controls.max()*1000),
            frame_max_displacement=int(np.argmax(norms)),
            displacement_norm_ratio=float(norms.max()/controls.max()),
            max_minbending_linear_gap_mm=float(np.linalg.norm(mb-linear,axis=1).max()*1000)))
        before[eid]=b.sides['right'].position[frames]
        after[eid]=a.sides['right'].position[frames]
        if eid==1:sample=(b,a,frames,mb,linear,data)
    assert len(before)==74
    b,a,frames,mb,linear,data=sample
    sample_linear=copy.deepcopy(a)
    sample_linear.sides['right'].pose_xyzw[:,:3]=b.sides['right'].position+linear
    config=_merge(DEFAULT_CONFIG,yaml.safe_load((args.root/'cfg/evaluation/bowl_arx.yaml').read_text()))
    sample_metrics={}
    for name,t in [('before',b),('after_min_bending',a),('diagnostic_linear',sample_linear)]:
        distances=[]
        for rid in heldout:
            _,d=_distance_components(t,real[rid],'right',config); distances.append(d)
        sample_metrics[name]={key:float(np.mean([d[key] for d in distances])) for key in ['D_pos_m','D_rot_deg']}
    summary=dict(episodes=74,calibration_healthy_ids=cal_ids,calibration_excluded=artifact['exclusions'],
        anchor_targets_m=targets.tolist(),real0_anchor_positions_m=real_positions['0'].tolist(),
        real0_frames=real_frames['0'],sample_ego_frames=frames,
        heldout_ids=heldout,real_healthy_count=len(real_positions),
        max_anchor_residual_mm=max(r['after_target_mm'] for r in records),
        max_parquet_anchor_residual_mm=max(r['parquet_target_mm'] for r in records),
        sample_metrics=sample_metrics,episodes_with_displacement_ratio_gt_1_1=sum(r['displacement_norm_ratio']>1.1 for r in per_episode),
        sample_open_close_geometry_mm={
            'ego_before_distance':float(np.linalg.norm(b.sides['right'].position[frames[2]]-b.sides['right'].position[frames[1]])*1000),
            'target_distance':float(np.linalg.norm(targets[2]-targets[1])*1000),
            'correction_change':float(np.linalg.norm(mb[frames[2]]-mb[frames[1]])*1000)},
        sample_position_ranges_m={name:{'min':np.nanmin(t.sides['right'].position,axis=0).tolist(),
                                       'max':np.nanmax(t.sides['right'].position,axis=0).tolist()}
                                  for name,t in [('real0',real['0']),('before',b),('after',a),('linear',sample_linear)]},
        sample_max_interpolation_difference_mm={
            'before_first_event':float(np.linalg.norm(mb[:frames[1]]-linear[:frames[1]],axis=1).max()*1000),
            'event_segment':float(np.linalg.norm(mb[frames[1]:frames[3]+1]-linear[frames[1]:frames[3]+1],axis=1).max()*1000)},
        sample_episode=next(r for r in per_episode if r['ego_source_id']==1),
        anchor_stats={k:{name:float(np.mean([r[name] for r in records if r['anchor']==k])) for name in
                       ['before_target_mm','after_target_mm','after_real0_mm','after_heldout_anchor_mean_mm',
                        'after_heldout_anchor_mean_distance_mm','before_target_rotation_deg','after_target_rotation_deg']} for k in ids})
    for name,rows in [('anchor_audit.csv',records),('episode_audit.csv',per_episode)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'audit.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained')
    tracks=[('Real episode 0',real['0'].sides['right'].position,'#333b45','-'),
            ('Ego 1 before',b.sides['right'].position,'#dc7847','-'),
            ('Ego 1 corrected',a.sides['right'].position,'#087f9b','-'),
            ('Linear diagnostic',sample_linear.sides['right'].position,'#799844','--')]
    for ax,(d1,d2) in zip(axes[0],[(0,1),(0,2),(1,2)]):
        for label,p,color,style in tracks:
            ax.plot(p[:,d1]*100,p[:,d2]*100,label=label,color=color,ls=style,lw=1.4)
        ax.scatter(targets[1:4,d1]*100,targets[1:4,d2]*100,marker='X',s=100,color='black',label='Calibration mean anchors',zorder=8)
        for j in range(1,4):
            ax.annotate(f'A{j-1}',targets[j,[d1,d2]]*100,xytext=(5,7),textcoords='offset points',weight='bold')
            ax.scatter(real_positions['0'][j,d1]*100,real_positions['0'][j,d2]*100,facecolors='none',edgecolors='black',s=90,zorder=8)
        ax.set(xlabel=f'{"XYZ"[d1]} (cm)',ylabel=f'{"XYZ"[d2]} (cm)',title=f'{"XYZ"[d1]}{"XYZ"[d2]} / full episode')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,loc='best')
    for dim,ax in enumerate(axes[1]):
        ax.plot(mb[:,dim]*100,color='#087f9b',label='Current min-bending displacement')
        ax.plot(linear[:,dim]*100,color='#799844',ls='--',label='Piecewise-linear displacement')
        ax.scatter(frames,mb[frames,dim]*100,color='black',s=35,zorder=5)
        for f,label in zip(frames,ids):
            ax.axvline(f,color='gray',alpha=.25);ax.text(f,.99,label,transform=ax.get_xaxis_transform(),va='top',fontsize=8)
        ax.axvspan(0,frames[1],color='gray',alpha=.07)
        ax.axvspan(frames[3],a.n-1,color='gray',alpha=.07)
        ax.set(xlabel='Ego source frame',ylabel=f'{"XYZ"[dim]} correction (cm)',title='Correction between fixed anchors')
        ax.grid(alpha=.2)
    axes[1,0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Bowl: anchors coincide with their targets, but the interpolated path can deviate\n'
                 'X forward / Y left / Z up; circles = real-0 events; X = corrected ego events / calibration means\n'
                 'Linear is a diagnostic only; original data are unchanged. Shaded frame ranges are outside evaluation.',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(out/f'bowl_anchor_diagnosis.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for j,ax in enumerate(axes):
        bp=np.array(list(before.values()))[:,j+1]*100
        rp=heldout_points[:,j+1]*100
        ax.scatter(bp[:,1],bp[:,2],color='#dc7847',alpha=.5,label='Before: 74 ego')
        ax.scatter(rp[:,1],rp[:,2],color='#333b45',marker='o',facecolors='none',s=65,label='9 held-out real')
        ax.scatter(targets[j+1,1]*100,targets[j+1,2]*100,color='#087f9b',marker='X',s=130,label='After: 74 coincide at target')
        ax.set(xlabel='Y / left (cm)',ylabel='Z / up (cm)',title=['A0: first open','A1: close','A2: last open'][j])
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Bowl event anchors / YZ projection (no trajectory interpolation)',fontsize=14)
    for ext in ['png','pdf']:fig.savefig(out/f'bowl_event_anchor_distribution.{ext}',dpi=170)
    plt.close(fig)
    # Inspect observed video events separately from numerical label alignment.
    import cv2
    video=data['source_video_original']; cap=cv2.VideoCapture(video)
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    for row,frame in enumerate(frames[1:4]):
        for col,delta in enumerate([-5,0,5]):
            index=frame+delta;cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,img=cap.read()
            if not ok:raise RuntimeError(f'Cannot read video frame {index}')
            axes[row,col].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));axes[row,col].axis('off')
            state='closed' if a.sides['right'].gripper_binary[index] else 'open'
            axes[row,col].set_title(f'A{row} {delta:+d} frames / frame {index} / {state}')
    cap.release();fig.suptitle('Bowl ego source episode 1 / detected events +/- 5 frames')
    fig.savefig(out/'ego1_event_frames.png',dpi=140);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
