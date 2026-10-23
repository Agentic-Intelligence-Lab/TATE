"""Calibration-only model selection and direct held-out paired evaluation.

The protocol must be frozen before examining test errors. Hyperparameters are
selected by inner episode-level CV on the calibration cohort, never test labels.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import platform
import scipy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import Delaunay, QhullError

from correction.paired_transport import PairedCalibrator
from scripts.random_position_experiment import load_dataset,change_anchors,global_rotation
from correction.scene_transport import SceneTrajectoryTransport


def errors(p,r,target_p,target_r,mask=None):
    d=np.linalg.norm(p-target_p,axis=2)
    angle=np.degrees((Rotation.from_matrix(r.reshape(-1,3,3)).inv()*Rotation.from_matrix(target_r.reshape(-1,3,3))).magnitude()).reshape(d.shape)
    mask=np.ones(d.shape,dtype=bool) if mask is None else mask
    path=np.sum(d*mask,axis=1)/mask.sum(1)
    rotation=np.sum(angle*mask,axis=1)/mask.sum(1)
    return dict(path_m=path,grasp_m=d[:,40],release_m=d[:,60],
                contact_m=(d[:,40]+d[:,60])/2,rotation_deg=rotation,
                grasp_rotation_deg=angle[:,40],release_rotation_deg=angle[:,60],
                selection_position_m=.5*path+.25*(d[:,40]+d[:,60]))


def observed_phase_mask(record):
    times=np.asarray(record['raw']['t'])[np.asarray(record['raw']['valid'])]
    grid=np.interp(np.linspace(0,4,81),np.arange(5),record['knots'])
    index=np.clip(np.searchsorted(times,grid,side='right'),1,len(times)-1)
    inside=(grid>times[index-1]+1e-9)&(grid<times[index]-1e-9)
    return ~((times[index]-times[index-1]>.25)&inside)


def arc_resample(p):
    parts=[]
    for start in [0,20,40,60]:
        q=p[start:start+21];length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
        if length[-1]<1e-9:out=np.repeat(q[:1],21,axis=0)
        else:
            u,idx=np.unique(length/length[-1],return_index=True)
            out=np.column_stack([np.interp(np.linspace(0,1,21),u,q[idx,j]) for j in range(3)])
        parts.append(out if start==60 else out[:-1])
    return np.concatenate(parts)


def stats(values):
    values=np.asarray(values)
    return dict(mean=float(values.mean()),median=float(np.median(values)),p95=float(np.quantile(values,.95)))


def bootstrap_improvement(before,after):
    delta=np.asarray(before)-np.asarray(after)
    rng=np.random.default_rng(183)
    means=np.mean(delta[rng.integers(0,len(delta),(3000,len(delta)))],axis=1)
    return dict(mean=float(delta.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                improved_fraction=float(np.mean(delta>0)))


def calibration_search(ep,er,scene,rp,rr,masks=None):
    folds=np.random.default_rng(52).permutation(len(ep))%5
    candidates=[]
    for feature in ['bias','scene','eef','hybrid']:
        for alpha,retain,smooth,basis in itertools.product(([1.] if feature=='bias' else [.1,1.,10.,100.]),[0.,.5,1.],[0.,1.],(['linear'] if feature=='bias' else ['linear','rbf'])):
            params=dict(features=feature,ridge=alpha,retain=retain,smooth=smooth,basis=basis)
            position=[];orientation=[]
            for fold in range(5):
                train=folds!=fold;valid=folds==fold
                model=PairedCalibrator(**params).fit(ep[train],er[train],scene[train],rp[train],rr[train])
                p,r=model.predict(ep[valid],er[valid],scene[valid])
                e=errors(p,r,rp[valid],rr[valid],None if masks is None else masks[valid])
                position.extend(e['selection_position_m']);orientation.extend(e['rotation_deg'])
            candidates.append(dict(parameters=params,position_cv_m=float(np.mean(position)),rotation_cv_deg=float(np.mean(orientation))))
    best_p=min(candidates,key=lambda c:c['position_cv_m'])
    best_r=min(candidates,key=lambda c:c['rotation_cv_deg'])
    return best_p,best_r,candidates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(args.protocol.read_text())
    cohorts,exclusions,data=load_dataset(args.dataset)
    by_kind={k:{rec['id']:rec for rec in group['records']} for k,group in cohorts.items()}
    common=sorted(set(by_kind['real'])&set(by_kind['ego']))
    missing=[i for i in range(100) if i not in common]
    if set(protocol['calibration_pairs'])&set(protocol['test_pairs']):raise ValueError('Split leakage')
    ids=np.array(common)
    train=np.isin(ids,protocol['calibration_pairs']);test=np.isin(ids,protocol['test_pairs'])
    if train.sum()<10:raise ValueError('Too few healthy fixed calibration pairs')
    ep=np.array([by_kind['ego'][i]['p'] for i in ids]);er=np.array([by_kind['ego'][i]['r'] for i in ids])
    rp=np.array([by_kind['real'][i]['p'] for i in ids]);rr=np.array([by_kind['real'][i]['r'] for i in ids])
    scene=np.array([by_kind['ego'][i]['scene'] for i in ids])
    real_scene=np.array([by_kind['real'][i]['scene'] for i in ids])
    masks=np.array([observed_phase_mask(by_kind['ego'][i]) & observed_phase_mask(by_kind['real'][i]) for i in ids])
    # No test target is passed into search or into fit calls below.
    best_p,best_r,cv=calibration_search(ep[train],er[train],scene[train],rp[train],rr[train],masks[train])
    (args.out/'calibration_cv.json').write_text(json.dumps(dict(best_position=best_p,best_rotation=best_r,candidates=cv),indent=2))
    print('CV selection',best_p,best_r,flush=True)
    pos_model=PairedCalibrator(**best_p['parameters']).fit(ep[train],er[train],scene[train],rp[train],rr[train])
    rot_model=PairedCalibrator(**best_r['parameters']).fit(ep[train],er[train],scene[train],rp[train],rr[train])
    pos,_=pos_model.predict(ep,er,scene);_,rot=rot_model.predict(ep,er,scene)
    preds={'raw':(ep,er),'paired_selected':(pos,rot)}
    for feature in ['eef','scene']:
        candidates=[c for c in cv if c['parameters']['features']==feature]
        chosen=min(candidates,key=lambda c:c['position_cv_m'])
        ablation=PairedCalibrator(**chosen['parameters']).fit(ep[train],er[train],scene[train],rp[train],rr[train])
        preds['paired_'+feature+'_only']=ablation.predict(ep,er,scene)
    base_r=global_rotation(er[train],rr[train],er)
    a=np.array([0,20,40,60,80]);rmean=rp[train][:,a].mean(0)
    preds['fixed_anchor_mean']=(change_anchors(ep,np.broadcast_to(rmean,(len(ep),5,3))),base_r)
    preds['anchor_shift']=(change_anchors(ep,ep[:,a]+rmean-ep[train][:,a].mean(0)),base_r)
    mean_residual=PairedCalibrator(features='bias',retain=1.,smooth=1.).fit(ep[train],er[train],scene[train],rp[train],rr[train])
    preds['paired_mean_residual']=mean_residual.predict(ep,er,scene)
    visual=PairedCalibrator(features='scene',ridge=1.,retain=0.,smooth=0.).fit(ep[train],er[train],scene[train],rp[train],rr[train])
    vp,vr=visual.predict(ep,er,scene)
    preds['visual_anchor']=(change_anchors(ep,vp[:,a]),base_r)
    preds['visual_mean']=(vp,vr)
    # Previous unpaired-style transport, using only the same 20 calibration pairs.
    old=SceneTrajectoryTransport().fit(ep[train],er[train],scene[train],rp[train],rr[train],real_scene[train])
    preds['previous_scene_transport']=old.predict(ep,er,scene)
    # Paired global affine position / fixed tool-rotation offset baseline.
    x=np.column_stack([ep[train].reshape(-1,3),np.ones(train.sum()*81)])
    coef=np.linalg.lstsq(x,rp[train].reshape(-1,3),rcond=None)[0]
    affine=(np.column_stack([ep.reshape(-1,3),np.ones(len(ep)*81)])@coef).reshape(ep.shape)
    preds['paired_global_affine']=(affine,base_r)
    report=dict(dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
                protocol=protocol,calibration_ids=ids[train].tolist(),test_ids=ids[test].tolist(),
                missing_or_unhealthy_pairs=missing,exclusions=exclusions,
                selected_position=best_p,selected_rotation=best_r,metrics={},per_pair={})
    all_arrays=dict(ids=ids,calibration_mask=train,test_mask=test,real_p=rp,real_r=rr,ego_p=ep,ego_r=er,scene=scene)
    all_arrays['observed_phase_mask']=masks
    report['long_gap_masked_pairs']=[dict(pair_id=int(i),masked_phase_points=int((~m).sum())) for i,m in zip(ids,masks) if not m.all()]
    error_cache={}
    pixel_delta=np.linalg.norm((scene-real_scene)*100,axis=1)
    quality_test=test&(pixel_delta<=6.)
    report['image_consistent_test_ids']=ids[quality_test].tolist()
    try:
        in_hull=Delaunay(scene[train]).find_simplex(scene)>=0
    except QhullError:
        in_hull=np.zeros(len(scene),dtype=bool)
    report['scene_extrapolation_test_ids']=ids[test & ~in_hull].tolist()
    report['image_inconsistent_pairs']=[dict(pair_id=int(i),relative_cube_pixel_delta=float(d),split='calibration' if a else 'test') for i,d,a in zip(ids,pixel_delta,train) if d>6.]
    for method,(p,r) in preds.items():
        e=errors(p,r,rp,rr,masks)
        e['arc_length_path_m']=np.array([np.linalg.norm(arc_resample(pp)-arc_resample(target),axis=1).mean() for pp,target in zip(p,rp)])
        error_cache[method]=e
        report['metrics'][method]={split:{key:stats(value[mask]) for key,value in e.items()}
                                   for split,mask in [('calibration',train),('test',test)]}
        if quality_test.any():
            report['metrics'][method]['image_consistent_test']={key:stats(value[quality_test]) for key,value in e.items()}
        for subset,subset_mask in [('scene_interpolation_test',test & in_hull),('scene_extrapolation_test',test & ~in_hull)]:
            if subset_mask.any():
                report['metrics'][method][subset]={key:stats(value[subset_mask]) for key,value in e.items()}
        report['metrics'][method]['test_contact_tolerances']={f'{int(cm)}cm':dict(
            grasp=float(np.mean(e['grasp_m'][test]<=cm/100)),release=float(np.mean(e['release_m'][test]<=cm/100)),
            both=float(np.mean((e['grasp_m'][test]<=cm/100)&(e['release_m'][test]<=cm/100)))) for cm in [1,2,5]}
        report['per_pair'][method]=[dict(pair_id=int(i),split='calibration' if is_train else 'test',**{key:float(value[j]) for key,value in e.items()}) for j,(i,is_train) in enumerate(zip(ids,train))]
        all_arrays[method+'_p']=p;all_arrays[method+'_r']=r
    report['bootstrap_vs_raw']={key:bootstrap_improvement(error_cache['raw'][key][test],error_cache['paired_selected'][key][test]) for key in ['path_m','grasp_m','release_m','rotation_deg']}
    # Pairing audit uses image observations only, not test TCPs for fitting.
    report['pairing_pixel_delta']=stats(np.linalg.norm((scene-real_scene)*100,axis=1))
    report['notes']=['Primary errors use corresponding real episode and fixed event-normalized time.',
        'Arc-length alignment is a supplementary geometry metric, not primary selection.',
        'No test labels used in feature scaling, reference rotations, coefficients, or hyperparameter selection.',
        'Same initial object pose does not make free-space human and robot motion identical.']
    (args.out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    bundle=dict(schema='tate.paired_calibration_bundle',schema_version=1,position=pos_model.artifact(),rotation=rot_model.artifact(),
                calibration_pair_ids=ids[train].tolist(),test_pair_ids=ids[test].tolist(),dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest())
    bundle['fallback_global_affine']=coef.tolist()
    bundle['fallback_local_rotation']=global_rotation(er[train],rr[train],np.eye(3)).tolist()
    (args.out/'model.json').write_text(json.dumps(bundle,indent=2))
    np.savez_compressed(args.out/'predictions.npz',**all_arrays)
    for method,m in report['metrics'].items():print(method,{k:round(v['mean'],5) for k,v in m['test'].items()},flush=True)


if __name__=='__main__':main()
