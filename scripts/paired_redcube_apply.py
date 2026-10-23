"""Apply a frozen paired model using ego observations only (no robot targets).

Input is the compact extract format. Any real records are ignored. Outputs a
native-frame payload consumable by scripts/random_position_export.py on server.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,Slerp

from correction.paired_transport import PairedCalibrator
from scripts.random_position_experiment import GRID,load_dataset


def apply(bundle,data,model_sha256):
    cohorts,exclusions,_=load_dataset({'records':{'ego':data['records']['ego']}})
    accepted={r['id']:r for r in cohorts['ego']['records']}
    pm=PairedCalibrator.from_artifact(bundle['position'])
    rm=PairedCalibrator.from_artifact(bundle['rotation'])
    pairs={p['pair_id']:p for p in data['pairing']['pairs']}
    payload=dict(schema='tate.experimental_correction_payload',exclusions=exclusions,
        experiment_family='paired_calibration',model_sha256=model_sha256,
        calibration=[pairs[i]['real_episode'] for i in bundle['calibration_pair_ids']],
        ego_calibration=[pairs[i]['ego_episode'] for i in bundle['calibration_pair_ids']],methods={'paired_selected':[]})
    metrics=[];fallback=[]
    for raw in data['records']['ego']:
        pair=raw['id'];times=np.asarray(raw['t']);valid=np.asarray(raw['valid']);v_idx=np.flatnonzero(valid)
        if len(v_idx)<2:raise ValueError(f'Pair {pair} has fewer than two valid frames')
        if pair in accepted:
            rec=accepted[pair]
            ep,er,scene=rec['p'][None],rec['r'][None],rec['scene'][None]
            pp,_=pm.predict(ep,er,scene);_,rr=rm.predict(ep,er,scene)
            p=CubicSpline(GRID,pp[0],bc_type='natural')(rec['phase'])
            r=Slerp(GRID,Rotation.from_matrix(rr[0]))(rec['phase']).as_matrix()
            mode='paired_selected'
        else:
            rawp=np.asarray(raw['p']);p0=np.column_stack([np.interp(times,times[v_idx],rawp[v_idx,k]) for k in range(3)])
            p=np.column_stack([p0,np.ones(len(p0))])@np.asarray(bundle['fallback_global_affine'])
            r0=Slerp(times[v_idx],Rotation.from_matrix(np.asarray(raw['r'])[v_idx]))(np.clip(times,times[v_idx[0]],times[v_idx[-1]])).as_matrix()
            r=r0@np.asarray(bundle['fallback_local_rotation']);mode='global_affine_fallback';fallback.append(pair)
        velocity=np.gradient(p,times,axis=0)
        rot=Rotation.from_matrix(r);omega=(rot[1:]*rot[:-1].inv()).as_rotvec()/np.diff(times)[:,None]
        payload['methods']['paired_selected'].append(dict(id=raw['episode_id'],pair_id=pair,mode=mode,
            source=raw['path'],source_sha256=raw['sha256'],p=p.tolist(),r=r.tolist(),v=velocity.tolist(),w=np.vstack([np.zeros((1,3)),omega]).tolist()))
        metrics.append(dict(pair_id=pair,mode=mode,frames=len(p),speed_p95_m_s=float(np.quantile(np.linalg.norm(velocity[valid],axis=1),.95)),
                            angular_speed_p95_rad_s=float(np.quantile(np.linalg.norm(omega,axis=1),.95))))
    return payload,dict(episodes=metrics,fallback_pairs=fallback)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    raw=args.model.read_bytes()
    payload,metrics=apply(json.loads(raw),json.load(gzip.open(args.dataset)),hashlib.sha256(raw).hexdigest())
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'native_payload.json.gz').write_bytes(gzip.compress(json.dumps(payload,allow_nan=False).encode()))
    (args.out/'native_metrics.json').write_text(json.dumps(metrics,indent=2))
    print('Native outputs',len(metrics['episodes']),'fallback pairs',metrics['fallback_pairs'])


if __name__=='__main__':main()
