"""Paired-truth stress test with known artificial corruption (not real ego GT)."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from scripts.random_position_experiment import GRID,load_dataset,predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    cohorts,_,_=load_dataset(args.dataset);real=cohorts["real"]
    runs=[]
    for seed in range(5):
        rng=np.random.default_rng(seed+829)
        order=rng.permutation(100);fit,test=order[:20],order[20:]
        truth=real["p"][test];rot=real["r"][test];scene=real["scene"][test]
        source=truth.copy()
        source[:,:,0]-=.10+.015*np.sin(GRID*np.pi/2)
        source[:,:,1]=.7*(source[:,:,1]-.2)+.2-.01
        source[:,:,2]-=.06+.01*GRID
        noise=gaussian_filter1d(rng.normal(size=source.shape),2,axis=1)
        noise*=.008/np.std(noise)
        source+=noise
        bias=Rotation.from_rotvec([.2,-.25,.1]).as_matrix()
        source_r=rot@bias
        pred,_=predictions(source,source_r,scene,real["p"][fit],real["r"][fit],real["scene"][fit],source,source_r,scene,seed)
        metrics={}
        for name,(p,r) in pred.items():
            position=np.linalg.norm(p-truth,axis=2)
            angular=(Rotation.from_matrix(r.reshape(-1,3,3)).inv()*Rotation.from_matrix(rot.reshape(-1,3,3))).magnitude()
            metrics[name]=dict(grasp_true_error_m=float(position[:,40].mean()),
                path_true_error_m=float(position.mean()),rotation_true_error_deg=float(np.degrees(angular).mean()))
        runs.append(dict(seed=seed,fit_ids=real["ids"][fit].tolist(),test_ids=real["ids"][test].tolist(),metrics=metrics))
    aggregate={name:{key:dict(mean=float(np.mean(vals)),std=float(np.std(vals)))
                    for key in runs[0]["metrics"][name]
                    for vals in [[run["metrics"][name][key] for run in runs]]}
               for name in runs[0]["metrics"]}
    args.out.write_text(json.dumps(dict(protocol="Known additive drift, y scale .7, 8mm smooth noise, fixed 19.2deg local orientation bias applied to real holdout; synthetic only",runs=runs,aggregate=aggregate),indent=2))
    for name,row in aggregate.items(): print(name,{k:round(v["mean"],4) for k,v in row.items()})


if __name__=="__main__":main()
