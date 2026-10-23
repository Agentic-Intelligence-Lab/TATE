"""Plot sealed unpaired results and export PRIMARY run; never pick best test run."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,Slerp
from scripts.random_position_experiment import load_dataset,GRID
from correction.unpaired_scene_bridge import SceneBridge


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();protocol=json.loads((args.root/'protocol.json').read_text())
    run=args.root/str(protocol['primary_seed']);z=np.load(run/'predictions.npz')
    results=json.loads((run/'results.json').read_text());summary=json.loads((args.root/'summary.json').read_text())
    names=protocol['methods'];labels=['Raw','Anchor shift','Scene residual','Scene mean','Prior unpaired OT','Unpaired bridge']
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');x=np.arange(len(names))
    for shift,key,label in [(-.25,'grasp_m','Grasp'),(0,'release_m','Release'),(.25,'path_m','Path')]:
        axes[0].bar(x+shift,[results['metrics'][m]['heldout80'][key]['mean']*100 for m in names],.25,label=label)
    axes[0].set(ylabel='Mean paired-reference error (cm)',title='Primary fixed run: 80 held-out robot references');axes[0].legend()
    axes[1].bar(x,[results['metrics'][m]['heldout80']['rotation_deg']['mean'] for m in names]);axes[1].set(ylabel='SO(3) angle (degrees)',title='No pairing available during fitting')
    for ax in axes:ax.set_xticks(x,labels,rotation=30,ha='right')
    fig.savefig(args.root/'unpaired_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,key,mult in zip(axes,['grasp_m','path_m'],[100,100]):
        for method,label in zip(names[1:],labels[1:]):
            values=[json.loads((args.root/str(seed)/'results.json').read_text())['metrics'][method]['heldout80'][key]['mean']*mult for seed in protocol['seeds']]
            ax.plot(range(1,6),values,'o-',label=label)
        ax.set(xlabel='Independent 20-real reference draw',ylabel='Mean error (cm)',title=key);ax.legend(fontsize=8)
    fig.savefig(args.root/'unpaired_reference_sensitivity.png',dpi=160);plt.close(fig)
    raw=json.load(gzip.open(args.root/'ego_native.json.gz'))
    cohorts,excluded,_=load_dataset({'records':{'ego':raw}});assert not excluded
    recs=cohorts['ego']['records'];model_bytes=(run/'model.json').read_bytes();model=SceneBridge.from_artifact(json.loads(model_bytes))
    p,r=model.predict(np.array([v['p'] for v in recs]),np.array([v['r'] for v in recs]),np.array([v['scene'] for v in recs]))
    np.testing.assert_allclose(p,z['unpaired_bridge_p'],atol=1e-12);np.testing.assert_allclose(r,z['unpaired_bridge_r'],atol=1e-12)
    meta=json.loads((run/'export_metadata.json').read_text())
    payload=dict(schema='tate.experimental_correction_payload',experiment_family='unpaired_scene_bridge',exclusions=[],
        scene_rule=protocol.get('scene_rule','initial red minus blue centroid'),
        model_sha256=hashlib.sha256(model_bytes).hexdigest(),calibration=meta['real_episode_ids'],ego_calibration=meta['ego_episode_ids'],methods={'unpaired_bridge':[]})
    for i,rec in enumerate(recs):
        row=rec['raw'];times=np.asarray(row['t']);phase=rec['phase']
        pp=CubicSpline(GRID,p[i],bc_type='natural')(phase);rr=Slerp(GRID,Rotation.from_matrix(r[i]))(phase).as_matrix()
        velocity=np.gradient(pp,times,axis=0);rot=Rotation.from_matrix(rr)
        omega=(rot[1:]*rot[:-1].inv()).as_rotvec()/np.diff(times)[:,None]
        payload['methods']['unpaired_bridge'].append(dict(id=row['episode_id'],mode='unpaired_bridge',source=row['path'],source_sha256=row['sha256'],
            p=pp.tolist(),r=rr.tolist(),v=velocity.tolist(),w=np.vstack([np.zeros((1,3)),omega]).tolist()))
    (args.root/'native_payload.json.gz').write_bytes(gzip.compress(json.dumps(payload,allow_nan=False).encode()))
    print('Exported primary run:',len(recs),'ego trajectories; native inference matches sealed predictions')


if __name__=='__main__':main()
