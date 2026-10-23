"""Plot true paired errors and export native-frame correction payload for all ego."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.random_position_experiment import load_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args()
    cohorts,exclusions,data=load_dataset(args.dataset)
    z=np.load(args.experiment/'predictions.npz')
    report=json.loads((args.experiment/'results.json').read_text())
    bundle=json.loads((args.experiment/'model.json').read_text())
    maps={kind:{rec['id']:rec for rec in rows} for kind,rows in data['records'].items()}
    scene={kind:np.array([np.array(maps[kind][int(i)]['landmarks']['red']['uv'])-np.array(maps[kind][int(i)]['landmarks']['blue']['uv']) for i in z['ids']]) for kind in ['real','ego']}
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for real,ego in zip(scene['real'],scene['ego']):
        axes[0].plot([real[0],ego[0]],[real[1],ego[1]],color='gray',alpha=.25)
    for kind,color in [('real','#333333'),('ego','#d85c52')]:
        axes[0].scatter(*scene[kind].T,s=16,label=kind,color=color)
    axes[0].set(title='Initial cube relative to plate',xlabel='u offset (pixels)',ylabel='v offset (pixels)');axes[0].legend()
    for mask,label,color in [(z['test_mask'],'Held-out ego','#aaaaaa'),(z['calibration_mask'],'Calibration ego','#248e79')]:
        axes[1].scatter(*scene['ego'][mask].T,s=25,label=label,color=color)
    axes[1].set(title='Frozen calibration coverage',xlabel='u offset (pixels)',ylabel='v offset (pixels)');axes[1].legend()
    for ax in axes[:2]:ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim')
    delta=np.linalg.norm(scene['real']-scene['ego'],axis=1)
    axes[2].hist(delta,bins=20,color='#357f88');axes[2].axvline(6,color='#d85c52',ls='--',label='6 px subset threshold')
    axes[2].set(title='Within-pair placement difference',xlabel='Relative cube displacement (pixels)',ylabel='Pairs');axes[2].legend()
    fig.savefig(args.experiment/'paired_scene_audit.png',dpi=170);plt.close(fig)
    methods=['raw','fixed_anchor_mean','anchor_shift','previous_scene_transport','visual_anchor','paired_global_affine','paired_selected']
    labels=['Raw','Fixed anchors','Anchor shift','Previous transport','Visual anchors','Paired affine','Selected paired']
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    x=np.arange(len(methods))
    for offset,metric,label in [(-.25,'grasp_m','Grasp'),(0,'release_m','Release'),(.25,'path_m','Path')]:
        axes[0].bar(x+offset,[report['metrics'][m]['test'][metric]['mean']*100 for m in methods],.25,label=label)
    axes[0].set_xticks(x,labels,rotation=35,ha='right');axes[0].set(ylabel='Paired position MAE (cm)',title='Direct paired real-reference errors');axes[0].legend()
    axes[1].bar(x,[report['metrics'][m]['test']['rotation_deg']['mean'] for m in methods],color='#357f88')
    axes[1].set_xticks(x,labels,rotation=35,ha='right');axes[1].set(ylabel='Mean geodesic angle (degrees)',title='Paired orientation errors')
    before=np.array([r['grasp_m'] for r in report['per_pair']['raw'] if r['split']=='test'])*100
    after=np.array([r['grasp_m'] for r in report['per_pair']['paired_selected'] if r['split']=='test'])*100
    axes[2].scatter(before,after,s=25,alpha=.75)
    limit=max(before.max(),after.max())*1.05;axes[2].plot([0,limit],[0,limit],'--',color='gray')
    axes[2].set(xlabel='Before grasp error (cm)',ylabel='After grasp error (cm)',title='Each point is one held-out pair',xlim=(0,limit),ylim=(0,limit))
    fig.suptitle(f"Paired redcube: {len(report['calibration_ids'])} calibration / {len(report['test_ids'])} held-out pairs\nHyperparameters selected only inside calibration cohort; event-phase alignment")
    fig.savefig(args.experiment/'paired_comparison.png',dpi=170);fig.savefig(args.experiment/'paired_comparison.svg');plt.close(fig)
    test=np.flatnonzero(z['test_mask']);chosen=test[np.linspace(0,len(test)-1,4).astype(int)]
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for col,index in enumerate(chosen):
        for key,label,color in [('real','Paired real','#333333'),('raw','Raw ego','#d85c52'),('paired_selected','Corrected ego','#248e79')]:
            p=z[key+'_p'][index]*100
            for row,(a,b) in enumerate([(0,1),(0,2)]):
                axes[row,col].plot(p[:,a],p[:,b],label=label,color=color)
                axes[row,col].scatter(p[[40,60],a],p[[40,60],b],s=20,color=color)
        axes[0,col].set(title=f"Held-out pair {z['ids'][index]}: XY",xlabel='x (cm)',ylabel='y (cm)')
        axes[1,col].set(title='XZ',xlabel='x (cm)',ylabel='z (cm)')
    axes[0,0].legend(fontsize=8);fig.savefig(args.experiment/'paired_trajectories.png',dpi=170);plt.close(fig)
    from scripts.paired_redcube_apply import apply
    payload,metrics=apply(bundle,data,hashlib.sha256((args.experiment/'model.json').read_bytes()).hexdigest())
    (args.experiment/'native_payload.json.gz').write_bytes(gzip.compress(json.dumps(payload,allow_nan=False).encode()))
    (args.experiment/'native_metrics.json').write_text(json.dumps(metrics,indent=2))
    print('Native outputs',len(metrics['episodes']),'fallback pairs',metrics['fallback_pairs'])


if __name__=='__main__':main()
