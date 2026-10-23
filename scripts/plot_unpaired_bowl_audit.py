"""Plot fixed examples and held-out error profiles without selecting good cases."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.root/'protocol.json').read_text())
    run = args.root/str(protocol['primary_seed'])
    z = np.load(run/'predictions.npz'); v = np.load(run/'evaluation_private.npz')
    t = np.load(run/'training.npz'); held = np.flatnonzero(v['heldout_mask'])
    chosen = held[np.argsort(v['pair_ids'][held])[:9]]
    fig, axes = plt.subplots(3,3,figsize=(12,12),layout='constrained')
    for ax,i in zip(axes.ravel(), chosen):
        for name, p, color in [('Real', v['real_p'][i], '#303030'),
                               ('Raw ego', z['raw_p'][i], '#dc6245'),
                               ('Corrected ego', z['unpaired_bridge_p'][i], '#008c95')]:
            ax.plot(p[:,0]*100,p[:,1]*100,color=color,label=name)
            ax.scatter(p[[40,60],0]*100,p[[40,60],1]*100,color=color,s=15)
        ax.set(title=f"Held-out pair {v['pair_ids'][i]}", xlabel='X (cm)', ylabel='Y (cm)')
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('First nine held-out pair IDs; XY trajectories; no error-based selection')
    fig.savefig(args.root/'fixed_nine_examples.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for method,color,label in [('raw','#dc6245','Raw ego'),('unpaired_bridge','#008c95','Corrected ego')]:
        d = np.linalg.norm(z[method+'_p'][held]-v['real_p'][held],axis=-1)*100
        d = np.where(v['observed_mask'][held], d, np.nan)
        axes[0].plot(np.linspace(0,4,81),np.nanmean(d,axis=0),color=color,label=label)
        q = np.nanquantile(d,[.25,.75],axis=0)
        axes[0].fill_between(np.linspace(0,4,81),q[0],q[1],color=color,alpha=.15)
    axes[0].set(xticks=[0,1,2,3,4],xticklabels=['Start','Open','Grasp','Release','End'],
                ylabel='Position error (cm)',title='80 held-out references: mean and interquartile range')
    axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].scatter(*t['source_scene'].T*100,s=20,label='100 ego initial scenes',alpha=.65)
    axes[1].scatter(*t['target_scene'].T*100,s=65,marker='x',label='20 real calibration scenes')
    axes[1].set(xlabel='Yellow minus pink: horizontal pixels',ylabel='Yellow minus pink: vertical pixels',
                title='Scene coverage: calibration inputs only')
    axes[1].set_aspect('equal',adjustable='datalim'); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(args.root/'phase_error_and_scene_coverage.png',dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
