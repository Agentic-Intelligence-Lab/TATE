"""Visualize one fixed held-out pair; selection never depends on error."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/unpaired_redcube_20261011'))
    parser.add_argument('--pair-id',type=int)
    parser.add_argument('--pairing',type=Path,default=Path('outputs/paired_redcube/pairing_verified.json'))
    args=parser.parse_args()
    protocol=json.loads((args.root/'protocol.json').read_text());run=args.root/str(protocol['primary_seed'])
    z=np.load(run/'predictions.npz');v=np.load(run/'evaluation_private.npz')
    eligible=np.flatnonzero(v['heldout_mask'])
    if args.pair_id is None:i=eligible[np.argmin(v['pair_ids'][eligible])]
    else:
        found=np.flatnonzero(v['pair_ids']==args.pair_id)
        if not len(found):raise ValueError('Pair missing')
        i=found[0]
        if not v['heldout_mask'][i]:raise ValueError('Choose a held-out reference pair')
    pair=int(v['pair_ids'][i]);ego=int(v['ego_episode_ids'][i])
    mapping=json.loads(args.pairing.read_text())
    real=next(r['real_episode'] for r in mapping['pairs'] if r['pair_id']==pair)
    out=args.root/'examples';out.mkdir(exist_ok=True)
    tracks=[('Real robot reference',v['real_p'][i]*100,'#303030'),
            ('Original ego',z['raw_p'][i]*100,'#dc6245'),
            ('Corrected ego (SceneBridge)',z['unpaired_bridge_p'][i]*100,'#008c95')]
    fig=plt.figure(figsize=(13,10),layout='constrained')
    axes=[fig.add_subplot(221,projection='3d'),fig.add_subplot(222),fig.add_subplot(223),fig.add_subplot(224)]
    for label,p,color in tracks:
        axes[0].plot(*p.T,color=color,lw=2,label=label)
        axes[0].scatter(*p[40],color=color,s=55,marker='o',edgecolors='white',linewidths=.6)
        axes[0].scatter(*p[60],color=color,s=55,marker='s',edgecolors='white',linewidths=.6)
        for ax,(a,b) in zip(axes[1:],[(0,1),(0,2),(1,2)]):
            ax.plot(p[:,a],p[:,b],color=color,lw=2)
            for index,marker in [(40,'o'),(60,'s')]:ax.scatter(p[index,a],p[index,b],color=color,s=50,marker=marker,edgecolors='white',linewidths=.6,zorder=5)
    allp=np.concatenate([p for _,p,_ in tracks]);lo=allp.min(0)-2;hi=allp.max(0)+2
    axes[0].set(xlabel='X (cm)',ylabel='Y (cm)',zlabel='Z (cm)',title='3D TCP path',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]))
    axes[0].set_box_aspect(hi-lo);axes[0].view_init(elev=26,azim=-58)
    for ax,(a,b),title in zip(axes[1:],[(0,1),(0,2),(1,2)],['XY projection','XZ projection','YZ projection']):
        ax.set(xlabel=f'{"XYZ"[a]} (cm)',ylabel=f'{"XYZ"[b]} (cm)',title=title,xlim=(lo[a],hi[a]),ylim=(lo[b],hi[b]))
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    handles=[Line2D([0],[0],color=c,lw=2,label=l) for l,_,c in tracks]
    handles += [Line2D([0],[0],marker=m,color='gray',ls='',label=l) for m,l in [('o','Grasp / close'),('s','Release / open')]]
    axes[0].legend(handles=handles,fontsize=8,loc='upper left')
    fig.suptitle(f'Unpaired calibration: held-out pair {pair} | real episode {real}, ego episode {ego}\n'
                 'Same coordinate frame; 81 event-aligned samples; axes in cm',fontsize=14)
    for ext in ['png','pdf']:
        fig.savefig(out/f'pair_{pair:03d}_trajectory.{ext}',dpi=180)
    plt.close(fig)
    result=dict(pair_id=pair,real_episode=real,ego_episode=ego,heldout=True,model='unpaired_bridge',seed=protocol['primary_seed'],
        selection='Smallest held-out pair ID' if args.pair_id is None else 'Explicit requested pair',metrics={})
    for method in ['raw','unpaired_bridge']:
        d=np.linalg.norm(z[method+'_p'][i]-v['real_p'][i],axis=1)*100
        angle=np.degrees((Rotation.from_matrix(v['real_r'][i]).inv()*Rotation.from_matrix(z[method+'_r'][i])).magnitude())
        result['metrics'][method]=dict(grasp_cm=float(d[40]),release_cm=float(d[60]),path_cm=float(d[v['observed_mask'][i]].mean()),orientation_deg=float(angle[v['observed_mask'][i]].mean()))
    (out/f'pair_{pair:03d}_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
