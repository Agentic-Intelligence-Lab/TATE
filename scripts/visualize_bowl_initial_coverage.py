"""Plot yellow-bowl first-frame coverage for full and chronological subsets."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scripts.paired_bowl_extract import detect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/mnt/data/xule/Data_TATE_1016/bowl_random'))
    parser.add_argument('--pairing', type=Path, default=Path('/mnt/data/xule/TATE/outputs/paired_bowl_20261016/pairing_verified.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    pairing = json.loads(args.pairing.read_text())['pairs']
    rows = []; background = None; groups = {}
    for domain in ['real', 'ego']:
        ids = sorted(p[domain+'_episode'] for p in pairing)
        assert len(ids) == len(set(ids)) == 100
        domain_rows = []
        for episode in ids:
            video = args.source/f'videos/observation.images.head/chunk-000/file-{episode:03d}.mp4'
            cap = cv2.VideoCapture(str(video)); ok, frame = cap.read(); cap.release()
            if not ok:
                raise RuntimeError(f'Cannot read first frame: {video}')
            center = detect(frame, 'red')  # Legacy slot: yellow moving bowl.
            if center is None:
                raise ValueError(f'Yellow bowl missing: {video}')
            if background is None:
                background = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(args.out/'reference_first_frame.png'), frame)
            row = dict(domain=domain, episode_id=episode, u=float(center[0]), v=float(center[1]),
                       frame_index=0, video=str(video))
            rows.append(row); domain_rows.append(row)
        groups[domain] = domain_rows
    specs = [
        ('real100', 'Real robot: all 100', [('real',100)]),
        ('ego100', 'Ego: all 100', [('ego',100)]),
        ('mixed_all', 'Overlay: 100 real + 100 ego', [('real',100),('ego',100)]),
        ('first_real20', 'Real robot: first 20', [('real',20)]),
        ('first_ego50', 'Ego: first 50', [('ego',50)]),
        ('mixed_first', 'Overlay: first 20 real + first 50 ego', [('real',20),('ego',50)]),
    ]
    colors = {'real':'#00b9ef', 'ego':'#ff8c16'}
    summary = dict(source=str(args.source), reference_video=rows[0]['video'], reference_frame=0,
                   selection='First N within each domain, ascending original episode ID; not sampled calibration/training subsets',
                   center_definition='Yellow-mask connected-component centroid in frame 0',
                   coordinates='Original image pixels, no registration; origin top left', subsets={})

    def draw(ax, spec):
        name, title, selections = spec
        ax.imshow(background)
        for domain, count in selections:
            chosen = groups[domain][:count]
            p = np.array([[r['u'],r['v']] for r in chosen]); color = colors[domain]
            hull = ConvexHull(p); boundary = p[np.r_[hull.vertices,hull.vertices[0]]]
            ax.plot(*boundary.T, color=color, ls='--', lw=1.5)
            label = f'{"Real robot" if domain=="real" else "Ego"} ({count})'
            if domain == 'real':
                ax.scatter(*p.T,s=30,facecolors='none',edgecolors=color,linewidths=1.2,label=label)
            else:
                ax.scatter(*p.T,s=28,c=color,marker='+',linewidths=1.3,label=label)
            summary['subsets'][f'{domain}{count}'] = dict(count=count,
                episode_ids=[r['episode_id'] for r in chosen],
                u_min=float(p[:,0].min()), u_max=float(p[:,0].max()),
                v_min=float(p[:,1].min()), v_max=float(p[:,1].max()),
                convex_hull_area_px2=float(hull.volume))
        height,width = background.shape[:2]
        ax.set(title=title, xlim=(-.5,width-.5), ylim=(height-.5,-.5), xlabel='u (pixels)', ylabel='v (pixels)')
        ax.legend(loc='upper left',facecolor='white',framealpha=.94,fontsize=9)

    for spec in specs:
        fig, ax = plt.subplots(figsize=(10,8),layout='constrained'); draw(ax,spec)
        fig.suptitle('Yellow bowl initial centers | Background: real episode 000000, frame 0\nDashed lines: observed convex hulls',fontsize=12)
        for ext in ['png','pdf']:
            fig.savefig(args.out/f'bowl_{spec[0]}.{ext}',dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(18,10),layout='constrained')
    for ax,spec in zip(axes.ravel(),specs):
        draw(ax,spec)
    fig.suptitle('Yellow bowl initial positions | Same first-frame background and pixel axes in every panel\nBlue circles: real robot; orange crosses: ego; dashed lines: observed convex hulls',fontsize=15)
    for ext in ['png','pdf']:
        fig.savefig(args.out/f'bowl_coverage_overview.{ext}',dpi=180)
    plt.close(fig)
    with (args.out/'centers.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='episode_ids'} for k,v in summary['subsets'].items()},indent=2))


if __name__ == '__main__':
    main()
