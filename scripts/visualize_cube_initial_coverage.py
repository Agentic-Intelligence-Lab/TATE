"""Overlay every cube's first-frame image centroid on a fixed reference frame."""
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


def detect(image):
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    mask = (((h > 164) | (h < 5)) & (s > 85) & (v > 65)).astype('uint8')
    height, width = mask.shape
    mask[:int(.25*height)] = 0; mask[int(.8*height):] = 0
    mask[:, :int(.15*width)] = 0; mask[:, int(.85*width):] = 0
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    candidates = [i for i in range(1, len(stats)) if stats[i, cv2.CC_STAT_AREA] > 100]
    if not candidates:
        raise ValueError('No cube detected in first frame')
    return centers[max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--real-count', type=int, default=100)
    parser.add_argument('--ego-count', type=int, default=100)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.manifest.read_text()); rows = []; background = None
    counts = {'real': args.real_count, 'ego': args.ego_count}
    records = [r for domain in ['real','ego'] for r in
               sorted([r for r in records if r['domain']==domain], key=lambda r:r['episode_id'])[:counts[domain]]]
    for rec in records:
        cap = cv2.VideoCapture(rec['video']); ok, frame = cap.read(); cap.release()
        if not ok:
            raise ValueError(rec['video'])
        uv = detect(frame)
        if background is None:
            background = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(args.out/'reference_first_frame.png'), frame)
        rows.append(dict(domain=rec['domain'], episode_id=rec['episode_id'],
                         u=float(uv[0]), v=float(uv[1]), video=rec['video'], frame_index=0))
    groups = {k: np.array([[r['u'],r['v']] for r in rows if r['domain']==k]) for k in ['real','ego']}
    assert all(len(x)==counts[k] for k,x in groups.items())
    colors = {'real':'#00b9ef', 'ego':'#ff8c16'}
    labels = {'real':f'Real robot (first {args.real_count})', 'ego':f'Ego (first {args.ego_count})'}

    def draw(ax, domains):
        ax.imshow(background)
        for domain in domains:
            p = groups[domain]; color = colors[domain]
            hull = ConvexHull(p); boundary = p[np.r_[hull.vertices,hull.vertices[0]]]
            ax.plot(*boundary.T, color=color, lw=1.6, linestyle='--')
            if domain == 'real':
                ax.scatter(*p.T, s=32, facecolors='none', edgecolors=color, linewidths=1.3, label=labels[domain])
            else:
                ax.scatter(*p.T, s=27, c=color, marker='+', linewidths=1.3, label=labels[domain])
        ax.set(xlim=(-.5,639.5),ylim=(479.5,-.5),xlabel='u (pixels)',ylabel='v (pixels)')
        ax.legend(loc='upper left',facecolor='white',framealpha=.94,fontsize=9)

    fig, ax = plt.subplots(figsize=(10,8),layout='constrained')
    draw(ax,['real','ego'])
    ax.set_title(f'Cube initial centers: first {args.real_count} real + first {args.ego_count} ego\nBackground: episode 000000, frame 0; dashed lines: observed convex hulls')
    for ext in ['png','pdf']:
        fig.savefig(args.out/f'cube_initial_centers_all.{ext}',dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(18,5.5),layout='constrained')
    for ax, domains, title in zip(axes,[['real'],['ego'],['real','ego']],['Real robot','Ego','Overlay']):
        draw(ax,domains); ax.set_title(title)
    fig.suptitle('First-frame cube centroids in common image coordinates; same background in all panels')
    fig.savefig(args.out/'cube_initial_centers_by_domain.png',dpi=180); plt.close(fig)
    with (args.out/'centers.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = dict(reference=records[0]['video'],reference_frame=0,
                   selection='First N episodes per domain, sorted by original episode ID; not sampled training subsets',
                   selected_episode_ids={k:[r['episode_id'] for r in records if r['domain']==k] for k in counts},
                   center_definition='Centroid of largest red connected component in first frame',
                   coordinates='Raw image pixels: origin top left, u right, v down; no registration',domains={})
    for domain,p in groups.items():
        summary['domains'][domain] = dict(count=len(p),u_range=p[:,0].min().item(),
            u_max=p[:,0].max().item(),v_min=p[:,1].min().item(),v_max=p[:,1].max().item(),
            convex_hull_area_px2=float(ConvexHull(p).volume))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
