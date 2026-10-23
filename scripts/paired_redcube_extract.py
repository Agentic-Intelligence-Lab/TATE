"""Export compact paired FK/EEF and independent initial-image landmarks."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def detect(image,color):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);h,s,v=cv2.split(hsv)
    mask=((((h>164)|(h<5)) if color=='red' else ((h>85)&(h<115)))&(s>85)&(v>65)).astype('uint8')
    height,width=mask.shape
    mask[:int(.25*height)]=0;mask[int(.8*height):]=0
    mask[:,:int(.15*width)]=0;mask[:,int(.85*width):]=0
    n,_,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    good=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>100]
    if not good:return None
    i=max(good,key=lambda j:stats[j,cv2.CC_STAT_AREA])
    return centers[i]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    root=Path('/mnt/data/xule/TATE/outputs/paired_redcube_20261010')
    source=Path('/mnt/data/xule/Data_TATE_0930/stack_redcube_arx_ego_1')
    pairing=json.loads((root/'pairing_verified.json').read_text())
    by_pair={p['pair_id']:p for p in pairing['pairs']}
    output=dict(records={'real':[],'ego':[]},missing=[],pairing=pairing)
    for kind in ['real','ego']:
        for pair in range(100):
            episode=by_pair[pair][kind+'_episode']
            candidates=([root/f'{folder}/episode_{episode:06d}.json' for folder in ['real_fk','real_fk_swapped']] if kind=='real' else
                [root/f'preprocess/{folder}/artifacts/eef/finger_center_hys085/episode_{episode:06d}/eef_raw.json'
                 for folder in [f'paired_redcube_w{pair%4}','paired_redcube_swapped']])
            path=next((p for p in candidates if p.exists()),candidates[0])
            if not path.exists():
                output['missing'].append(dict(kind=kind,pair=pair,path=str(path)))
                continue
            raw=path.read_bytes();data=json.loads(raw)
            p=[];r=[];t=[];grip=[];valid=[];events=[];previous=None
            for i,frame in enumerate(data['frames']):
                arm=frame.get('arms',{}).get('right') if kind=='real' else frame.get('hand_r')
                good=arm is not None and arm.get('valid',True)
                pose=np.array(arm['T_tcp_in_output_frame' if kind=='real' else 'tcp_pose_eef_frame']) if good else np.eye(4)
                state=int(arm['gripper']['binary'] if kind=='real' else arm['grasp_state']) if good else -1
                time=float(frame.get('timestamp_s',i/data['fps']))
                p.append(pose[:3,3].tolist());r.append(pose[:3,:3].tolist());t.append(time);valid.append(good);grip.append(state)
                if good:
                    if previous is not None and previous[0]!=state:
                        events.append(dict(frame=i,transition='close' if state else 'open',ambiguous=i!=previous[1]+1 and time-previous[2]>.25))
                    previous=state,i,time
            video=source/f'videos/observation.images.head/chunk-000/file-{episode:03d}.mp4'
            cap=cv2.VideoCapture(str(video));points={'red':[],'blue':[]}
            for index in [0,5,10]:
                cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read()
                if not ok:raise RuntimeError(f'{video}:{index}')
                for color in points:
                    point=detect(image,color)
                    if point is not None:points[color].append(point)
            cap.release();landmarks={}
            for color,values in points.items():
                if not values:landmarks[color]=None;continue
                values=np.array(values);center=np.median(values,axis=0)
                landmarks[color]=dict(uv=center.tolist(),jitter_px=float(np.linalg.norm(values-center,axis=1).max()),observations=len(values))
            record=dict(id=pair,episode_id=int(episode),path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                        video=str(video),p=p,r=r,t=t,grip=grip,valid=valid,events=events,landmarks=landmarks,image_size=[640,480])
            if kind=='ego':
                record['camera']=dict(k=data['k'],distortion=data['d'],transform=data['arm_transforms']['T_cam_in_output_frame']['right'])
            output['records'][kind].append(record)
    if output['missing'] and not args.partial:raise RuntimeError(f"Still missing {len(output['missing'])} EEF files")
    sys.stdout.buffer.write(gzip.compress(json.dumps(output,allow_nan=False).encode()))


if __name__=='__main__':main()
