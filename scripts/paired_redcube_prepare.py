"""Prepare isolated paired-redcube FK and ego EEF jobs on the data server."""
import json
import os
from pathlib import Path
import subprocess

import yaml
import numpy as np
import pyarrow.parquet as pq


def main():
    repo=Path('/mnt/data/xule/TATE')
    source=Path('/mnt/data/xule/Data_TATE_0930/stack_redcube_arx_ego_1')
    root=repo/'outputs/paired_redcube_20261010'
    root.mkdir(exist_ok=True)
    if (root/'jobs.json').exists():
        raise FileExistsError('Jobs already initialized; inspect/resume existing jobs explicitly.')
    view=root/'real_source_view'
    (view/'meta').mkdir(parents=True,exist_ok=True)
    (view/'data/chunk-000').mkdir(parents=True,exist_ok=True)
    (view/'meta/info.json').symlink_to(source/'meta/info.json')
    info=json.loads((source/'meta/info.json').read_text())
    names=info['features']['observation.state']['names']
    joints=[names.index(f'right_joint_{j}') for j in range(1,7)]
    gripper=names.index('right_gripper')
    audit=[]
    for i in range(200):
        table=pq.read_table(source/f'data/chunk-000/file-{i:03d}.parquet',columns=['observation.state'])
        state=np.asarray(table['observation.state'].to_pylist())
        span=float(np.linalg.norm(np.ptp(state[:,joints],axis=0)))
        grip=float(np.ptp(state[:,gripper]))
        kind='real' if span>.1 and grip>1 else 'ego'
        audit.append(dict(episode=i,joint_span=span,gripper_span=grip,kind=kind))
    pairs=[]
    for i in range(100):
        group=audit[2*i:2*i+2]
        if {g['kind'] for g in group}!={'real','ego'}:
            raise ValueError(f'Pair {i} has ambiguous robot/ego roles: {group}')
        pairs.append(dict(pair_id=i,**{g['kind']+'_episode':g['episode'] for g in group}))
    (root/'pairing_verified.json').write_text(json.dumps(dict(pairs=pairs,episode_audit=audit),indent=2))
    for i in [p['real_episode'] for p in pairs]:
        (view/f'data/chunk-000/file-{i:03d}.parquet').symlink_to(source/f'data/chunk-000/file-{i:03d}.parquet')
    (root/'pairs.json').write_text(json.dumps(dict(source=str(source),pairs=pairs),indent=2))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    py='/mnt/data/envs/lifego/bin/python'
    jobs=[]
    commands=[('real_fk',[py,'-m','real_data.arx_lerobot_adapter','--input',str(view),'--out',str(root/'real_fk')])]
    for worker in range(4):
        config=dict(experiment_id=f'paired_redcube_w{worker}',
            source=dict(ego_dataset=str(source),video_key='observation.images.head'),
            trajectory=dict(arm_mode='single_arm',active_sides=['right'],trim=dict(start_seconds=0.,end_seconds=0.)),
            outputs=dict(cache_root=str(root/'wilor_cache'),experiment_root=str(root/'preprocess')),
            wilor=dict(id='wilor_mini_d405_v1',preprocess_config='cfg/preprocess/base/Preprocess.yaml',pretrained_dir='/mnt/data/xule/.cache/wilor_mini'),
            geometry=dict(camera_calibration='cfg/preprocess/base/RealSenseD405.yaml'),
            hand2gripper_variants=dict(finger_center_hys085=dict(mode='finger_center',grasp_mode='finger_center',eef_config='cfg/preprocess/base/EEFExport.yaml',grasp_close_ratio=.85,grasp_open_ratio=.9,grasp_min_frames=10)),
            correction_variants=dict(none=dict(mode='none')),
            runs=[dict(id='finger_center_hys085__none',hand2gripper='finger_center_hys085',correction='none')],
            retarget=dict(enabled=False),lerobot=dict(enabled=False))
        path=root/f'worker_{worker}.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
        ids=','.join(str(pairs[i]['ego_episode']) for i in range(worker,100,4))
        commands.append((f'ego_w{worker}',[py,'-m','preprocess.batch_preprocess','--config',str(path),'--episodes',ids,'--stages','wilor,eef']))
    for name,command in commands:
        log=root/f'{name}.log'
        with log.open('w') as stream:
            proc=subprocess.Popen(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        jobs.append(dict(name=name,pid=proc.pid,command=command,log=str(log)))
    (root/'jobs.json').write_text(json.dumps(jobs,indent=2))
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
