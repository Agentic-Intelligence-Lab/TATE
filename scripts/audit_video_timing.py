#!/usr/bin/env python3
"""Audit original/exported/visualization video timing without changing data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq

TASKS = {'bowl':'bowl_1','hanging':'hanging_1','drawer':'drawer_1012'}


def probe(path):
    result=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=r_frame_rate,avg_frame_rate,time_base,duration,nb_frames',
        '-of','json',str(path)],text=True))['streams'][0]
    return dict(path=str(path),fps=float(Fraction(result['avg_frame_rate'])),
                frames=int(result['nb_frames']),duration_s=float(result['duration']),
                time_base=result['time_base'])


def read_metadata(root):
    info=json.loads((root/'meta/info.json').read_text())
    rows={}
    for f in sorted((root/'meta/episodes').rglob('*.parquet')):
        for row in pq.read_table(f).to_pylist():rows[int(row['episode_index'])]=row
    return info,rows


def video_path(root,info,row,key):
    prefix=f'videos/{key}/'
    return root/info['video_path'].format(video_key=key,chunk_index=int(row[prefix+'chunk_index']),
                                         file_index=int(row[prefix+'file_index']))


def data_path(root,info,row):
    return root/info['data_path'].format(chunk_index=int(row['data/chunk_index']),file_index=int(row['data/file_index']))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def stats(values):
    a=np.array(values,dtype=float)
    return dict(min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),max=float(a.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/data/xule/TATE'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    all_probes=[];comparisons=[];summaries={};pts_checks=[];source_clocks=[]
    for task,experiment in TASKS.items():
        exp=args.root/'outputs/experiments'/experiment
        manifest=json.loads((exp/'manifest.json').read_text())
        source=Path(manifest['source_ego_dataset']['path']);info,episodes=read_metadata(source)
        real_manifest=json.loads((args.root/f'outputs/arx_real_flange/{task}_arx/manifest.json').read_text())
        real_root=Path(real_manifest['dataset_root']);ri,re=read_metadata(real_root)
        jobs=[]
        for kind,root,inf,rows in [('ego_source',source,info,episodes),('real_source',real_root,ri,re)]:
            for eid,row in rows.items():jobs.append((kind,eid,video_path(root,inf,row,'observation.images.head')))
        for eid in episodes:
            vis=exp/f'artifacts/visualizations/finger_center_hys085/episode_{eid:06d}/wilor_eef_vis.mp4'
            if vis.exists():jobs.append(('wilor_visualization',eid,vis))
        with ThreadPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(lambda job:dict(task=task,kind=job[0],source_episode=job[1],**probe(job[2])),jobs))
        all_probes.extend(results)
        by_kind={kind:{r['source_episode']:r for r in results if r['kind']==kind}
                 for kind in ['ego_source','real_source','wilor_visualization']}
        raw_clock_error=0.;export_clock_error=0.;index_errors=[]
        for eid,row in episodes.items():
            p=data_path(source,info,row)
            t=pq.read_table(p,columns=['episode_index','frame_index','timestamp']).to_pydict()
            indices=np.flatnonzero(np.array(t['episode_index'])==eid)
            fi=np.array(t['frame_index'])[indices];ts=np.array(t['timestamp'])[indices]
            raw_clock_error=max(raw_clock_error,float(np.max(np.abs(ts-fi/info['fps']))))
            assert len(indices)==int(row['length'])
            assert by_kind['ego_source'][eid]['frames']==len(indices)
            assert np.array_equal(fi,np.arange(len(fi)))
        for variant in ['none','position_rotation']:
            dst=exp/'datasets'/('finger_center_hys085__'+variant);di,dr=read_metadata(dst)
            provenance=json.loads((dst/'meta/tate_preprocess.json').read_text())
            keys=[k for k,v in di['features'].items() if v.get('dtype')=='video']
            for ep in provenance['episodes']:
                sid=int(ep['source_episode_index']);did=int(ep['derived_episode_index'])
                row=dr[did];srcrow=episodes[sid]
                d=pq.read_table(data_path(dst,di,row),columns=['tate.source_frame_index','tate.source_timestamp','timestamp']).to_pydict()
                n=len(d['timestamp']);correct_indices=np.array_equal(d['tate.source_frame_index'],np.arange(n))
                if not correct_indices:index_errors.append([variant,sid])
                export_clock_error=max(export_clock_error,float(np.max(np.abs(np.array(d['timestamp'])-np.arange(n)/di['fps']))))
                assert n==int(srcrow['length'])==ep['length']
                for key in keys:
                    src=video_path(source,info,srcrow,key);dest=video_path(dst,di,row,key)
                    s1,s2=src.stat(),dest.stat();same_inode=(s1.st_dev,s1.st_ino)==(s2.st_dev,s2.st_ino)
                    identical=same_inode or (s1.st_size==s2.st_size and sha(src)==sha(dest))
                    comparisons.append(dict(task=task,variant=variant,source_episode=sid,export_episode=did,
                        camera=key,source=str(src),export=str(dest),same_inode=same_inode,byte_identical=identical,
                        source_frames=int(srcrow['length']),export_rows=n,source_indices_contiguous=correct_indices,
                        source_start_s=srcrow[f'videos/{key}/from_timestamp'],export_start_s=row[f'videos/{key}/from_timestamp'],
                        source_stop_s=srcrow[f'videos/{key}/to_timestamp'],export_stop_s=row[f'videos/{key}/to_timestamp']))
        # PTS and actual decoded count on fixed sample episodes, not chosen by apparent speed.
        for eid in [min(episodes),sorted(episodes)[len(episodes)//2],max(episodes)]:
            for kind in ['ego_source','wilor_visualization']:
                p=by_kind[kind][eid]['path']
                frames=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                    '-show_frames','-show_entries','frame=best_effort_timestamp_time', '-of','json',p],text=True))['frames']
                ts=np.array([float(f['best_effort_timestamp_time']) for f in frames])
                pts_checks.append(dict(task=task,kind=kind,source_episode=eid,decoded_frames=len(ts),
                    frame_delta_s=stats(np.diff(ts)),max_grid_error_s=float(np.max(np.abs(ts-ts[0]-np.arange(len(ts))/30)))))
        sample_eef=json.loads((exp/'artifacts/variants/finger_center_hys085__none/episode_000000/eef.json').read_text())
        cache=Path(sample_eef['source_wilor_cache']);c=json.loads(cache.read_text());cf=c['frames']
        source_clocks.append(dict(task=task,source_columns=pq.read_schema(data_path(source,info,episodes[0])).names,
             metadata_fps=info['fps'],cache_fps=c.get('fps'),cache_frame_count=len(cf),
             cache_indices_contiguous=[f['idx'] for f in cf]==list(range(len(cf))),
             eef_timestamp_examples=[{k:f.get(k) for k in ['idx','ts','timestamp_s']} for f in sample_eef['frames'][:3]],
             raw_timestamp_max_error_from_frame_over_fps_s=raw_clock_error,
             export_timestamp_max_error_from_frame_over_fps_s=export_clock_error))
        visual_matches={eid:(v['frames']==by_kind['ego_source'][eid]['frames'] and v['fps']==by_kind['ego_source'][eid]['fps']
            and abs(v['duration_s']-by_kind['ego_source'][eid]['duration_s'])<1e-5)
            for eid,v in by_kind['wilor_visualization'].items()}
        comps=[r for r in comparisons if r['task']==task]
        summaries[task]=dict(ego_episodes=len(episodes),real_episodes=len(re),
            source_ego_duration_s=stats([r['duration_s'] for r in by_kind['ego_source'].values()]),
            source_real_duration_s=stats([r['duration_s'] for r in by_kind['real_source'].values()]),
            video_fps=sorted(set(r['fps'] for r in results)),
            visualizations_checked=len(visual_matches),visualizations_timing_identical=sum(visual_matches.values()),
            exported_videos_checked=len(comps),exported_videos_byte_identical=sum(r['byte_identical'] for r in comps),
            exported_videos_same_inode=sum(r['same_inode'] for r in comps),index_errors=index_errors,
            variant_episode_counts={v:len({r['source_episode'] for r in comps if r['variant']==v}) for v in ['none','position_rotation']},
            source_episode0=by_kind['ego_source'][0])
        print(task,json.dumps(summaries[task]),flush=True)
    for name,rows in [('video_streams.csv',all_probes),('export_source_identity.csv',comparisons)]:
        with (args.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(summary=summaries,source_clock_checks=source_clocks,pts_checks=pts_checks,
                acquisition_wall_clock_available=False,
                limitation='Source timestamps are frame-index/FPS; they cannot establish actual acquisition cadence.')
    (args.output/'audit.json').write_text(json.dumps(result,indent=2))
    assert all(r['byte_identical'] for r in comparisons),'Some exports differ from source bytes; inspect CSV'
    assert all(v['visualizations_checked']==v['visualizations_timing_identical'] for v in summaries.values())
    print('DONE',flush=True)


if __name__=='__main__':main()
