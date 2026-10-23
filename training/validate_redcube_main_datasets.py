"""Audit cotrain labels, frame alignment, masks, source sets and camera lengths."""
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
import cv2
import numpy as np
import pyarrow.parquet as pq
from training.build_unpaired_redcube_cotrain import read_ego
from training.build_arx_cotrain_datasets import CANONICAL_ARM,VIDEO_KEYS,read_json,write_json


def main():
    root=REPO/'outputs/unpaired_redcube_20261011';audit=REPO/'output/lerobot/main_redcube_audit'
    complete=read_json(audit/'complete.json');manifest=read_json(root/'corrected/manifest.json')['methods']['unpaired_bridge']
    paths={(variant,r['id']):Path(r['source']) if variant=='none' else REPO/r['output'] for r in manifest for variant in ['none','unpaired_bridge']}
    from training.build_arx_cotrain_datasets import ArxForwardKinematics,DEFAULT_SCENE,DEFAULT_CALIBRATION,joints_to_eef,apply_arm_mask,eef_mask
    fk=ArxForwardKinematics(DEFAULT_SCENE,DEFAULT_CALIBRATION);real_cache={}
    cache={key:read_ego(path) for key,path in paths.items()};video_cache={};comparisons={};summaries=[]
    for result in complete['results']:
        dataset=Path(result['dataset']);info=read_json(dataset/'meta/info.json');prov=read_json(dataset/'cotrain_provenance.json')
        variant=result['variant'];n=result['real_demonstrations'];e=result['ego_demonstrations'];frames=0;ids={'ego':set(),'real':set()};label_rows=0
        assert info['features']['observation.state']['shape']==[16] and info['features']['action']['shape']==[16]
        for row in prov['episodes']:
            ep=row['output_episode_index'];domain=row['source_domain'];source=row['source_episode_index'];ids[domain].add(source)
            t=pq.read_table(dataset/f'data/chunk-000/file-{ep:03d}.parquet')
            s=np.asarray(t['observation.state'].to_pylist(),np.float32);a=np.asarray(t['action'].to_pylist(),np.float32)
            f=np.asarray(t['policy.source_frame_index'].to_pylist());length=len(t)
            assert length==row['frames'] and np.array_equal(f,np.arange(row['source_frame_range'][0],row['source_frame_range'][1]+1))
            assert np.isfinite(s).all() and np.isfinite(a).all()
            assert np.allclose(s[:,:8],CANONICAL_ARM) and np.allclose(a[:,:8],CANONICAL_ARM)
            for key in ['policy.state_mask','policy.action_mask']:
                assert np.all(np.asarray(t[key].to_pylist())==np.array([False]*8+[True]*8))
            assert np.all(np.asarray(t['policy.image_mask'].to_pylist())==np.array([True,domain=='real',domain=='real']))
            assert np.all(np.asarray(t['policy.domain'].to_pylist())==(0 if domain=='ego' else 1))
            assert np.allclose(np.linalg.norm(s[:,11:15],axis=1),1,atol=1e-5)
            assert np.allclose(np.linalg.norm(a[:,11:15],axis=1),1,atol=1e-5)
            assert np.isin(s[:,15],[0,1]).all() and np.isin(a[:,15],[0,1]).all()
            assert np.array_equal(a[:-1],s[1:])
            if domain=='ego':
                original,valid=cache[variant,source];assert np.all(valid[f]&valid[f+1])
                np.testing.assert_allclose(s,original[f],rtol=0,atol=1e-7)
                np.testing.assert_allclose(a,original[f+1],rtol=0,atol=1e-7);label_rows+=length
            if domain=='real':
                if source not in real_cache:
                    original_table=pq.read_table(Path(complete['plan']['source'])/f'data/chunk-000/file-{source:03d}.parquet')
                    real_cache[source]=apply_arm_mask(joints_to_eef(np.asarray(original_table['observation.state'].to_pylist(),np.float32),fk),eef_mask(('right',)))
                original=real_cache[source]
                np.testing.assert_allclose(s,original[f],rtol=0,atol=1e-7)
                np.testing.assert_allclose(a,original[f+1],rtol=0,atol=1e-7)
            for key in VIDEO_KEYS:
                video=dataset/'videos'/key/'chunk-000'/f'file-{ep:03d}.mp4'
                st=video.stat();identity=(st.st_dev,st.st_ino)
                if identity not in video_cache:
                    cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ok,img=cap.read();cap.release()
                    assert ok and img is not None,video
                    video_cache[identity]=count
                assert video_cache[identity]==length+1,(video,video_cache[identity],length)
            frames+=length
        assert len(ids['ego'])==e and len(ids['real'])==n
        assert ids['ego']==(set(complete['plan']['ego_ids'][str(e)]) if e else set())
        assert ids['real']==set(complete['plan']['real_ids'][str(n)])
        assert frames==info['total_frames'] and len(prov['episodes'])==info['total_episodes']
        comparisons[variant,e,n]=(dataset,prov)
        summaries.append(dict(dataset=str(dataset),source_ego=e,source_real=n,output_episodes=info['total_episodes'],frames=frames,ego_rows_compared_to_json=label_rows))
        print(json.dumps(summaries[-1]),flush=True)
    for e in [20,50,100]:
        n=20
        a,pa=comparisons['none',e,n];b,pb=comparisons['unpaired_bridge',e,n]
        assert pa['episodes']==pb['episodes']
        for row in pa['episodes']:
            ep=row['output_episode_index'];ta=pq.read_table(a/f'data/chunk-000/file-{ep:03d}.parquet');tb=pq.read_table(b/f'data/chunk-000/file-{ep:03d}.parquet')
            for column in ta.column_names:
                if column in ['observation.state','action'] and row['source_domain']=='ego':continue
                assert ta[column].equals(tb[column]),(n,ep,column)
    write_json(audit/'validation.json',dict(passed=True,datasets=summaries,unique_videos_checked=len(video_cache),
        before_after_same_source_frames_and_masks=True,before_after_real_labels_identical=True,ego_labels_match_native_json=True))

if __name__=='__main__':main()
