"""Validate materialized bowl EEF and evaluate the exported native-frame poses."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay
from scripts.random_position_experiment import load_dataset
from scripts.paired_redcube_experiment import errors, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    protocol = json.loads((root/'protocol.json').read_text())
    run = root/str(protocol['primary_seed'])
    manifest = json.loads((root/'ego_corrected/manifest.json').read_text())
    rows = manifest['methods']['unpaired_bridge']
    assert len(rows) == 100 and len({r['id'] for r in rows}) == 100
    assert len(list((root/'ego_raw').glob('*/eef_raw.json'))) == 100
    lookup = {row['id']: row for row in rows}
    raw_records = json.load(gzip.open(root/'ego_native.json.gz'))
    corrected_records = []
    totals = dict(episodes=100, frames=0, valid_frames=0, changed_position_frames=0, changed_rotation_frames=0)
    for rec in raw_records:
        row = lookup[rec['episode_id']]
        source_bytes = Path(row['source']).read_bytes()
        assert hashlib.sha256(source_bytes).hexdigest() == rec['sha256']
        corrected_bytes = Path(row['output']).read_bytes()
        assert hashlib.sha256(corrected_bytes).hexdigest() == row['sha256']
        source = json.loads(source_bytes)
        corrected = json.loads(corrected_bytes)
        assert len(source['frames']) == len(corrected['frames']) == row['frames']
        assert corrected['correction']['cross_domain_pairing_used'] is False
        assert len(corrected['correction']['real_calibration_episode_ids']) == 20
        p, r = [], []
        for before, after in zip(source['frames'], corrected['frames']):
            totals['frames'] += 1
            assert before.get('timestamp_s') == after.get('timestamp_s')
            b, a = before.get('hand_r'), after.get('hand_r')
            if b is None or not b.get('valid', True):
                assert a == b
                p.append([0., 0., 0.]); r.append(np.eye(3).tolist())
                continue
            totals['valid_frames'] += 1
            assert a.get('valid', True) == b.get('valid', True)
            assert a['grasp_state'] == b['grasp_state']
            t, old = np.asarray(a['tcp_pose_eef_frame']), np.asarray(b['tcp_pose_eef_frame'])
            assert np.isfinite(t).all()
            np.testing.assert_allclose(t[3], [0, 0, 0, 1], atol=1e-12)
            np.testing.assert_allclose(t[:3,:3].T@t[:3,:3], np.eye(3), atol=1e-8)
            np.testing.assert_allclose(np.linalg.det(t[:3,:3]), 1., atol=1e-8)
            np.testing.assert_allclose(a['tcp_pose_eef_frame_before_scene_transport'], old, atol=0)
            np.testing.assert_allclose(a['eef_pose_world'], t, atol=0)
            camera = np.asarray(corrected['arm_transforms']['T_cam_in_output_frame']['right'])
            np.testing.assert_allclose(camera@np.asarray(a['tcp_pose_cam']), t, atol=1e-10)
            totals['changed_position_frames'] += int(np.linalg.norm(t[:3,3]-old[:3,3]) > 1e-9)
            totals['changed_rotation_frames'] += int(np.linalg.norm(t[:3,:3]-old[:3,:3]) > 1e-9)
            p.append(t[:3,3].tolist()); r.append(t[:3,:3].tolist())
        corrected_records.append(dict(rec, p=p, r=r))
    cohort, excluded, _ = load_dataset({'records': {'ego': corrected_records}})
    assert not excluded
    c = cohort['ego']
    v = np.load(run/'evaluation_private.npz')
    z = np.load(run/'predictions.npz')
    t = np.load(run/'training.npz')
    test = v['heldout_mask']
    err = errors(c['p'], c['r'], v['real_p'], v['real_r'], v['observed_mask'])
    raw_err = errors(z['raw_p'], z['raw_r'], v['real_p'], v['real_r'], v['observed_mask'])
    scene_distance = np.linalg.norm(t['source_scene']-v['real_scene'], axis=1)*100
    inside = Delaunay(t['target_scene']).find_simplex(t['source_scene']) >= 0
    report = dict(validation=totals, all_checks_passed=True,
        native_export_metrics={split: {key: stats(value[mask]) for key,value in err.items()}
            for split,mask in [('heldout80', test), ('all100', np.ones(100,bool))]},
        raw_metrics={key: stats(value[test]) for key,value in raw_err.items()},
        export_resampling_difference_m=stats(np.linalg.norm(c['p']-z['unpaired_bridge_p'],axis=-1).ravel()),
        scene_pair_difference_px=stats(scene_distance),
        scene_pair_difference_over_6px=int((scene_distance>6).sum()),
        ego_outside_real20_scene_hull=int((~inside).sum()),
        phase_samples_excluded=int((~v['observed_mask']).sum()),
        primary_seed=protocol['primary_seed'],
        real_episode_ids=json.loads((run/'export_metadata.json').read_text())['real_episode_ids'])
    (root/'export_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
