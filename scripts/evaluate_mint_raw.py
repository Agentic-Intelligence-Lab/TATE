"""Reconstruct MINT and compare uncorrected EEF with WiLoR using TATE evaluation.

Uses an existing experiment only for its source episodes and raw WiLoR EEF.
No IK, robot correction, or LeRobot repackaging is needed for this comparison.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path

from preprocess.MintHands import MintReconstructor
from preprocess.PipelineIO import export_eef_from_cache, load_yaml, write_json
from preprocess.batch.dataset import discover_episodes
from evaluation import run_eval as ev
from evaluation.loaders import load_eef_json
from evaluation.schemas import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--real-manifest', required=True)
    p.add_argument('--real-split', required=True)
    p.add_argument('--eval-config', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--mint-repo', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--hand-mode', choices=['hard', 'blend', 'smooth'], default='blend')
    a = p.parse_args()
    if a.limit is not None and a.limit <= 0:
        p.error('--limit must be positive')
    a.experiment = a.experiment.resolve()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output / '.run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f'Another run is using {a.output}') from exc
    config = load_yaml(a.experiment / 'resolved_experiment.yaml')
    original = json.loads((a.experiment / 'manifest.json').read_text())
    baseline = original['variants']['finger_center_hys085__none']
    sources = {str(s.episode_index): s for s in discover_episodes(config['source']['ego_dataset'], config['source']['video_key'])}
    ids = sorted((k for k, v in baseline['episodes'].items() if v['status'] == 'complete'), key=int)
    if a.limit is not None:
        ids = ids[:a.limit]
    manifest = {k: copy.deepcopy(original[k]) for k in
                ['schema', 'schema_version', 'source_ego_dataset', 'source_real_dataset', 'active_sides', 'arm_mode']}
    manifest['experiment_id'] = a.output.name
    manifest['variants'] = {name: dict(id=name, correction_config={'mode': 'none'}, episodes={})
                            for name in ['wilor_raw', 'mint_raw']}
    reconstructor = None
    health = {}
    eval_config = ev._merge(ev.DEFAULT_CONFIG, load_yaml(a.eval_config))
    calibration = config['geometry']['camera_calibration']
    hcfg = config['hand2gripper_variants']['finger_center_hys085']
    if not ids:
        raise ValueError('No complete WiLoR baseline episodes')
    write_json(dict(
        source_experiment=str(a.experiment), requested_episodes=ids,
        real_manifest=str(Path(a.real_manifest).resolve()),
        real_split=str(Path(a.real_split).resolve()),
        eval_config=str(Path(a.eval_config).resolve()), hand2gripper=hcfg,
        camera_calibration=calibration, hand_mode=a.hand_mode,
        fingerprints={str(path): sha256_file(path) for path in
                      [a.experiment / 'manifest.json', Path(calibration), Path(hcfg['eef_config']),
                       Path(a.real_manifest), Path(a.real_split), Path(a.eval_config)]},
        implementation_sha256={str(path): sha256_file(path) for path in
                               [Path('preprocess/MintHands.py'), Path('preprocess/PipelineIO.py'),
                                Path('preprocess/Hand2Gripper.py'), Path(__file__)]}),
        a.output / 'run_config.json')
    for eid in ids:
        src = sources[eid]
        directory = a.output / 'episodes' / f'{int(eid):06d}'
        cache_path, eef_path = directory / 'mint_hands.json', directory / 'eef_raw.json'
        if not a.evaluate_only:
            if cache_path.exists():
                raise FileExistsError(f'{cache_path}: use a new output directory or --evaluate-only; stale caches are not silently reused')
            if reconstructor is None:
                reconstructor = MintReconstructor(a.mint_repo, a.checkpoint, hand_mode=a.hand_mode)
            trim = config.get('trajectory', {}).get('trim', {})
            src = src.with_trim_seconds(trim.get('start_seconds', 0), trim.get('end_seconds', 0))
            cache = reconstructor.reconstruct(src.video_path, load_yaml(calibration),
                                              start_frame=src.processing_start_frame,
                                              max_frames=src.effective_length)
            if abs(cache['fps'] - src.fps) > 1e-3:
                raise ValueError(f'Episode {eid}: video FPS differs from dataset FPS')
            cache['temporal_window'] = src.temporal_window()
            cache['source_episode_identity'] = src.identity()
            write_json(cache, cache_path)
            export_eef_from_cache(cache_path, calibration, eef_path,
                                 eef_config_path=hcfg['eef_config'],
                                 grasp_close_ratio=hcfg['grasp_close_ratio'],
                                 grasp_open_ratio=hcfg['grasp_open_ratio'],
                                 grasp_min_frames=hcfg['grasp_min_frames'],
                                 hand2gripper_mode=hcfg['mode'], grasp_mode=hcfg['grasp_mode'],
                                 arm_mode=original['arm_mode'], active_sides=original['active_sides'])
        health[eid] = {}
        for name, path in [('wilor_raw', Path(baseline['episodes'][eid]['final_eef'])), ('mint_raw', eef_path)]:
            if name == 'wilor_raw':
                # Validate the original provenance before using the canonical EEF.
                trajectory = ev._load_variant_trajectory(a.experiment / 'manifest.json', original['experiment_id'],
                                                         'finger_center_hys085__none', baseline, eid)
            else:
                trajectory = load_eef_json(path, episode_id=eid)
            manifest['variants'][name]['episodes'][eid] = dict(status='complete', final_eef=str(path),
                                                               final_eef_fingerprint=sha256_file(path))
            health[eid][name] = {side: ev._health_reason(trajectory, side, eval_config)
                                 for side in original['active_sides']}
        print(f'Completed episode {eid}', flush=True)
        write_json(manifest, a.output / 'manifest.json')
        write_json(health, a.output / 'health.json')
    # Report native coverage first; then compare on the same healthy episodes
    # per arm. Bimanual common cohort requires both arms healthy in both methods.
    common = [eid for eid in ids if all(health[eid][name][side][0] is None
                                      for name in manifest['variants'] for side in original['active_sides'])]
    write_json(dict(requested=ids, common_healthy=common,
                    source_manifest=str(a.experiment / 'manifest.json'),
                    source_manifest_sha256=sha256_file(a.experiment / 'manifest.json')),
               a.output / 'cohort.json')
    from scripts.evaluate_five_task_corrections import compile_dtw
    write_json(compile_dtw(), a.output / 'dtw_validation.json')
    ev._compute_eta_dispersion = lambda ego, sides, config, floors: {
        side: dict(available=False, value=None, reason='Not requested') for side in sides}
    for label, episodes in [('all', ids), ('common', common)]:
        if not episodes:
            continue
        try:
            ev.run(argparse.Namespace(experiment_manifest=str(a.output / 'manifest.json'),
                                  real_manifest=a.real_manifest, real_split=a.real_split,
                                  eval_config=a.eval_config, out=str(a.output / label),
                                  variants=None, episodes=episodes, sides=None, segment=None,
                                  pairing=None, limit=None, dry_run=False, resume=True, force=False))
        except RuntimeError as exc:
            if 'produced no evaluable ego-real pairs' not in str(exc):
                raise
            write_json({'available': False, 'reason': str(exc)}, a.output / label / 'unavailable.json')


if __name__ == '__main__':
    main()
