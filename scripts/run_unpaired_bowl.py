"""Run the frozen bowl protocol after all independent EEF workers finish.

Run on the data server from the repository root. No policy learning.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    prep = Path('/mnt/data/xule/TATE/outputs/paired_bowl_20261016')
    out = Path('/mnt/data/xule/TATE/outputs/unpaired_bowl_20261016')
    if (out/'protocol.json').exists():
        raise FileExistsError('An experiment already exists; inspect before resuming.')
    deadline = time.monotonic()+7200
    while True:
        paths = sorted(prep.glob('preprocess/*/artifacts/eef/finger_center_hys085/*/eef_raw.json'))
        print(f'EEF complete: {len(paths)}/100', flush=True)
        if len(paths) == 100:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('EEF workers did not finish within two hours')
        time.sleep(30)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLCONFIGDIR='/tmp/tate-bowl-mpl')

    def run(module, *args, stdout=None):
        command = [sys.executable, '-m', module, *map(str,args)]
        print('RUN', command, flush=True)
        subprocess.run(command, check=True, env=env, stdout=stdout)

    dataset = prep/'dataset.json.gz'
    with dataset.open('wb') as stream:
        run('scripts.paired_bowl_extract', stdout=stream)
    run('scripts.unpaired_redcube_prepare', '--dataset', dataset, '--out', out,
        '--first-seed', 20261016,
        '--scene-rule', 'initial yellow bowl minus pink bowl centroid / 100 pixels',
        '--benchmark-note', 'New bowl task; SceneBridge parameters frozen before paired evaluation. Only single-domain image/event preprocessing adapted.')
    for seed in range(20261016,20261021):
        run('scripts.unpaired_redcube_fit', '--training', out/str(seed)/'training.npz', '--out', out/str(seed))
    # All predictions must be sealed before any paired error is computed.
    run('scripts.unpaired_redcube_evaluate', '--root', out)
    run('scripts.unpaired_redcube_artifacts', '--root', out)
    run('scripts.random_position_export', '--payload', out/'native_payload.json.gz', '--out', out/'ego_corrected')
    for source in paths:
        dest = out/'ego_raw'/source.parent.name/'eef_raw.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(source)
    run('scripts.visualize_unpaired_example', '--root', out, '--pairing', prep/'pairing_verified.json')
    run('scripts.validate_unpaired_bowl', '--root', out)
    run('scripts.plot_unpaired_bowl_audit', '--root', out)
    print('Pipeline complete: 100 raw EEF links and 100 corrected EEF files.', flush=True)


if __name__ == '__main__':
    main()
