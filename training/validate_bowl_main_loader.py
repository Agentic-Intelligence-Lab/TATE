"""Smoke-test actual LeRobot decoding and action windows at episode boundaries."""
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / 'output/lerobot/main_bowl_audit'
VIDEO_KEYS = ['observation.images.' + camera for camera in ['head', 'left', 'right']]


def main():
    complete = json.loads((AUDIT / 'complete.json').read_text())
    results = []
    for result in complete['results']:
        root = Path(result['dataset'])
        ds = LeRobotDataset('local/' + root.name, root=root,
                            delta_timestamps={'action': [i / 30 for i in range(50)]},
                            video_backend='pyav')
        assert len(ds) == result['frames']
        provenance = json.loads((root / 'cotrain_provenance.json').read_text())
        offsets = np.cumsum([0] + [row['frames'] for row in provenance['episodes']])
        targets = {0, len(ds) - 1, len(ds) // 2}
        ego_frames = result['counts']['ego']['frames']
        if ego_frames:
            targets.update([ego_frames - 1, ego_frames])
        for i, row in enumerate(provenance['episodes']):
            if row['source_frame_range'][0] > 0:
                targets.update([int(offsets[i]), int(offsets[i + 1] - 1)])
        for index in sorted(targets):
            sample = ds[index]
            assert tuple(sample['observation.state'].shape) == (16,)
            assert tuple(sample['action'].shape) == (50, 16)
            assert torch.isfinite(sample['action']).all()
            for key in VIDEO_KEYS:
                assert sample[key].ndim == 3 and sample[key].shape[0] == 3
                assert torch.isfinite(sample[key]).all()
            episode = int(np.searchsorted(offsets, index, side='right') - 1)
            local = int(index - offsets[episode])
            table = pq.read_table(root / f'data/chunk-000/file-{episode:03d}.parquet')
            actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
            expected = actions[np.minimum(np.arange(local, local + 50), len(actions) - 1)]
            np.testing.assert_allclose(sample['action'].numpy(), expected, rtol=0, atol=1e-7)
        results.append(dict(dataset=str(root), length=len(ds), sampled_indices=sorted(targets),
                            action_horizon=50, three_camera_decode=True, episode_boundary_padding=True))
        print(json.dumps(results[-1]), flush=True)
    (AUDIT / 'loader_validation.json').write_text(json.dumps(dict(passed=True, results=results), indent=2))


if __name__ == '__main__':
    main()
