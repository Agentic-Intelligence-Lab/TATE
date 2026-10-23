from pathlib import Path
import tempfile
import unittest

import numpy as np

from preprocess.MintHands import remap_keypoints, build_mint_cache
from preprocess.PipelineIO import export_eef_from_cache, write_json


class MintPipelineTest(unittest.TestCase):
    def test_anatomical_mapping_and_palm(self):
        points = np.arange(63, dtype=float).reshape(21, 3)
        result = remap_keypoints(points)
        for aria, openpose in [(0, 4), (1, 8), (5, 0), (6, 2), (8, 5), (11, 9), (17, 17)]:
            np.testing.assert_array_equal(result[aria], points[openpose])
        np.testing.assert_array_equal(result[20], points[[0, 5, 9]].mean(axis=0))

    def test_presence_nan_and_timing(self):
        points = np.ones((3, 21, 3))
        bad = points.copy()
        bad[2, 0, 0] = np.nan
        cache = build_mint_cache(dict(left=bad, right=points),
                                [[0.9, 0.1], [0.1, 0.9], [0.9, 0.9]],
                                fps=30, width=640, height=480, K=np.eye(3), d=np.zeros(5),
                                source_video='test.mp4', metadata={})
        self.assertIsNone(cache['frames'][0]['hand_r'])
        self.assertIsNone(cache['frames'][1]['hand_l'])
        self.assertIsNone(cache['frames'][2]['hand_l'])
        self.assertIsNotNone(cache['frames'][2]['hand_r'])
        self.assertEqual(cache['frames'][2]['ts'], 66666667)

    def test_same_keypoints_produce_same_eef_for_both_backends(self):
        rng = np.random.default_rng(8)
        points = rng.normal(scale=0.03, size=(4, 21, 3))
        points[..., 2] += 0.6
        cache = build_mint_cache(dict(left=points, right=points), np.ones((4, 2)),
                                fps=30, width=640, height=480, K=np.eye(3), d=np.zeros(5),
                                source_video='test.mp4', metadata={})
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            outputs = []
            for backend in ['mint', 'wilor']:
                cache['schema'] = f'tate.{backend}_hands'
                path = write_json(cache, directory / f'{backend}.json')
                outputs.append(export_eef_from_cache(path,
                    root / 'cfg/preprocess/base/RealSenseD405.yaml', directory / f'{backend}_eef.json'))
            self.assertEqual(outputs[0]['frames'], outputs[1]['frames'])
            self.assertIsNone(outputs[0]['source_wilor_cache'])
            self.assertIsNotNone(outputs[1]['source_wilor_cache'])


if __name__ == '__main__':
    unittest.main()
