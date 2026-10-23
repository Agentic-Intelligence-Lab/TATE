"""Reconstruct a video with MINT into a reusable TATE camera-frame cache."""
import argparse
from pathlib import Path

from preprocess.MintHands import MintReconstructor
from preprocess.PipelineIO import load_yaml, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--mint-repo', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--calibration', default='cfg/preprocess/base/RealSenseD405.yaml')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--window', type=int, default=32)
    p.add_argument('--hand-mode', choices=['hard', 'blend', 'smooth'], default='blend')
    p.add_argument('--presence-threshold', type=float, default=0.5)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--max-frames', type=int)
    a = p.parse_args()
    reconstructor = MintReconstructor(a.mint_repo, a.checkpoint, device=a.device,
                                     window=a.window, hand_mode=a.hand_mode,
                                     presence_threshold=a.presence_threshold)
    cache = reconstructor.reconstruct(a.video, load_yaml(a.calibration),
                                      start_frame=a.start_frame, max_frames=a.max_frames)
    print(write_json(cache, Path(a.session) / 'preprocess/mint_hands.json'))


if __name__ == '__main__':
    main()
