"""Run only RGB-to-WiLoR reconstruction and write a reusable camera-frame cache."""

from __future__ import annotations

import argparse

from preprocess.Preprocess import Preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input ego RGB video")
    parser.add_argument("--session", required=True, help="Output session directory")
    parser.add_argument("--cfg", default="./cfg/preprocess/base/Preprocess.yaml")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--wilor-pretrained-dir", default=None)
    parser.add_argument("--video-vis", action="store_true", help="Also render the overlay MP4")
    args = parser.parse_args()
    runner = Preprocess(
        mps_path=args.session,
        cfg_path=args.cfg,
        video_path=args.video,
        export_video=False,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        wilor_pretrained_dir=args.wilor_pretrained_dir,
    )
    cache_path = runner.run_wilor()
    if args.video_vis:
        runner.run_cache_visualization(cache_path)


if __name__ == "__main__":
    main()
