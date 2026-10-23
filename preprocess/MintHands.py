"""MINT camera-space reconstruction, using the upstream MANO decoder.

MINT is an optional external checkout; importing this module needs only NumPy.
No camera-motion prediction or fitted robot correction is applied to the hands.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np


OPENPOSE_TO_ARIA = [4, 8, 12, 16, 20, 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 0]


def remap_keypoints(joints):
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape[-2:] != (21, 3) or not np.isfinite(joints).all():
        raise ValueError("Expected finite OpenPose joints (..., 21, 3) in metres")
    result = joints[..., OPENPOSE_TO_ARIA, :].copy()
    result[..., 20, :] = joints[..., [0, 5, 9], :].mean(axis=-2)
    return result


def build_mint_cache(joints, probabilities, *, fps, width, height, K, d,
                     source_video, metadata, presence_threshold=0.5):
    if not np.isfinite(fps) or fps <= 0 or not 0 <= presence_threshold <= 1:
        raise ValueError("Invalid fps or presence threshold")
    probabilities = np.asarray(probabilities)
    n = len(joints['left'])
    if probabilities.shape != (n, 2) or len(joints['right']) != n:
        raise ValueError("MINT hands/presence/frame counts disagree")
    frames = []
    for i in range(n):
        frame = dict(idx=i, ts=round(i * 1e9 / fps))
        for column, side in enumerate(('left', 'right')):
            points = np.asarray(joints[side][i])
            prob = float(probabilities[i, column])
            valid = (np.isfinite(prob) and prob >= presence_threshold
                     and np.isfinite(points).all() and (points[:, 2] > 0).all())
            frame['hand_l' if side == 'left' else 'hand_r'] = (
                dict(is_right=side == 'right', confidence=prob,
                     keypoints_3d_cam=remap_keypoints(points).tolist(),
                     depth_source='mint_metric_mano') if valid else None)
        frames.append(frame)
    return dict(schema='tate.mint_hands', schema_version=1,
                source_video=str(source_video), total_frames=n, fps=float(fps),
                width=int(width), height=int(height), K=np.asarray(K).tolist(),
                d=np.asarray(d).tolist(),
                coordinate_frame='opencv_camera_x_right_y_down_z_forward',
                keypoint_order='aria_humanego_21',
                reconstruction={'backend': 'mint', 'presence_threshold': presence_threshold,
                                **metadata}, frames=frames)


class MintReconstructor:
    def __init__(self, repo, checkpoint, *, device='cuda:0', window=32,
                 hand_mode='blend', presence_threshold=0.5):
        self.repo = Path(repo).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not (self.repo / 'configs/training/mint_step2.yaml').is_file():
            raise FileNotFoundError(f'Not a compatible MINT checkout: {self.repo}')
        if window <= 0 or not 0 <= presence_threshold <= 1:
            raise ValueError('Invalid window size or presence threshold')
        if hand_mode not in ('hard', 'blend', 'smooth'):
            raise ValueError('Unknown MINT hand mode')
        sys.path.insert(0, str(self.repo))
        sys.path.insert(0, str(self.repo / 'eval/model_effect'))
        from mint.inference.engine import StudentEngine
        from visualization.reproj_core import mano
        mano.ensure_mano_weights()
        self.mano = mano
        self.engine = StudentEngine(str(self.repo / 'configs/training/mint_step2.yaml'),
                                    ckpt=str(self.checkpoint), device=device, window=window)
        self.hand_mode = hand_mode
        self.threshold = presence_threshold
        with self.checkpoint.open('rb') as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        self.metadata = dict(
            upstream_commit=subprocess.check_output(['git', '-C', str(self.repo), 'rev-parse', 'HEAD'], text=True).strip(),
            checkpoint=str(self.checkpoint),
            checkpoint_sha256=digest.hexdigest(),
            window=window, hand_mode=hand_mode, camera_mode='chunked',
            units='metres', hand_frame='camera', camera_motion_applied=False)

    def reconstruct(self, video, calibration, *, start_frame=0, max_frames=None):
        import cv2
        from preprocess.PipelineIO import select_intrinsics
        if start_frame < 0 or (max_frames is not None and max_frames <= 0):
            raise ValueError('Invalid start frame or frame count')
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f'Cannot open {video}')
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            while max_frames is None or len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not frames or (max_frames is not None and len(frames) != max_frames):
            raise ValueError('Video did not contain the requested frame interval')
        height, width = frames[0].shape[:2]
        K, d, profile = select_intrinsics(calibration, width, height)
        prediction = self.engine.predict(np.stack(frames), cam_mode='chunked', hand_mode=self.hand_mode)
        hand = np.asarray(prediction['hand'])
        if hand.shape != (len(frames), 218):
            raise ValueError(f'Unexpected MINT hand shape: {hand.shape}')
        logits = np.asarray(prediction['hand_presence_logits'])
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        joints = {}
        for side, offset in [('left', 0), ('right', 109)]:
            part = hand[:, offset:offset + 109]
            chunks = []
            for start in range(0, len(part), 128):
                p = part[start:start + 128]
                decoded = self.mano.decode_hand_6d(p[:, :3], p[:, 3:9], p[:, 9:99], p[:, 99:109], is_right=side == 'right')
                _, j = self.mano.run_mano(**decoded, is_right=side == 'right')
                j[~np.isfinite(p).all(axis=1)] = np.nan
                chunks.append(j[:, :21])
            joints[side] = np.concatenate(chunks)
        return build_mint_cache(joints, probabilities, fps=fps, width=width, height=height,
                                K=K, d=d, source_video=video,
                                metadata={**self.metadata, 'intrinsics_profile': profile,
                                          'source_start_frame': start_frame},
                                presence_threshold=self.threshold)
