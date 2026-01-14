from __future__ import annotations
import os
import cv2
import numpy as np

def load_video_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0) if frames else np.empty((0,))

def load_npy_frames(npy_path: str) -> np.ndarray:
    arr = np.load(npy_path, allow_pickle=True)
    # Accept common shapes:
    # - (T,H,W) grayscale
    # - (T,H,W,C) color
    # - (H,W,T) or (H,W,C,T) -> try to move T to first dim
    if arr.ndim == 3:
        # assume (T,H,W) OR (H,W,T)
        if arr.shape[0] < 8 and arr.shape[-1] >= 8:
            arr = np.transpose(arr, (2,0,1))
        return arr
    if arr.ndim == 4:
        # assume (T,H,W,C) OR (H,W,C,T)
        if arr.shape[0] < 8 and arr.shape[-1] >= 8:
            arr = np.transpose(arr, (3,0,1,2))
        return arr
    raise ValueError(f"Unsupported npy array shape: {arr.shape} for {npy_path}")
