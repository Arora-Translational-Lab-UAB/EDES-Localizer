from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO

Box = Tuple[int, int, int, int]  # x1,y1,x2,y2

@dataclass
class CropResult:
    frames: np.ndarray
    box: Box

class BaseCropper:
    def crop(self, frames: np.ndarray) -> CropResult:
        raise NotImplementedError

class FullFrameCropper(BaseCropper):
    def crop(self, frames: np.ndarray) -> CropResult:
        if frames.size == 0:
            return CropResult(frames=frames, box=(0,0,0,0))
        h, w = frames[0].shape[:2]
        return CropResult(frames=frames, box=(0,0,w,h))

@dataclass
class FixedCropper(BaseCropper):
    x1: int
    y1: int
    x2: int
    y2: int

    def crop(self, frames: np.ndarray) -> CropResult:
        if frames.size == 0:
            return CropResult(frames=frames, box=(self.x1,self.y1,self.x2,self.y2))
        # clamp to frame bounds
        h, w = frames[0].shape[:2]
        x1 = max(0, min(self.x1, w))
        x2 = max(0, min(self.x2, w))
        y1 = max(0, min(self.y1, h))
        y2 = max(0, min(self.y2, h))
        return CropResult(frames=frames[:, y1:y2, x1:x2], box=(x1,y1,x2,y2))

class YoloBestBoxCropper(BaseCropper):
    """YOLO crop using the single best-confidence box across all frames (not per-frame)."""
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def _best_box_single(self, frame: np.ndarray) -> tuple[Box, float]:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(img, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        h, w = frame.shape[:2]
        if len(confs) == 0:
            return (0, 0, w, h), 0.0
        idx = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes[idx]
        return (int(x1), int(y1), int(x2), int(y2)), float(confs[idx])

    def crop(self, frames: np.ndarray) -> CropResult:
        if frames.size == 0:
            return CropResult(frames=frames, box=(0,0,0,0))

        t = len(frames)
        all_boxes: List[Box] = []
        all_confs = np.empty(t, dtype=float)
        for i, frm in enumerate(frames):
            bx, conf = self._best_box_single(frm)
            all_boxes.append(bx)
            all_confs[i] = conf

        best_idx = int(np.nanargmax(all_confs))
        x1, y1, x2, y2 = all_boxes[best_idx]
        cropped = frames[:, y1:y2, x1:x2]
        return CropResult(frames=cropped, box=(x1,y1,x2,y2))
