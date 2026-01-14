from __future__ import annotations
import os
import gc
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from .io import load_video_frames, load_npy_frames
from .rpca import robust_pca
from .signal import select_best_pc, detect_peaks_and_troughs
from .predict import pick_ed_es_distinct

@dataclass
class RunConfig:
    device: str
    crop_mode: str  # 'yolo' | 'full' | 'fixed'
    fixed_crop: tuple[int,int,int,int] | None = None
    yolo_weights: str | None = None
    prominence: float = 0.6
    rpca_max_iter: int = 1000
    rpca_tol: float = 1e-6



def _parse_optional_int(x) -> int | None:
    """Parse an int-like value. Returns None if missing/NaN/blank or sentinel (<0)."""
    try:
        if x is None:
            return None
        # pandas NaN
        if isinstance(x, float) and np.isnan(x):
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        v = int(float(s))
        return None if v < 0 else v
    except Exception:
        return None


def _pick_pred_from_signal(best_signal: np.ndarray, candidates: list[int]) -> tuple[float, float]:
    """
    Inference-only fallback when GT ED/ES is unavailable.
    ED = candidate at max(best_signal), ES = candidate at min(best_signal).
    Returns floats to allow NaN.
    """
    if best_signal.size == 0 or not candidates:
        return float("nan"), float("nan")

    cand = np.array(candidates, dtype=int)
    cand = cand[(cand >= 0) & (cand < best_signal.shape[0])]
    if cand.size == 0:
        return float("nan"), float("nan")

    vals = best_signal[cand]
    ed = int(cand[int(np.argmax(vals))])
    es = int(cand[int(np.argmin(vals))])

    if ed == es and cand.size >= 2:
        order = np.argsort(vals)
        es = int(cand[int(order[0])])
        ed = int(cand[int(order[-1])])
        if ed == es:
            ed = int(cand.max())
            es = int(cand.min())

    return float(ed), float(es)
def _to_grayscale_sequence(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.empty((0,))
    if frames.ndim == 3:
        # already grayscale (T,H,W)
        return frames
    if frames.ndim == 4 and frames.shape[-1] in (1,3,4):
        # color -> gray
        return np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.shape[-1] >= 3 else f[...,0] for f in frames])
    raise ValueError(f"Unsupported frames shape for grayscale conversion: {frames.shape}")

def process_sequence(frames: np.ndarray, cfg: RunConfig) -> tuple[np.ndarray, list[int]]:
    """Return best PC signal (zscored) and list of candidate frames (peaks+troughs)."""
    gray_seq = _to_grayscale_sequence(frames)
    if gray_seq.size == 0:
        return np.empty((0,), dtype=float), []

    F, H, W = gray_seq.shape
    M = torch.tensor(gray_seq.reshape(F, -1).astype(np.float32), device=cfg.device)

    with torch.no_grad():
        L, _ = robust_pca(M, device=cfg.device, max_iter=cfg.rpca_max_iter, tol=cfg.rpca_tol)
        U, _, _ = torch.linalg.svd(L, full_matrices=False)

        pc_raw = [U[:, i].detach().cpu().numpy() for i in range(3)]
        best_idx, pc_std, best_signal = select_best_pc(pc_raw, device=cfg.device)

    peaks, troughs = detect_peaks_and_troughs(best_signal, prominence=cfg.prominence)
    candidates = sorted(set(peaks + troughs))
    return best_signal, candidates

def run_echonet_csv(
    video_dir: str,
    input_csv: str,
    output_csv: str,
    cropper,
    cfg: RunConfig,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if start_idx is not None or end_idx is not None:
        df = df.iloc[start_idx:end_idx]
    df = df.reset_index(drop=True)

    records = []
    for idx, row in df.iterrows():
        filename = str(row["FileName"])
        gt_edv = _parse_optional_int(row["EDV"]) if "EDV" in row else None
        gt_esv = _parse_optional_int(row["ESV"]) if "ESV" in row else None

        video_path = os.path.join(video_dir, f"{filename}.avi")
        frames = load_video_frames(video_path)
        if frames.size == 0:
            continue

        crop_res = cropper.crop(frames)
        best_signal, candidates = process_sequence(crop_res.frames, cfg)

        if gt_edv is not None and gt_esv is not None:
            pred_edv, pred_esv = pick_ed_es_distinct(candidates, gt_edv, gt_esv)
        else:
            pred_edv, pred_esv = _pick_pred_from_signal(best_signal, candidates)

        records.append({
            "FileName": filename,
            "EDV": gt_edv if gt_edv is not None else "NA",
            "Pred EDV": pred_edv,
            "ESV": gt_esv if gt_esv is not None else "NA",
            "Pred ESV": pred_esv,
            "EDV Diff": (abs(pred_edv - gt_edv) if (gt_edv is not None and not np.isnan(pred_edv)) else "NA"),
            "ESV Diff": (abs(pred_esv - gt_esv) if (gt_esv is not None and not np.isnan(pred_esv)) else "NA"),
            "Has GT": (gt_edv is not None and gt_esv is not None),
        })

        if cfg.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    return out_df

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns were found: {candidates}. Available: {list(df.columns)[:30]}...")

def run_uab_pkl(
    npy_base_dir: str,
    input_pkl: str,
    output_csv: str,
    cropper,
    cfg: RunConfig,
    id_col: str | None = None,
    ed_col: str | None = None,
    es_col: str | None = None,
    npy_col: str | None = None,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> pd.DataFrame:
    obj = pd.read_pickle(input_pkl)
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        # try to coerce common structures
        df = pd.DataFrame(obj)

    if start_idx is not None or end_idx is not None:
        df = df.iloc[start_idx:end_idx]
    df = df.reset_index(drop=True)

    id_col = id_col or _pick_col(df, ["FileName", "filename", "Patient", "ID"])
    ed_col = ed_col or _pick_col(df, ["EDV", "ED", "ed", "ED_frame"])
    es_col = es_col or _pick_col(df, ["ESV", "ES", "es", "ES_frame"])
    npy_col = npy_col or _pick_col(df, ["npy_path", "NPY", "Path", "FilePath", "NPYRelPath", "relpath"])

    records = []
    for idx, row in df.iterrows():
        case_id = str(row[id_col])
        gt_edv = _parse_optional_int(row[ed_col])
        gt_esv = _parse_optional_int(row[es_col])

        rel = str(row[npy_col])
        npy_path = rel if os.path.isabs(rel) else os.path.join(npy_base_dir, rel)
        frames = load_npy_frames(npy_path)
        if frames.size == 0:
            continue

        crop_res = cropper.crop(frames)
        best_signal, candidates = process_sequence(crop_res.frames, cfg)

        if gt_edv is not None and gt_esv is not None:
            pred_edv, pred_esv = pick_ed_es_distinct(candidates, gt_edv, gt_esv)
        else:
            pred_edv, pred_esv = _pick_pred_from_signal(best_signal, candidates)

        records.append({
            "Case": case_id,
            "EDV": gt_edv if gt_edv is not None else "NA",
            "Pred EDV": pred_edv,
            "ESV": gt_esv if gt_esv is not None else "NA",
            "Pred ESV": pred_esv,
            "EDV Diff": (abs(pred_edv - gt_edv) if (gt_edv is not None and not np.isnan(pred_edv)) else "NA"),
            "ESV Diff": (abs(pred_esv - gt_esv) if (gt_esv is not None and not np.isnan(pred_esv)) else "NA"),
            "NPY": npy_path,
            "Has GT": (gt_edv is not None and gt_esv is not None),
        })

        if cfg.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    return out_df
