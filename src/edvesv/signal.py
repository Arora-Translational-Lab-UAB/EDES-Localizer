from __future__ import annotations
import numpy as np
import torch
from scipy.stats import zscore
from scipy.signal import find_peaks

def compute_fft_dominance_torch(signal_torch: torch.Tensor) -> torch.Tensor:
    signal_torch = signal_torch - torch.mean(signal_torch)
    fft_vals = torch.fft.fft(signal_torch)
    fft_vals = fft_vals[: len(fft_vals) // 2]
    power = torch.abs(fft_vals) ** 2
    total_energy = torch.sum(power)
    if total_energy == 0:
        return torch.tensor(0.0, device=signal_torch.device)
    dominant_energy = torch.max(power[1:]) if len(power) > 1 else torch.tensor(0.0, device=signal_torch.device)
    return dominant_energy / total_energy

def zero_crossing_variance(signal: np.ndarray) -> float:
    try:
        signal = signal - np.mean(signal)
        N = len(signal)
        fft_vals = np.fft.fft(signal)
        fft_vals = fft_vals[: N // 2]
        dominant_idx = np.argmax(np.abs(fft_vals[1:])) + 1
        if dominant_idx <= 0:
            return np.nan
        period_frames = N // dominant_idx
        if period_frames < 2:
            return np.nan

        n_cycles = N // period_frames
        zc_counts = []
        for i in range(n_cycles):
            cycle = signal[i * period_frames : (i + 1) * period_frames]
            cycle = cycle - np.mean(cycle)
            zc = np.sum(np.diff(np.sign(cycle)) != 0)
            zc_counts.append(zc)
        if len(zc_counts) < 2:
            return np.nan
        return float(np.var(zc_counts))
    except Exception:
        return np.nan

def detect_peaks_and_troughs(signal: np.ndarray, prominence: float = 0.6):
    # same padding trick as notebook
    padded_peak = np.concatenate(([-np.inf], signal, [-np.inf]))
    padded_trough = np.concatenate(([np.inf], signal, [np.inf]))
    peaks_raw, _ = find_peaks(padded_peak, prominence=prominence)
    troughs_raw, _ = find_peaks(-padded_trough, prominence=prominence)

    peaks = peaks_raw - 1
    troughs = troughs_raw - 1
    valid_peaks = [int(i) for i in peaks if 0 <= i < len(signal)]
    valid_troughs = [int(i) for i in troughs if 0 <= i < len(signal)]
    return valid_peaks, valid_troughs

def select_best_pc(pc_raw: list[np.ndarray], device: str) -> tuple[int, list[np.ndarray], np.ndarray]:
    """Select best PC using ZC variance with FFT tie-break and FFT fallback."""
    pc_std = [zscore(pc) for pc in pc_raw]

    fft_scores = [
        compute_fft_dominance_torch(torch.tensor(pc_std[i], device=device, dtype=torch.float32)).cpu().item()
        for i in range(3)
    ]
    fft_scores = np.array(fft_scores)

    zc_scores = [zero_crossing_variance(pc_std[i]) for i in range(3)]
    zc_array = np.array(zc_scores, dtype=float)
    valid = ~np.isnan(zc_array)

    if np.any(valid):
        valid_zc = zc_array[valid]
        valid_modes = np.arange(3)[valid]

        # if unique, just argmin
        if len(np.unique(valid_zc)) == len(valid_zc):
            best = int(valid_modes[np.argmin(valid_zc)])
        else:
            min_zc = np.min(valid_zc)
            min_idx = np.where(valid_zc == min_zc)[0]
            if len(min_idx) == 1:
                best = int(valid_modes[min_idx[0]])
            else:
                tie_modes = valid_modes[min_idx]
                tie_fft = fft_scores[tie_modes]
                best = int(tie_modes[np.argmax(tie_fft)])
    else:
        best = int(np.argmax(fft_scores))

    return best, pc_std, pc_std[best]
