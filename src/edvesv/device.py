from __future__ import annotations
import torch

def pick_device(device: str | None = None) -> str:
    """Return 'cuda' if available unless user forces 'cpu' or 'cuda'."""
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'auto', 'cpu', or 'cuda', got: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        # don't hard-fail; make it explicit
        return "cpu"
    return device
