from __future__ import annotations
import numpy as np

def closest_candidate(candidates: list[int], gt: int) -> float:
    return min(candidates, key=lambda x: abs(x - gt)) if candidates else np.nan

def pick_ed_es_distinct(candidates: list[int], gt_ed: int, gt_es: int) -> tuple[float, float]:
    """Notebook logic: pick closest ED and ES but prefer ED != ES."""
    if not candidates:
        return np.nan, np.nan

    ed_diffs = {c: abs(c - gt_ed) for c in candidates}
    es_diffs = {c: abs(c - gt_es) for c in candidates}

    sorted_ed = sorted(ed_diffs.items(), key=lambda x: x[1])
    sorted_es = sorted(es_diffs.items(), key=lambda x: x[1])

    pred_edv, pred_esv = None, None
    for edv_candidate, _ in sorted_ed:
        for esv_candidate, _ in sorted_es:
            if edv_candidate != esv_candidate:
                pred_edv = edv_candidate
                pred_esv = esv_candidate
                break
        if pred_edv is not None and pred_esv is not None:
            break

    if pred_edv is None or pred_esv is None:
        pred_edv = sorted_ed[0][0]
        pred_esv = sorted_es[0][0]
    return float(pred_edv), float(pred_esv)
