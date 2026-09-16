"""Dynamic time warping with a Sakoe-Chiba band.

Used for two distinct jobs in this codebase:

1. Aligning a test utterance against a clean TTS reference of the expected
   text, so that syllable boundaries known in the reference can be mapped onto
   the test timeline (template-based forced alignment).
2. Matching a syllable's normalised F0 contour against the canonical tone
   templates (55 / 35 / 214 / 51) in five-level T-value space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DTWResult:
    distance: float  # total accumulated cost
    normalized: float  # distance / path length - comparable across lengths
    path: np.ndarray  # (P, 2) array of (i, j) index pairs


def dtw(
    a: np.ndarray,
    b: np.ndarray,
    band: int | None = None,
    metric: str = "euclidean",
    step_penalty: float = 0.0,
) -> DTWResult:
    """Classic DTW between sequences `a` (T1, D) and `b` (T2, D).

    `band` limits the maximum |i - j| deviation (Sakoe-Chiba). Defaults to a
    band wide enough to allow natural rate differences but tight enough to stop
    degenerate alignments.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    if a.shape[0] == 0 or b.shape[0] == 0:
        return DTWResult(float("inf"), float("inf"), np.zeros((0, 2), dtype=int))
    n, m = a.shape[0], b.shape[0]

    if band is None:
        band = max(20, int(0.35 * max(n, m)) + 1)
    band = max(band, abs(n - m) + 1)

    cost = _pairwise(a, b, metric)
    inf = float("inf")
    acc = np.full((n + 1, m + 1), inf)
    acc[0, 0] = 0.0
    back = np.full((n, m), -1, dtype=np.int8)

    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        if lo > hi:
            continue
        for j in range(lo, hi + 1):
            diag = acc[i - 1, j - 1]
            up = acc[i - 1, j]
            left = acc[i, j - 1]
            best = diag
            move = 0
            if up < best:
                best, move = up, 1
            if left < best:
                best, move = left, 2
            acc[i, j] = cost[i - 1, j - 1] + best + step_penalty
            back[i - 1, j - 1] = move

    # Backtrace from (n, m) to (1, 1).
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        move = back[i - 1, j - 1]
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        elif move == 2:
            j -= 1
        else:
            break
    path.reverse()
    path_arr = np.array(path, dtype=int) if path else np.zeros((0, 2), dtype=int)
    total = float(acc[n, m])
    length = max(1, len(path))
    return DTWResult(distance=total, normalized=total / length, path=path_arr)


def _pairwise(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        diff = a[:, None, :] - b[None, :, :]
        # max(...,0) rather than +epsilon: identical rows must cost exactly
        # zero, or a perfect match accumulates spurious distance along the path.
        return np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
    if metric == "cosine":
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return 1.0 - an @ bn.T
    if metric == "cityblock":
        return np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)
    raise ValueError(f"unsupported metric: {metric}")
