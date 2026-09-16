"""Template-based forced alignment.

There is no HMM/GMM acoustic model in this pipeline. Instead it uses the
classic pre-DNN technique: obtain a clean reference reading of the expected
text, DTW-align the test utterance's MFCC sequence against the reference's, and
carry the reference's syllable boundaries across the warping path onto the test
timeline. The product is the same one Viterbi forced alignment produces -
[ts, te] per syllable - paid for with a reference recording instead of a phone
loop search graph.

For single-syllable items (most of this benchmark) alignment is trivial: the
syllable is the voiced span. Alignment earns its keep on the sentence-length
trap items, where the target word sits at a known index among other syllables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp import FrameFeatures, cmn
from .dtw import dtw


@dataclass
class Segment:
    start: float
    end: float
    label: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def slice(self, times: np.ndarray) -> np.ndarray:
        return (times >= self.start) & (times <= self.end)


def voiced_span(feats: FrameFeatures, threshold_db: float = -38.0) -> tuple[float, float]:
    """First and last frame times whose energy clears the silence threshold."""
    db = 20 * np.log10(feats.rms + 1e-12)
    active = np.flatnonzero(db > threshold_db)
    if active.size == 0:
        return float(feats.times[0]), float(feats.times[-1])
    return float(feats.times[active[0]]), float(feats.times[active[-1]])


def _smooth(values: np.ndarray, width: int = 3) -> np.ndarray:
    if width <= 1 or values.size < width:
        return values
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def find_nuclei(
    feats: FrameFeatures,
    n_syllables: int,
    min_gap: float = 0.07,
    threshold_db: float = -38.0,
) -> list[int]:
    """Pick the `n_syllables` strongest energy peaks, well separated in time.

    Mandarin syllables are energy-prominent nuclei separated by valleys, so
    picking peaks is a serviceable stand-in for a syllable detector on clean
    speech.
    """
    rms = _smooth(feats.rms.astype(np.float64), 3)
    db = 20 * np.log10(rms + 1e-12)
    min_frames = max(1, int(round(min_gap / feats.hop)))

    candidates: list[tuple[float, int]] = []
    for i in range(1, len(rms) - 1):
        if db[i] <= threshold_db:
            continue
        if rms[i] >= rms[i - 1] and rms[i] >= rms[i + 1]:
            candidates.append((float(rms[i]), i))
    candidates.sort(reverse=True)

    picked: list[int] = []
    for _, idx in candidates:
        if all(abs(idx - p) >= min_frames for p in picked):
            picked.append(idx)
        if len(picked) == n_syllables:
            break

    if len(picked) < n_syllables:
        # Not enough distinct peaks: fall back to evenly spaced nuclei across
        # the voiced span so every expected syllable still gets a window.
        start_t, end_t = voiced_span(feats, threshold_db)
        grid = np.linspace(start_t, end_t, n_syllables + 2)[1:-1]
        return [int(np.argmin(np.abs(feats.times - t))) for t in grid]
    return sorted(picked)


def segment_syllables(
    feats: FrameFeatures,
    n_syllables: int,
    labels: list[str] | None = None,
    threshold_db: float = -38.0,
) -> list[Segment]:
    """Split an utterance into `n_syllables` segments by nuclei and valleys."""
    if n_syllables <= 1:
        start, end = voiced_span(feats, threshold_db)
        return [Segment(start, end, labels[0] if labels else "")]

    nuclei = find_nuclei(feats, n_syllables, threshold_db=threshold_db)
    start_t, end_t = voiced_span(feats, threshold_db)

    bounds = [start_t]
    for left, right in zip(nuclei[:-1], nuclei[1:]):
        window = feats.rms[left:right]
        valley = left + int(np.argmin(window)) if window.size else (left + right) // 2
        bounds.append(float(feats.times[valley]))
    bounds.append(end_t)

    segments = []
    for i in range(n_syllables):
        label = labels[i] if labels and i < len(labels) else ""
        segments.append(Segment(bounds[i], bounds[i + 1], label))
    return segments


def align_frames(test_feats: FrameFeatures, ref_feats: FrameFeatures) -> np.ndarray:
    """DTW between two MFCC sequences; returns a ref-frame -> test-frame map."""
    test = cmn(test_feats.mfcc[:, :26].astype(np.float64))
    ref = cmn(ref_feats.mfcc[:, :26].astype(np.float64))
    result = dtw(test, ref, metric="euclidean")

    mapping = np.full(ref.shape[0], -1, dtype=np.int64)
    for i, j in result.path:
        if mapping[j] < 0:
            mapping[j] = i
    missing = np.flatnonzero(mapping < 0)
    if missing.size:
        known = np.flatnonzero(mapping >= 0)
        if known.size == 0:
            mapping[:] = np.arange(ref.shape[0])
        else:
            mapping[missing] = np.interp(missing, known, mapping[known]).astype(np.int64)
    return np.clip(mapping, 0, test_feats.times.size - 1)


def align_segments(
    ref_segments: list[Segment],
    mapping: np.ndarray,
    ref_times: np.ndarray,
    test_times: np.ndarray,
) -> list[Segment]:
    """Carry reference-time segments onto the test timeline via the DTW map.

    `seg.start` / `seg.end` are reference-clock times, so they are converted to
    reference frame indices first; `mapping` then sends those to test frames.
    """
    mapped: list[Segment] = []
    for seg in ref_segments:
        lo = int(np.searchsorted(ref_times, seg.start))
        hi = int(np.searchsorted(ref_times, seg.end))
        lo = min(max(lo, 0), mapping.size - 1)
        hi = min(max(hi, lo), mapping.size - 1)
        test_lo = int(mapping[lo])
        test_hi = int(mapping[hi])
        if test_hi < test_lo:
            test_lo, test_hi = test_hi, test_lo
        mapped.append(
            Segment(
                float(test_times[test_lo]),
                float(test_times[test_hi]),
                seg.label,
            )
        )
    return mapped


def align_utterance(
    test_feats: FrameFeatures,
    ref_feats: FrameFeatures,
    n_syllables: int,
    labels: list[str] | None = None,
) -> tuple[list[Segment], list[Segment], float]:
    """Full alignment: segment the reference, then map onto the test.

    Returns (test-time segments, reference-time segments, normalised DTW cost).
    The reference segments are returned because callers need per-syllable
    reference templates, not just the test-side boundaries.
    """
    if n_syllables <= 1:
        start, end = voiced_span(test_feats)
        ref_start, ref_end = voiced_span(ref_feats)
        label = labels[0] if labels else ""
        return (
            [Segment(start, end, label)],
            [Segment(ref_start, ref_end, label)],
            float("nan"),
        )

    ref_segments = segment_syllables(ref_feats, n_syllables, labels)
    mapping = align_frames(test_feats, ref_feats)
    segments = align_segments(ref_segments, mapping, ref_feats.times, test_feats.times)

    test = cmn(test_feats.mfcc[:, :26].astype(np.float64))
    ref = cmn(ref_feats.mfcc[:, :26].astype(np.float64))
    cost = dtw(test, ref, metric="euclidean").normalized
    return segments, ref_segments, float(cost)
