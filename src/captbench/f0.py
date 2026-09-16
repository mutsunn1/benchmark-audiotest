"""Fundamental frequency tracking and five-level T-value tone modelling.

This is the part of the pipeline that Mandarin makes non-negotiable: every
syllable carries a lexical tone, so alongside the MFCC stream we run a dedicated
F0 tracker over the voiced (rhyme) portion of each syllable.

Cross-speaker normalisation follows Shi Feng's logarithmic five-level transform,

    T = 5 * (log10(F0) - log10(Fmin)) / (log10(Fmax) - log10(Fmin))

which projects absolute Hz onto Zhao Yuanren's 1-5 relative pitch scale and
removes the speaker's anatomical pitch range. `F0Range` carries the (Fmin, Fmax)
bounds; they must come from a calibration set of the same speaker, *not* from
the clip under test - normalising a single clip by its own range would force
every syllable to span the full scale and destroy the very distinction we are
trying to measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .dsp import autocorrelation_voicing, frame_signal
from .g2p import TONE_TEMPLATES

DEFAULT_FMIN = 70.0
DEFAULT_FMAX = 450.0

#: A pitch range narrower than this ratio is unusable as a normalisation basis.
MIN_RANGE_RATIO = 1.15


@dataclass
class F0Track:
    times: np.ndarray  # (T,) seconds
    f0: np.ndarray  # (T,) Hz, NaN where unvoiced
    voiced: np.ndarray  # (T,) bool
    backend: str = "parselmouth"

    @property
    def voiced_count(self) -> int:
        return int(self.voiced.sum())


@dataclass(frozen=True)
class F0Range:
    """Speaker pitch bounds used for T-value normalisation."""

    fmin: float
    fmax: float
    source: str = "calibration"

    def usable(self) -> bool:
        return self.fmin > 0 and self.fmax / self.fmin >= MIN_RANGE_RATIO

    @classmethod
    def from_values(cls, values: np.ndarray, low_pct: float = 5.0, high_pct: float = 95.0,
                    source: str = "clip") -> "F0Range":
        vals = values[np.isfinite(values) & (values > 0)]
        if vals.size == 0:
            return cls(DEFAULT_FMIN, DEFAULT_FMAX, source="default")
        fmin = float(np.percentile(vals, low_pct))
        fmax = float(np.percentile(vals, high_pct))
        if fmax / max(fmin, 1e-6) < MIN_RANGE_RATIO:
            # Degenerate (e.g. one monotone syllable): widen slightly so the
            # transform stays numerically sane, and record that we did.
            centre = float(np.median(vals))
            fmin, fmax = centre / 1.1, centre * 1.1
            source = f"{source}:widened"
        return cls(fmin, fmax, source=source)


def track_f0(
    signal: np.ndarray,
    sr: int,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    time_step: float = 0.01,
) -> F0Track:
    """Track F0 with Praat (parselmouth); fall back to a numpy YIN variant."""
    try:
        return _track_parselmouth(signal, sr, fmin, fmax, time_step)
    except Exception:
        return _track_yin(signal, sr, fmin, fmax, time_step)


def _track_parselmouth(
    signal: np.ndarray, sr: int, fmin: float, fmax: float, time_step: float
) -> F0Track:
    import parselmouth

    sound = parselmouth.Sound(np.asarray(signal, dtype=np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch_ac(
        time_step=time_step,
        pitch_floor=fmin,
        pitch_ceiling=fmax,
        very_accurate=True,
        silence_threshold=0.03,
        voicing_threshold=0.45,
    )
    freqs = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    times = np.asarray(pitch.xs(), dtype=np.float64)
    voiced = freqs > 0
    f0 = np.where(voiced, freqs, np.nan)
    return F0Track(times=times, f0=f0, voiced=voiced, backend="parselmouth")


def _track_yin(
    signal: np.ndarray, sr: int, fmin: float, fmax: float, time_step: float
) -> F0Track:
    """YIN-style tracker: cumulative mean normalised difference + parabolic peak."""
    hop = max(1, int(round(time_step * sr)))
    frames, times = frame_signal(signal, sr, frame_len=0.04, hop=hop / sr)
    periodicity, lag = autocorrelation_voicing(frames, sr, fmin=fmin, fmax=fmax)
    voiced = periodicity > 0.35
    with np.errstate(divide="ignore", invalid="ignore"):
        f0 = np.where(voiced & (lag > 0), sr / np.maximum(lag, 1), np.nan)
    return F0Track(times=times, f0=f0, voiced=voiced, backend="yin")


def to_t_values(f0: np.ndarray, f0_range: F0Range) -> np.ndarray:
    """Shi Feng five-level transform: Hz -> T values on a 0-5 scale."""
    f0 = np.asarray(f0, dtype=np.float64)
    fmin = max(f0_range.fmin, 1e-6)
    fmax = max(f0_range.fmax, fmin * MIN_RANGE_RATIO)
    lo, hi = np.log10(fmin), np.log10(fmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = 5.0 * (np.log10(np.where(f0 > 0, f0, np.nan)) - lo) / (hi - lo)
    return np.clip(t, -1.0, 6.0)


def extract_contour(
    track: F0Track,
    start: float,
    end: float,
    trim: float = 0.06,
    min_frames: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Voiced F0 contour inside [start, end], with edges trimmed.

    Returns (times, f0). Empty arrays when the window holds too little voicing -
    which is itself diagnostic (a fully devoiced syllable has no tone).
    """
    mask = (track.times >= start) & (track.times <= end) & track.voiced
    if mask.sum() < min_frames:
        return np.array([]), np.array([])
    times = track.times[mask]
    f0 = track.f0[mask]
    span = times[-1] - times[0]
    if span > 0 and trim > 0:
        keep = (times >= times[0] + trim * span) & (times <= times[-1] - trim * span)
        if keep.sum() >= min_frames:
            times, f0 = times[keep], f0[keep]
    return times, f0


def resample_contour(
    times: np.ndarray, values: np.ndarray, n_points: int = 20
) -> np.ndarray:
    """Time-normalise a contour to a fixed number of equally spaced points."""
    if times.size == 0 or values.size == 0:
        return np.array([])
    if times.size == 1:
        return np.full(n_points, float(values[0]))
    grid = np.linspace(times[0], times[-1], n_points)
    return np.interp(grid, times, values)


@lru_cache(maxsize=64)
def template_curve(tone: int, variant: int = 0, n_points: int = 20) -> tuple[float, ...]:
    """Canonical tone contour sampled at `n_points`, in T-value space."""
    variants = TONE_TEMPLATES.get(tone)
    if not variants:
        raise KeyError(f"no template for tone {tone}")
    control = variants[min(variant, len(variants) - 1)]
    xs = np.array([p[0] for p in control], dtype=np.float64)
    ys = np.array([p[1] for p in control], dtype=np.float64)
    grid = np.linspace(0.0, 1.0, n_points)
    return tuple(np.interp(grid, xs, ys).tolist())


def contour_features(t_values: np.ndarray) -> dict[str, float]:
    """Shape descriptors a tone diagnostician reasons over.

    These are what turn "the DTW cost was 3.2" into "the 3rd tone never dipped
    below T=2.4, so the learner substituted a rising tone".
    """
    t = np.asarray(t_values, dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size < 3:
        return {}
    n = t.size
    idx = np.arange(n, dtype=np.float64)
    slope = float(np.polyfit(idx, t, 1)[0])
    curvature = float(np.mean(np.abs(np.diff(t, n=2)))) if n >= 3 else 0.0
    dip_idx = int(np.argmin(t))
    peak_idx = int(np.argmax(t))
    return {
        "start": float(t[0]),
        "end": float(t[-1]),
        "mean": float(t.mean()),
        "min": float(t.min()),
        "max": float(t.max()),
        "range": float(t.max() - t.min()),
        "std": float(t.std()),
        "slope": slope,  # per-sample slope across the normalised contour
        "curvature": curvature,
        "dip_pos": float(dip_idx / (n - 1)),
        "dip_depth": float(t[0] - t.min()),
        "peak_pos": float(peak_idx / (n - 1)),
        "late_rise": float(t[-1] - t.min()),
        "late_fall": float(t[0] - t[-1]),
        "end_level": float(np.mean(t[int(0.85 * n) :])),
        "start_level": float(np.mean(t[: max(1, int(0.15 * n))])),
    }


def describe_tone_deviation(expected_tone: int, features: dict[str, float]) -> list[str]:
    """Turn contour features into the error taxonomy a tutor would use."""
    notes: list[str] = []
    if not features:
        return ["no measurable pitch contour (fully devoiced or unvoiced)"]
    mn, mx = features["min"], features["max"]
    end, start = features["end"], features["start"]

    if expected_tone == 1:  # 55: should stay high and flat
        if features["slope"] < -0.06:
            notes.append("尾音下滑 (pitch drifts down across a level tone)")
        if features["mean"] < 3.5:
            notes.append(f"调高不足 (level tone sits low, mean T={features['mean']:.1f})")
    elif expected_tone == 2:  # 35: should rise substantially
        if features["late_rise"] < 1.5:
            notes.append(
                f"升幅不足 (rise of only {features['late_rise']:.1f} T over the second half)"
            )
        if features["slope"] < 0.02:
            notes.append("阳平平化 (no positive slope)")
    elif expected_tone == 3:  # 214: needs a real dip
        if mn > 1.5:
            notes.append(f"三声未能压低 (dip floor only reaches T={mn:.1f})")
        if features["dip_pos"] > 0.6:
            notes.append("过早升调 (the dip arrives too late to count as 214)")
        if features["dip_depth"] < 0.8:
            notes.append("缺失低平段 (no low plateau)")
    elif expected_tone == 4:  # 51: should fall all the way
        if end > 1.8:
            notes.append(f"去声未降到底 (final level still T={end:.1f})")
        if features["late_fall"] < 2.5:
            notes.append("降幅截断 (fall shallower than a full 51)")

    if mx - mn < 1.2 and expected_tone in (2, 3, 4):
        notes.append("调型趋平 (contour is nearly flat where movement is required)")
    if start > 4.5 and expected_tone == 2:
        notes.append("起调过高 (starts so high there is no room to rise)")
    return notes
