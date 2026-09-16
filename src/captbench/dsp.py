"""Frame-level signal processing: the front end of a classic CAPT pipeline.

Implements, in numpy/scipy only, the feature stages the reference architecture
describes: pre-emphasis H(z) = 1 - 0.97 z^-1, 25 ms Hamming-windowed frames
with a 10 ms hop, a 26-band mel filterbank, and 13 MFCCs plus their first and
second differences (39 dimensions). Also the auxiliary frame features the
segmental checks need: RMS envelope, zero-crossing rate, spectral centroid,
band energy ratios and a per-frame periodicity measure for voicing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, rfft, rfftfreq
from scipy.signal import get_window

PREEMPHASIS = 0.97
FRAME_LEN = 0.025
FRAME_HOP = 0.010
N_MELS = 26
N_MFCC = 13


@dataclass
class FrameFeatures:
    """Frame-level features on a common time grid."""

    times: np.ndarray  # (T,) frame centre times in seconds
    rms: np.ndarray  # (T,) root-mean-square energy
    zcr: np.ndarray  # (T,) zero-crossing rate
    centroid: np.ndarray  # (T,) spectral centroid (Hz)
    mfcc: np.ndarray  # (T, 39) MFCC + delta + delta-delta
    power: np.ndarray  # (T, F) power spectrum magnitudes
    sr: int = 16000
    n_fft: int = 512  # FFT size backing `power`

    @property
    def hop(self) -> float:
        return float(self.times[1] - self.times[0]) if self.times.size > 1 else FRAME_HOP

    @property
    def freqs(self) -> np.ndarray:
        """Frequency axis (Hz) for the columns of `power`."""
        return rfftfreq(self.n_fft, d=1.0 / self.sr)


def preemphasis(signal: np.ndarray, coeff: float = PREEMPHASIS) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return signal.astype(np.float32)
    out = np.append(signal[0], signal[1:] - coeff * signal[:-1])
    return out.astype(np.float32)


def frame_signal(
    signal: np.ndarray,
    sr: int,
    frame_len: float = FRAME_LEN,
    hop: float = FRAME_HOP,
) -> tuple[np.ndarray, np.ndarray]:
    """Split into overlapping Hamming-windowed frames.

    Returns (frames (T, N), times (T,)).
    """
    n_frame = int(round(frame_len * sr))
    n_hop = int(round(hop * sr))
    if signal.size < n_frame:
        padded = np.zeros(n_frame, dtype=np.float32)
        padded[: signal.size] = signal
        signal = padded
    n_frames = 1 + (signal.size - n_frame) // n_hop
    idx = np.arange(n_frame)[None, :] + n_hop * np.arange(n_frames)[:, None]
    frames = signal[idx] * get_window("hamming", n_frame)[None, :]
    times = (np.arange(n_frames) * n_hop + n_frame / 2.0) / sr
    return frames.astype(np.float32), times


def mel_filterbank(sr: int, n_fft: int, n_mels: int = N_MELS) -> np.ndarray:
    """HTK-style triangular mel filterbank, (n_mels, n_fft//2+1)."""
    to_mel = lambda f: 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)
    from_mel = lambda m: 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

    points = from_mel(np.linspace(to_mel(0.0), to_mel(sr / 2.0), n_mels + 2))
    bins = np.floor((n_fft + 1) * points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_mels):
        left, centre, right = bins[i], bins[i + 1], bins[i + 2]
        if centre == left:
            centre = min(left + 1, n_fft // 2)
        if right == centre:
            right = min(centre + 1, n_fft // 2)
        for k in range(left, centre):
            fb[i, k] = (k - left) / max(1, centre - left)
        for k in range(centre, right):
            fb[i, k] = (right - k) / max(1, right - centre)
    return fb


def _deltas(feats: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression deltas along time (padded at the edges)."""
    t = feats.shape[0]
    if t < 2 * width + 1:
        return np.zeros_like(feats)
    padded = np.pad(feats, ((width, width), (0, 0)), mode="edge")
    denom = 2 * sum(i * i for i in range(1, width + 1))
    out = np.zeros_like(feats)
    for n in range(t):
        acc = np.zeros(feats.shape[1], dtype=np.float64)
        for i in range(1, width + 1):
            acc += i * (padded[n + width + i] - padded[n + width - i])
        out[n] = acc / denom
    return out


def extract_frames(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int = N_MFCC,
    n_mels: int = N_MELS,
    with_deltas: bool = True,
) -> FrameFeatures:
    """Full front-end: pre-emphasis -> framing -> mel -> MFCC (+ deltas)."""
    signal = np.asarray(signal, dtype=np.float32)
    pre = preemphasis(signal)
    frames, times = frame_signal(pre, sr)
    n_fft = int(2 ** np.ceil(np.log2(max(frames.shape[1], 2))))

    windowed = np.zeros((frames.shape[0], n_fft), dtype=np.float64)
    windowed[:, : frames.shape[1]] = frames
    spectrum = rfft(windowed, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float64)

    fb = mel_filterbank(sr, n_fft, n_mels)
    mel_energy = power @ fb.T
    log_mel = np.log(mel_energy + 1e-10)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]

    if with_deltas:
        d1 = _deltas(mfcc)
        d2 = _deltas(d1)
        mfcc_full = np.concatenate([mfcc, d1, d2], axis=1)
    else:
        mfcc_full = mfcc

    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1).astype(np.float64)

    freqs = rfftfreq(n_fft, d=1.0 / sr)
    mag = np.abs(spectrum)
    centroid = (mag @ freqs) / (mag.sum(axis=1) + 1e-12)

    return FrameFeatures(
        times=times,
        rms=rms,
        zcr=zcr,
        centroid=centroid,
        mfcc=mfcc_full.astype(np.float32),
        power=power[:, : n_fft // 2 + 1].astype(np.float32),
        sr=sr,
        n_fft=n_fft,
    )


def cmn(feats: np.ndarray) -> np.ndarray:
    """Cepstral mean normalisation along time.

    Removes channel and voice colour so DTW compares shape rather than recording
    conditions.

    Mean only - deliberately not variance normalisation. Dividing by the
    standard deviation is fine for a whole utterance, but on a short segment
    (a single syllable is 5-15 frames) it rescales whatever noise is present to
    unit variance, which makes two different syllables z-score to the same
    shape and collapses the distance between them to zero.
    """
    mean = feats.mean(axis=0, keepdims=True)
    return (feats - mean).astype(np.float32)


def band_energy_ratio(
    power_row: np.ndarray, sr: int, n_fft: int, low: float, high: float
) -> float:
    """Fraction of energy in [low, high) Hz for a single frame's power spectrum."""
    freqs = rfftfreq(n_fft, d=1.0 / sr)
    mask = (freqs >= low) & (freqs < high)
    total = power_row.sum() + 1e-12
    return float(power_row[mask].sum() / total)


def spectral_centroid_band(
    power_row: np.ndarray, sr: int, n_fft: int, low: float, high: float
) -> float:
    """Spectral centroid restricted to a band - the s/sh discriminator."""
    freqs = rfftfreq(n_fft, d=1.0 / sr)
    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        return float("nan")
    p = power_row[mask]
    f = freqs[mask]
    return float((p @ f) / (p.sum() + 1e-12))


def autocorrelation_voicing(
    frames: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 450.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame periodicity (normalised autocorrelation peak) and its lag.

    Used for voice-onset-time measurement: a frame counts as voiced once the
    periodicity in the plausible pitch range crosses a threshold.
    """
    n = frames.shape[1]
    min_lag = max(2, int(sr / fmax))
    max_lag = min(n - 1, int(sr / fmin))
    periodicity = np.zeros(frames.shape[0], dtype=np.float64)
    lag = np.zeros(frames.shape[0], dtype=np.int64)
    if max_lag <= min_lag:
        return periodicity, lag
    windowed = frames.astype(np.float64)
    for i in range(windowed.shape[0]):
        x = windowed[i] - windowed[i].mean()
        denom = float(x @ x) + 1e-12
        ac = np.correlate(x, x, mode="full")[n - 1 :]
        seg = ac[min_lag:max_lag]
        if seg.size == 0:
            continue
        k = int(np.argmax(seg))
        periodicity[i] = seg[k] / denom
        lag[i] = min_lag + k
    return periodicity, lag
