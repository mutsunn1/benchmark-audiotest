"""Audio I/O and format normalisation.

Every clip in the corpus - whatever its source (Qwen TTS mp3/wav, macOS `say`
AIFF, a real microphone recording) - is normalised to 16 kHz mono float32 WAV
before it reaches the scorers. That is the sample rate the classic CAPT
pipeline assumes (pre-emphasis, 25 ms frames, 0-8 kHz analysis band).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def load_audio(path: str | Path, target_sr: int | None = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32, optionally resampled."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        mono = resample(mono, sr, target_sr)
        sr = target_sr
    return np.ascontiguousarray(mono, dtype=np.float32), sr


def save_audio(path: str | Path, signal: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(signal, dtype=np.float32), sr, subtype="PCM_16")
    return path


def resample(signal: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return signal
    from math import gcd

    g = gcd(sr, target_sr)
    return resample_poly(signal, target_sr // g, sr // g).astype(np.float32)


def normalize_peak(signal: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(signal))) if signal.size else 0.0
    return signal if m < 1e-9 else (signal * (peak / m)).astype(np.float32)


def trim_silence(
    signal: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    pad: float = 0.05,
) -> np.ndarray:
    """Trim leading/trailing silence, keeping `pad` seconds of margin."""
    if signal.size == 0:
        return signal
    frame = max(1, int(0.01 * sr))
    n = signal.size // frame
    if n == 0:
        return signal
    rms = np.sqrt(
        np.mean(np.square(signal[: n * frame].reshape(n, frame)), axis=1) + 1e-12
    )
    db = 20 * np.log10(rms + 1e-12)
    voiced = np.flatnonzero(db > threshold_db)
    if voiced.size == 0:
        return signal
    start = max(0, int(voiced[0] * frame - pad * sr))
    end = min(signal.size, int((voiced[-1] + 1) * frame + pad * sr))
    return signal[start:end]


def duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)
