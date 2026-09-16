"""Per-syllable acoustic measurements: the evidence a diagnosis is built on.

Each function here implements one of the physical correlates the reference
architecture names, and returns numbers rather than verdicts, so the decision
rules in `traditional.py` stay auditable:

* aspiration  -> voice onset time (VOT), the burst-to-voicing delay
* 平翘舌      -> spectral centroid of the fricative noise
* 圆唇        -> F3 of the vowel (ü pulls F3 down ~1000 Hz from i)
* 前后鼻音    -> F2/F3 convergence (velar pinch) in the coda
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import Segment
from .dsp import FrameFeatures, autocorrelation_voicing, extract_frames, preemphasis, frame_signal
from .f0 import F0Track, track_f0


@dataclass
class Utterance:
    """Everything the scorers need about one clip, computed once."""

    signal: np.ndarray
    sr: int
    feats: FrameFeatures
    f0: F0Track
    periodicity: np.ndarray  # (T,) normalised autocorrelation peak per frame

    @property
    def duration(self) -> float:
        return float(self.signal.size) / self.sr


def prepare(signal: np.ndarray, sr: int) -> Utterance:
    feats = extract_frames(signal, sr)
    f0 = track_f0(signal, sr)

    # Recompute the frames for the periodicity measure. `frame_signal` is
    # deterministic, so this reproduces exactly the grid `extract_frames` used;
    # assert it rather than trust it.
    frames, _ = frame_signal(preemphasis(signal), sr)
    periodicity, _ = autocorrelation_voicing(frames, sr)
    if periodicity.size != feats.times.size:
        periodicity = np.resize(periodicity, feats.times.size)

    return Utterance(signal=np.asarray(signal, dtype=np.float32), sr=sr,
                     feats=feats, f0=f0, periodicity=periodicity)


@dataclass
class OnsetMeasurement:
    """Timing of the consonant release and the start of phonation."""

    onset: float  # first frame with real energy in the window
    burst: float  # loudest frame (the release burst / fricative peak)
    voicing_onset: float  # first reliably periodic frame
    vot: float  # voicing_onset - onset, in seconds
    aperiodic: float  # duration of the noise-only stretch

    def as_dict(self) -> dict[str, float]:
        return {
            "onset": self.onset,
            "burst": self.burst,
            "voicing_onset": self.voicing_onset,
            "vot": self.vot,
            "aperiodic": self.aperiodic,
        }


def measure_onset(
    utt: Utterance,
    seg: Segment,
    energy_floor_ratio: float = 0.10,
    periodicity_threshold: float = 0.45,
) -> OnsetMeasurement:
    """Measure VOT / frication duration inside a syllable window."""
    times = utt.feats.times
    mask = seg.slice(times)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return OnsetMeasurement(seg.start, seg.start, seg.start, 0.0, 0.0)

    rms = utt.feats.rms[idx]
    peak = float(rms.max()) + 1e-12
    floor = energy_floor_ratio * peak
    active = idx[rms > floor]
    if active.size == 0:
        return OnsetMeasurement(seg.start, seg.start, seg.start, 0.0, 0.0)

    onset_idx = int(active[0])
    burst_idx = int(idx[int(np.argmax(rms))])

    periodicity = utt.periodicity
    voiced_idx = None
    run = 0
    for i in active:
        if periodicity[i] > periodicity_threshold and utt.feats.rms[i] > floor:
            run += 1
            if run >= 2:
                voiced_idx = int(i) - 1
                break
        else:
            run = 0
    if voiced_idx is None:
        voiced_idx = int(active[-1])

    onset_t = float(times[onset_idx])
    voicing_t = float(times[voiced_idx])
    return OnsetMeasurement(
        onset=onset_t,
        burst=float(times[burst_idx]),
        voicing_onset=voicing_t,
        vot=max(0.0, voicing_t - onset_t),
        aperiodic=max(0.0, voicing_t - onset_t),
    )


def fricative_profile(
    utt: Utterance,
    start: float,
    end: float,
    min_frames: int = 2,
) -> dict[str, float]:
    """Spectral shape of the aperiodic (fricative/aspiration) stretch.

    Averages the power spectra over [start, end] and reports the centroid plus
    the balance between the two bands that separate dental from retroflex
    sibilants: dental s/z peaks around 5-8 kHz, retroflex sh/zh around 2.5-4 kHz
    because the larger sublingual cavity lowers the noise resonance.
    """
    times = utt.feats.times
    idx = np.flatnonzero((times >= start) & (times <= end))
    if idx.size < min_frames:
        # Too short to average: use whatever frames fall inside, else bail out.
        idx = np.flatnonzero((times >= start) & (times <= end))
        if idx.size == 0:
            return {}
    power = utt.feats.power[idx].astype(np.float64).mean(axis=0)
    freqs = utt.feats.freqs
    total = power.sum() + 1e-12

    def centroid_in(low: float, high: float) -> float:
        m = (freqs >= low) & (freqs < high)
        if not m.any():
            return float("nan")
        p = power[m]
        return float((p @ freqs[m]) / (p.sum() + 1e-12))

    def ratio(low: float, high: float) -> float:
        m = (freqs >= low) & (freqs < high)
        return float(power[m].sum() / total)

    low_band = centroid_in(2000.0, 4000.0)
    high_band = centroid_in(5000.0, 8000.0)
    ratio_low = ratio(2000.0, 4000.0)
    ratio_high = ratio(5000.0, 8000.0)
    return {
        "centroid": float((power @ freqs) / total),
        "centroid_low": low_band,
        "centroid_high": high_band,
        "ratio_low": ratio_low,
        "ratio_high": ratio_high,
        "high_low_db": float(10 * np.log10((ratio_high + 1e-12) / (ratio_low + 1e-12))),
        "frames": float(idx.size),
    }


def formant_track(signal: np.ndarray, sr: int, max_formant: float = 5500.0):
    """Burg formant track as (times, F1, F2, F3)."""
    import parselmouth

    sound = parselmouth.Sound(np.asarray(signal, dtype=np.float64), sampling_frequency=sr)
    formants = sound.to_formant_burg(
        time_step=0.005,
        max_number_of_formants=5,
        maximum_formant=max_formant,
        window_length=0.025,
    )
    times = np.arange(formants.n_frames) * formants.dt + formants.x1
    f1 = np.array([formants.get_value_at_time(1, t) for t in times])
    f2 = np.array([formants.get_value_at_time(2, t) for t in times])
    f3 = np.array([formants.get_value_at_time(3, t) for t in times])
    return times, f1, f2, f3


def vowel_formants(
    utt: Utterance,
    start: float,
    end: float,
    cache: dict | None = None,
) -> dict[str, float]:
    """F1/F2/F3 sampled across the steady-state middle of the vowel."""
    key = (round(start, 4), round(end, 4))
    if cache is not None and key in cache:
        return cache[key]

    times, f1, f2, f3 = formant_track(utt.signal, utt.sr)
    span = end - start
    if span <= 0:
        result: dict[str, float] = {}
    else:
        samples = {}
        for label, frac in (("early", 0.25), ("mid", 0.5), ("late", 0.75)):
            t = start + frac * span
            k = int(np.argmin(np.abs(times - t)))
            samples[label] = (float(f1[k]), float(f2[k]), float(f3[k]))
        mid = samples["mid"]
        late = samples["late"]
        result = {
            "f1": mid[0],
            "f2": mid[1],
            "f3": mid[2],
            "f1_early": samples["early"][0],
            "f2_early": samples["early"][1],
            "f2_late": late[1],
            "f3_late": late[2],
            # Velar pinch: the F2/F3 gap closes towards a velar nasal coda.
            "f3_f2_gap_mid": mid[2] - mid[1],
            "f3_f2_gap_late": late[2] - late[1],
        }
    if cache is not None:
        cache[key] = result
    return result


#: Reference formant values (Hz) for the vowel contrasts we test, from
#: standard Mandarin acoustic descriptions. Used only as sanity anchors.
VOWEL_REFERENCE = {
    "i": {"f1": 280, "f2": 2250, "f3": 3000},
    "v": {"f1": 280, "f2": 2100, "f3": 2100},  # ü: F3 collapses toward F2
    "a": {"f1": 800, "f2": 1200, "f3": 2600},
    "u": {"f1": 320, "f2": 800, "f3": 2400},
    "e": {"f1": 500, "f2": 1300, "f3": 2500},
    "o": {"f1": 500, "f2": 900, "f3": 2400},
}
