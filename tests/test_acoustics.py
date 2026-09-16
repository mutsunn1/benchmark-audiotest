"""Signal processing: framing, MFCC, F0 tracking, T-values and DTW."""

import numpy as np
import pytest

from captbench.dsp import (
    N_MFCC,
    extract_frames,
    frame_signal,
    mel_filterbank,
    preemphasis,
)
from captbench.dtw import dtw
from captbench.f0 import (
    F0Range,
    contour_features,
    extract_contour,
    resample_contour,
    to_t_values,
)

SR = 16000


def sine(freq: float, duration: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------- dsp
def test_preemphasis_is_a_first_order_highpass():
    signal = np.ones(100, dtype=np.float32)
    out = preemphasis(signal, 0.97)
    assert out[0] == pytest.approx(1.0)  # first sample passes through
    assert out[1] == pytest.approx(0.03, abs=1e-6)  # DC is attenuated
    assert np.all(np.abs(out[1:]) < 0.05)


def test_frame_signal_shape_and_hop():
    signal = sine(200, 1.0)
    frames, times = frame_signal(signal, SR)
    assert frames.shape[1] == int(0.025 * SR)
    # 1 s at a 10 ms hop, allowing for the final partial frame.
    assert abs(frames.shape[0] - 98) <= 2
    assert times[1] - times[0] == pytest.approx(0.01, abs=1e-9)


def test_mel_filterbank_is_triangular_and_covers_the_band():
    fb = mel_filterbank(SR, 512, 26)
    assert fb.shape == (26, 512 // 2 + 1)
    assert np.all(fb >= 0)
    # Every filter must respond to something.
    assert np.all(fb.sum(axis=1) > 0)
    # Filters should cover low to high frequencies.
    peaks = [int(np.argmax(fb[i])) for i in range(26)]
    assert peaks == sorted(peaks), "filter centres should increase with index"


def test_extract_frames_shapes():
    feats = extract_frames(sine(200, 1.0), SR)
    n = feats.times.size
    assert feats.mfcc.shape == (n, 39)
    assert feats.rms.shape == (n,)
    assert feats.zcr.shape == (n,)
    assert feats.power.shape[0] == n
    assert feats.freqs.size == feats.power.shape[1]
    assert feats.freqs[-1] == pytest.approx(8000, abs=1)


def test_spectral_centroid_tracks_the_tone_frequency():
    """The centroid is used for relative band comparisons, not absolute pitch.

    Pre-emphasis tilts the spectrum upward, so a pure tone's centroid sits
    above its nominal frequency. What matters is that it rises monotonically
    with the tone and stays well separated between the bands we compare.
    """
    centroids = [
        extract_frames(sine(f, 0.5), SR).centroid.mean() for f in (300, 1000, 3000)
    ]
    assert centroids == sorted(centroids), "centroid must increase with frequency"
    assert centroids[2] == pytest.approx(3000, rel=0.2)
    assert centroids[0] < 1000, "a low tone must not read as high-frequency energy"
    # The sibilant decision needs these two bands to stay far apart.
    assert centroids[2] / centroids[0] > 3


# ---------------------------------------------------------------- f0
def test_f0_tracking_finds_a_pure_tone():
    from captbench.f0 import track_f0

    track = track_f0(sine(200, 0.5), SR)
    voiced = track.f0[track.voiced]
    assert voiced.size > 10
    assert np.median(voiced) == pytest.approx(200, rel=0.05)


def test_f0_range_from_values_uses_robust_percentiles():
    values = np.concatenate([np.linspace(100, 300, 100), [10_000.0]])
    rng = F0Range.from_values(values, 5, 95)
    assert rng.fmax < 1000, "a single outlier must not set the range"
    assert rng.usable()


def test_f0_range_widens_when_degenerate():
    """A single monotone syllable must not produce a zero-width range."""
    rng = F0Range.from_values(np.full(50, 150.0))
    assert rng.usable()
    assert "widened" in rng.source


def test_t_values_map_the_range_onto_zero_to_five():
    rng = F0Range(100.0, 400.0)
    assert to_t_values(np.array([100.0]), rng)[0] == pytest.approx(0.0, abs=1e-6)
    assert to_t_values(np.array([400.0]), rng)[0] == pytest.approx(5.0, abs=1e-6)
    # Geometric midpoint sits at the middle of the scale.
    mid = float(np.sqrt(100.0 * 400.0))
    assert to_t_values(np.array([mid]), rng)[0] == pytest.approx(2.5, abs=1e-6)


def test_t_values_are_nan_for_unvoiced():
    rng = F0Range(100.0, 400.0)
    assert np.isnan(to_t_values(np.array([np.nan]), rng)[0])


def test_resample_contour_is_length_stable():
    times = np.linspace(0, 1, 37)
    values = np.linspace(0, 5, 37)
    out = resample_contour(times, values, 20)
    assert out.size == 20
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(5.0)


def test_contour_features_describe_a_falling_tone():
    falling = np.linspace(5, 0, 20)
    feats = contour_features(falling)
    assert feats["start"] > feats["end"]
    assert feats["slope"] < 0
    assert feats["late_fall"] == pytest.approx(5.0)
    assert feats["end_level"] < 1.0

    rising = np.linspace(0, 5, 20)
    rfeats = contour_features(rising)
    assert rfeats["slope"] > 0
    assert rfeats["late_rise"] == pytest.approx(5.0)

    dipping = np.concatenate([np.linspace(2, 0.3, 10), np.linspace(0.3, 4, 10)])
    dfeats = contour_features(dipping)
    assert dfeats["min"] < 0.5
    assert 0.3 < dfeats["dip_pos"] < 0.7


def test_extract_contour_returns_empty_when_unvoiced():
    from captbench.f0 import F0Track

    track = F0Track(
        times=np.linspace(0, 1, 50),
        f0=np.full(50, np.nan),
        voiced=np.zeros(50, dtype=bool),
    )
    times, values = extract_contour(track, 0.0, 1.0)
    assert times.size == 0 and values.size == 0


# ---------------------------------------------------------------- dtw
def test_dtw_identical_sequences_cost_nothing():
    a = np.linspace(0, 1, 30).reshape(-1, 1)
    result = dtw(a, a)
    assert result.distance == pytest.approx(0.0, abs=1e-9)
    assert result.path.shape[0] == 30


def test_dtw_is_smaller_for_a_closer_sequence():
    a = np.linspace(0, 1, 30).reshape(-1, 1)
    close = a + 0.05
    far = a + 2.0
    assert dtw(a, close).normalized < dtw(a, far).normalized


def test_dtw_handles_length_differences():
    a = np.linspace(0, 1, 20).reshape(-1, 1)
    b = np.linspace(0, 1, 35).reshape(-1, 1)
    result = dtw(a, b)
    assert np.isfinite(result.distance)
    # The path must start and end at the corners.
    assert tuple(result.path[0]) == (0, 0)
    assert tuple(result.path[-1]) == (19, 34)


def test_dtw_path_is_monotonic():
    a = np.sin(np.linspace(0, 3, 40)).reshape(-1, 1)
    b = np.sin(np.linspace(0, 3.2, 50)).reshape(-1, 1)
    path = dtw(a, b).path
    assert np.all(np.diff(path[:, 0]) >= 0)
    assert np.all(np.diff(path[:, 1]) >= 0)
