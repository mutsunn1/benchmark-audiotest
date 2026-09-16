"""Segmental (initial/final) error detection from physical measurements.

Each check measures one acoustic correlate on the test clip and compares it to
a decision boundary. The boundary is not hard-coded: it is measured on the
*reference* productions of both members of the contrast, so a threshold sits
midway between this voice's realisations of /s/ and /ʂ/ rather than at a value
chosen for some other speaker.

Contrasts with no dedicated physical correlate (r/l, n/l, uo/e) fall through to
the general DTW segmental cost against the reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .align import Segment
from .features import Utterance, fricative_profile, measure_onset, vowel_formants

#: Which acoustic measurement separates each contrast, and in which direction.
#: The value is the feature key; `higher_means` records which member scores high.
PHONE_CONTRASTS: dict[str, dict[str, str]] = {
    # Fricative noise resonance: retroflex has a larger sublingual cavity, so
    # the noise energy sits lower (2.5-4 kHz) than dental (5-8 kHz).
    "zh_z": {"feature": "sibilant", "high": "dental", "low": "retroflex"},
    "ch_c": {"feature": "sibilant", "high": "dental", "low": "retroflex"},
    "sh_s": {"feature": "sibilant", "high": "dental", "low": "retroflex"},
    "sh_s2": {"feature": "sibilant", "high": "dental", "low": "retroflex"},
    # Aspiration is VOT in Mandarin, not voicing.
    "b_p": {"feature": "vot", "high": "aspirated", "low": "unaspirated"},
    "d_t": {"feature": "vot", "high": "aspirated", "low": "unaspirated"},
    "g_k": {"feature": "vot", "high": "aspirated", "low": "unaspirated"},
    "j_q": {"feature": "vot", "high": "aspirated", "low": "unaspirated"},
    # ü lowers F3 by roughly a kilohertz against i.
    "v_i": {"feature": "rounding", "high": "i", "low": "v"},
    "v_i2": {"feature": "rounding", "high": "i", "low": "v"},
    # Velar nasals pinch F2/F3 together in the coda.
    "an_ang": {"feature": "nasal", "high": "alveolar", "low": "velar"},
    "en_eng": {"feature": "nasal", "high": "alveolar", "low": "velar"},
    "in_ing": {"feature": "nasal", "high": "alveolar", "low": "velar"},
}

#: Candidate measurements per feature family. The calibrator measures all of
#: them on the reference pair and keeps whichever separates the two best.
FEATURE_CANDIDATES: dict[str, list[str]] = {
    "sibilant": ["centroid", "centroid_high", "high_low_db"],
    "vot": ["vot", "aperiodic"],
    "rounding": ["f3", "f3_f2_gap_mid", "f3_late"],
    "nasal": ["f2_late", "f3_f2_gap_late", "f2"],
}


@dataclass
class ContrastBoundary:
    """A decision boundary measured between two reference realisations."""

    contrast: str
    feature_family: str
    feature: str  # the winning candidate measurement
    threshold: float
    value_high: float  # measured on the member named in `high`
    value_low: float  # measured on the member named in `low`
    separation: float  # relative separation; low values mean an unreliable cue
    labels: tuple[str, str]  # (high_label, low_label)

    def classify(self, value: float) -> str:
        if not np.isfinite(value):
            return "unknown"
        return self.labels[0] if value >= self.threshold else self.labels[1]

    def confidence(self, value: float) -> float:
        """0.5 = sitting on the boundary, 1.0 = deep in one class."""
        span = abs(self.value_high - self.value_low)
        if span < 1e-9 or not np.isfinite(value):
            return 0.0
        return float(min(1.0, 0.5 + abs(value - self.threshold) / span))


@dataclass
class SegmentalVerdict:
    dim: str  # "initial" | "final" | "none"
    ok: bool
    observed: str  # what the acoustics say the learner produced
    expected: str
    score: float  # 0-100, mapped from the distance to the boundary
    method: str
    notes: list[str] = field(default_factory=list)
    evidence: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "dim": self.dim,
            "ok": self.ok,
            "observed": self.observed,
            "expected": self.expected,
            "score": self.score,
            "method": self.method,
            "notes": self.notes,
            "evidence": {k: round(float(v), 4) for k, v in self.evidence.items()},
        }


def measure_feature(
    family: str,
    name: str,
    utt: Utterance,
    seg: Segment,
    onset,
    formant_cache: dict | None = None,
) -> float:
    """Evaluate one named measurement on one syllable window."""
    if family == "sibilant":
        if onset is None:
            return float("nan")
        profile = fricative_profile(utt, onset.onset, onset.voicing_onset)
        return float(profile.get(name, float("nan")))
    if family == "vot":
        if onset is None:
            return float("nan")
        return float(getattr(onset, name, float("nan")))
    if family == "rounding":
        values = vowel_formants(utt, seg.start, seg.end, cache=formant_cache)
        return float(values.get(name, float("nan")))
    if family == "nasal":
        values = vowel_formants(utt, seg.start, seg.end, cache=formant_cache)
        return float(values.get(name, float("nan")))
    raise KeyError(f"unknown feature family {family}")


def calibrate_boundary(
    contrast: str,
    utt_high: Utterance,
    seg_high: Segment,
    onset_high,
    utt_low: Utterance,
    seg_low: Segment,
    onset_low,
) -> ContrastBoundary | None:
    """Measure every candidate feature on both reference members.

    Keeps the candidate that separates them most, relative to their magnitude.
    A cue that barely separates the two references cannot be trusted to judge a
    learner, and is reported as low-confidence rather than silently used.
    """
    spec = PHONE_CONTRASTS.get(contrast)
    if spec is None:
        return None
    family = spec["feature"]
    best: ContrastBoundary | None = None

    for name in FEATURE_CANDIDATES[family]:
        v_high = measure_feature(family, name, utt_high, seg_high, onset_high)
        v_low = measure_feature(family, name, utt_low, seg_low, onset_low)
        if not (np.isfinite(v_high) and np.isfinite(v_low)):
            continue
        scale = abs(v_high) + abs(v_low) + 1e-9
        separation = abs(v_high - v_low) / scale
        if best is None or separation > best.separation:
            best = ContrastBoundary(
                contrast=contrast,
                feature_family=family,
                feature=name,
                threshold=(v_high + v_low) / 2.0,
                value_high=v_high,
                value_low=v_low,
                separation=separation,
                labels=(spec["high"], spec["low"]),
            )
    return best


#: Which acoustic class each phone belongs to, per feature family.
_SIBILANT_RETROFLEX = {"zh", "ch", "sh", "r"}
_SIBILANT_DENTAL = {"z", "c", "s"}
_ASPIRATED = {"p", "t", "k", "q", "c", "ch"}
_UNASPIRATED = {"b", "d", "g", "j", "z", "zh"}


def unit_to_class(family: str, unit: str) -> str | None:
    """Map a phone symbol onto the class name a boundary uses."""
    if not unit:
        return None
    if family == "sibilant":
        if unit in _SIBILANT_RETROFLEX:
            return "retroflex"
        if unit in _SIBILANT_DENTAL:
            return "dental"
        return None
    if family == "vot":
        if unit in _ASPIRATED:
            return "aspirated"
        if unit in _UNASPIRATED:
            return "unaspirated"
        return None
    if family == "rounding":
        return unit if unit in ("i", "v") else None
    if family == "nasal":
        if unit.endswith("ng"):
            return "velar"
        if unit.endswith("n"):
            return "alveolar"
        return None
    return None


def judge_segmental(
    boundary: ContrastBoundary | None,
    expected_unit: str,
    spoken_unit: str,
    value: float,
) -> SegmentalVerdict:
    """Compare a measurement against a calibrated boundary and name the error."""
    if boundary is None:
        return SegmentalVerdict(
            dim="none", ok=True, observed="?", expected=expected_unit,
            score=float("nan"), method="uncalibrated",
            notes=["no calibrated cue for this contrast"],
        )

    family = boundary.feature_family
    observed = boundary.classify(value)
    expected_class = unit_to_class(family, expected_unit)
    spoken_class = unit_to_class(family, spoken_unit)

    if expected_class is None:
        return SegmentalVerdict(
            dim="none", ok=True, observed=observed, expected=expected_unit,
            score=float("nan"), method=f"{family}:{boundary.feature}",
            notes=[f"expected unit {expected_unit!r} has no class in this family"],
        )

    ok = observed == expected_class
    confidence = boundary.confidence(value)
    # 50 is the boundary; deep inside the right class approaches 100.
    score = 50.0 + (50.0 if ok else -50.0) * confidence

    notes = []
    if boundary.separation < 0.05:
        notes.append(
            f"weak cue: {boundary.feature} differs by only "
            f"{boundary.separation:.1%} between the two reference readings"
        )
    if not np.isfinite(value):
        notes.append(f"{boundary.feature} could not be measured on this clip")
    if not ok and spoken_class is not None and observed == spoken_class:
        notes.append(
            f"measured value matches the substituted phone {spoken_unit!r} "
            "rather than the expected one"
        )

    return SegmentalVerdict(
        dim="initial" if family in ("sibilant", "vot") else "final",
        ok=bool(ok),
        observed=observed,
        expected=expected_class,
        score=float(np.clip(score, 0.0, 100.0)),
        method=f"{family}:{boundary.feature}",
        notes=notes,
        evidence={
            "value": value,
            "threshold": boundary.threshold,
            "reference_high": boundary.value_high,
            "reference_low": boundary.value_low,
            "separation": boundary.separation,
        },
    )
