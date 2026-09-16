"""Speaker-adapted tone prototypes: the tone classifier.

Canonical 55/35/214/51 templates assume an idealised speaker who swings across
their whole pitch range on every tone. Real speakers - and every TTS voice -
do not. Measured on this corpus, 阳平 rises 195 -> 223 Hz while 阴平 sits at
256-280 Hz, so a contour judged against absolute templates lands on the wrong
tone even though it is perfectly intelligible.

The fix is the one CAPT practice has always used: a *calibration set*. The
report's own normalisation step takes Fmin/Fmax from "the speaker's whole
sentence or pronunciation calibration set". This module extends that same idea
from the range to the contours - build each tone's prototype from the speaker's
own correct productions, then classify by DTW distance to those prototypes.

To keep the benchmark honest, callers must build the model from clips the item
under test is not drawn from (leave-one-contrast-out). A prototype fitted on the
very clip it later judges would measure memorisation, not recognition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dtw import dtw
from .f0 import F0Range, contour_features, resample_contour, template_curve, to_t_values

N_POINTS = 20
MIN_SAMPLES_PER_TONE = 2


@dataclass
class TonePrototype:
    tone: int
    contour: np.ndarray  # (N_POINTS,) mean T-value contour
    count: int
    spread: float  # mean distance from members to the prototype
    threshold: float  # accept/reject radius, from the calibration distances
    source: str = "calibrated"


@dataclass
class ToneModel:
    f0_range: F0Range
    prototypes: dict[int, TonePrototype]
    n_points: int = N_POINTS
    source: str = "calibrated"

    def available_tones(self) -> list[int]:
        return sorted(self.prototypes)

    def classify(self, contour: np.ndarray) -> "ToneDecision":
        """Nearest-prototype tone for a contour already in T-value space."""
        if contour.size == 0:
            return ToneDecision(
                tone=0, distances={}, scores={}, features={},
                notes=["no measurable pitch contour - cannot judge tone"],
            )

        distances: dict[int, float] = {}
        for tone, proto in self.prototypes.items():
            distances[tone] = dtw(
                contour.reshape(-1, 1), proto.contour.reshape(-1, 1)
            ).normalized

        # Normalise so the spread of each class sets its own scale: a tone the
        # speaker realises inconsistently should not be penalised for it.
        scores = {
            tone: distances[tone] / max(self.prototypes[tone].spread, 1e-3)
            for tone in distances
        }
        best = min(scores, key=lambda t: scores[t])
        notes = []
        if distances[best] > self.prototypes[best].threshold:
            notes.append(
                f"contour is {distances[best]:.2f} from the nearest prototype "
                f"(accept radius {self.prototypes[best].threshold:.2f}) - "
                "atypical even for the closest tone"
            )
        return ToneDecision(
            tone=best,
            distances=distances,
            scores=scores,
            features=contour_features(contour),
            notes=notes,
            contour=[float(v) for v in contour],
        )

    def classify_hz(
        self,
        values: np.ndarray,
        learner_range: F0Range | None = None,
        min_frames: int = 4,
    ) -> "ToneDecision":
        """Classify a raw Hz contour from the test speaker.

        The prototypes live in the *calibration* speaker's T-value space. The
        test contour is projected with the *learner's* own range, which is what
        makes the comparison meaningful across speakers: both sides are
        expressed relative to their own vocal range rather than in absolute Hz.
        """
        rng = learner_range if (learner_range and learner_range.usable()) else self.f0_range
        vals = np.asarray(values, dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < min_frames:
            decision = ToneDecision(
                tone=0, distances={}, scores={}, features={},
                notes=[f"only {vals.size} voiced frames in this window - "
                       "cannot judge tone"],
            )
            decision.range_source = rng.source
            return decision

        tvals = to_t_values(vals, rng)
        tvals = tvals[np.isfinite(tvals)]
        contour = resample_contour(np.arange(tvals.size, dtype=np.float64), tvals, self.n_points)
        decision = self.classify(contour)
        decision.range_source = rng.source
        return decision


@dataclass
class ToneDecision:
    tone: int
    distances: dict[int, float]
    scores: dict[int, float]
    features: dict[str, float]
    notes: list[str] = field(default_factory=list)
    contour: list[float] = field(default_factory=list)  # the T-value contour judged
    range_source: str = ""  # which F0 range was used to project the contour

    def ranked(self) -> list[tuple[int, float]]:
        return sorted(self.scores.items(), key=lambda kv: kv[1])

    def margin(self) -> float:
        """How much better the winner is than the runner-up (in spread units)."""
        order = self.ranked()
        if len(order) < 2:
            return float("inf")
        return order[1][1] - order[0][1]


def canonical_model(f0_range: F0Range, n_points: int = N_POINTS) -> ToneModel:
    """Fallback model built from idealised 55/35/214/51 templates."""
    prototypes = {}
    for tone in (1, 2, 3, 4):
        curve = np.array(template_curve(tone, 0, n_points))
        prototypes[tone] = TonePrototype(
            tone=tone, contour=curve, count=0, spread=0.5, threshold=2.5,
            source="canonical",
        )
    return ToneModel(f0_range=f0_range, prototypes=prototypes,
                     n_points=n_points, source="canonical")


def fit_tone_model(
    samples: list[tuple[np.ndarray, int]],
    n_points: int = N_POINTS,
    threshold_pct: float = 90.0,
) -> ToneModel:
    """Build prototypes from (voiced F0 values, tone) pairs.

    `samples` must come from correct productions only - an error production
    folded into a prototype would teach the model that the error is correct.
    """
    usable = [(np.asarray(v, dtype=np.float64), t) for v, t in samples if np.size(v) >= 4]
    if not usable:
        return canonical_model(F0Range(90.0, 220.0), n_points)

    pooled = np.concatenate([v[np.isfinite(v) & (v > 0)] for v, _ in usable])
    f0_range = F0Range.from_values(pooled, 2.0, 98.0, source="calibration")

    contours: dict[int, list[np.ndarray]] = {}
    for values, tone in usable:
        tvals = to_t_values(values, f0_range)
        tvals = tvals[np.isfinite(tvals)]
        if tvals.size < 4:
            continue
        # Evenly spaced in time by construction: F0 frames are uniform.
        time = np.arange(tvals.size, dtype=np.float64)
        contours.setdefault(tone, []).append(resample_contour(time, tvals, n_points))

    prototypes: dict[int, TonePrototype] = {}
    for tone, items in contours.items():
        if len(items) < MIN_SAMPLES_PER_TONE:
            continue
        stack = np.vstack(items)
        mean_curve = stack.mean(axis=0)
        dists = np.array([
            dtw(c.reshape(-1, 1), mean_curve.reshape(-1, 1)).normalized for c in items
        ])
        prototypes[tone] = TonePrototype(
            tone=tone,
            contour=mean_curve,
            count=len(items),
            spread=float(dists.mean()) if dists.size else 0.5,
            threshold=float(np.percentile(dists, threshold_pct)) if dists.size else 2.0,
        )

    missing = [t for t in (1, 2, 3, 4) if t not in prototypes]
    if missing:
        fallback = canonical_model(f0_range, n_points)
        for tone in missing:
            prototypes[tone] = fallback.prototypes[tone]

    return ToneModel(f0_range=f0_range, prototypes=prototypes,
                     n_points=n_points, source="calibrated")


def extract_tone_sample(
    track,
    start: float,
    end: float,
    trim: float = 0.04,
) -> np.ndarray:
    """Voiced F0 values for one syllable window, ready to feed a model."""
    from .f0 import extract_contour

    _, values = extract_contour(track, start, end, trim=trim)
    return values


# --------------------------------------------------------------------------
# Precomputed shape anchors, used only to sanity-check the fitted model.
# --------------------------------------------------------------------------
CANONICAL_SHAPES = {1: "high level", 2: "rising", 3: "dipping", 4: "falling"}


def describe_model(model: ToneModel) -> str:
    lines = [f"tone model ({model.source}), F0 {model.f0_range.fmin:.0f}-"
             f"{model.f0_range.fmax:.0f} Hz"]
    for tone in sorted(model.prototypes):
        proto = model.prototypes[tone]
        c = proto.contour
        lines.append(
            f"  T{tone} ({CANONICAL_SHAPES.get(tone, '?')}): "
            f"n={proto.count} start={c[0]:.1f} end={c[-1]:.1f} "
            f"min={c.min():.1f} max={c.max():.1f} "
            f"spread={proto.spread:.2f} thr={proto.threshold:.2f} [{proto.source}]"
        )
    return "\n".join(lines)
