"""The traditional CAPT scorer.

Reads a clip, aligns it against the teacher's reading of the expected text,
then judges each syllable on two independent axes and reports the one with the
stronger evidence:

* **suprasegmental** - the F0 contour against the speaker-adapted tone
  prototypes (`tone_model`).
* **segmental** - the physical cue calibrated for that contrast (`segmental`:
  fricative centroid, VOT, F3, nasal pinch), falling back to a nearest-reference
  template comparison when a contrast has no dedicated cue.

Every judgement carries its raw measurements, not just a verdict, so thresholds
can be swept offline and the reasoning audited.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .align import Segment, align_utterance
from .audio import load_audio
from .dtw import dtw
from .dsp import cmn
from .features import (
    OnsetMeasurement,
    Utterance,
    measure_onset,
    prepare,
    vowel_formants,
)
from .f0 import F0Range
from .g2p import Syllable, analyze
from .inventory import Stimulus
from .segmental import (
    ContrastBoundary,
    SegmentalVerdict,
    judge_segmental,
    measure_feature,
)
from .tone_model import ToneModel

#: A calibrated cue must land this far past the decision boundary before the
#: syllable is called wrong. 0.5 is exactly on the midpoint between the two
#: reference realisations.
DEFAULT_SEGMENTAL_CONFIDENCE = 0.5

#: How much better a competing tone must fit, in units of the winner's own
#: calibration spread, before the system overrules the expected tone.
DEFAULT_TONE_MARGIN = 0.0

@dataclass
class ToneJudgement:
    expected_tone: int
    observed_tone: int
    ok: bool
    margin: float  # score(wrong best) - score(expected); > 0 means a mismatch
    scores: dict[int, float]
    distances: dict[int, float]
    features: dict[str, float]
    contour: list[float]
    range_source: str
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "expected_tone": self.expected_tone,
            "observed_tone": self.observed_tone,
            "ok": self.ok,
            "margin": round(self.margin, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "distances": {k: round(v, 4) for k, v in self.distances.items()},
            "features": {k: round(float(v), 3) for k, v in self.features.items()},
            "contour": [round(float(v), 3) for v in self.contour],
            "range_source": self.range_source,
            "notes": self.notes,
        }


@dataclass
class SyllableJudgement:
    index: int
    char: str
    expected_pinyin: str
    window: tuple[float, float]
    tone: ToneJudgement | None
    segmental: SegmentalVerdict | None
    template: dict | None
    dtw_cost: float
    onset: dict
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "char": self.char,
            "expected_pinyin": self.expected_pinyin,
            "window": [round(self.window[0], 4), round(self.window[1], 4)],
            "tone": self.tone.as_dict() if self.tone else None,
            "segmental": self.segmental.as_dict() if self.segmental else None,
            "template": self.template,
            "dtw_cost": round(self.dtw_cost, 4),
            "onset": {k: round(float(v), 4) for k, v in self.onset.items()},
            "notes": self.notes,
        }


@dataclass
class ClipJudgement:
    item_id: str
    scorer: str
    predicted_correct: bool
    predicted_dim: str | None
    predicted_unit: str | None
    confidence: float
    syllables: list[SyllableJudgement]
    latency_s: float
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "scorer": self.scorer,
            "predicted_correct": self.predicted_correct,
            "predicted_dim": self.predicted_dim,
            "predicted_unit": self.predicted_unit,
            "confidence": round(self.confidence, 4),
            "latency_s": round(self.latency_s, 4),
            "evidence": self.evidence,
            "syllables": [s.as_dict() for s in self.syllables],
        }


class TraditionalScorer:
    """Rule-and-measurement CAPT engine: no learned end-to-end model."""

    name = "traditional"

    def __init__(
        self,
        tone_model: ToneModel,
        boundaries: dict[str, ContrastBoundary] | None = None,
        learner_range: F0Range | None = None,
        segmental_confidence: float = DEFAULT_SEGMENTAL_CONFIDENCE,
        tone_margin: float = DEFAULT_TONE_MARGIN,
    ):
        self.tone_model = tone_model
        self.boundaries = boundaries or {}
        self.learner_range = learner_range
        self.segmental_confidence = segmental_confidence
        self.tone_margin = tone_margin
        self._ref_cache: dict[str, Utterance] = {}
        self._formant_cache: dict = {}

    # -- reference management -------------------------------------------
    def reference(self, path: Path) -> Utterance:
        key = str(path)
        if key not in self._ref_cache:
            signal, sr = load_audio(path)
            self._ref_cache[key] = prepare(signal, sr)
        return self._ref_cache[key]

    def clear_cache(self) -> None:
        self._ref_cache.clear()
        self._formant_cache.clear()

    # -- main entry point -----------------------------------------------
    def score(
        self,
        stimulus: dict,
        audio_path: Path,
        ref_path: Path,
        contrast_refs: dict[str, Path] | None = None,
    ) -> ClipJudgement:
        started = time.time()
        signal, sr = load_audio(audio_path)
        utt = prepare(signal, sr)
        ref_utt = self.reference(ref_path)

        expected_syllables = analyze(
            stimulus["expected_text"], sandhi=stimulus["family"] == "trap"
        )
        n_syllables = len(expected_syllables)
        labels = [s.display for s in expected_syllables]
        segments, ref_segments, alignment_cost = align_utterance(
            utt.feats, ref_utt.feats, n_syllables, labels=labels
        )

        focus = stimulus.get("focus_index", 0)
        span = stimulus.get("focus_span", 1)
        focus_indices = set(range(focus, min(focus + span, n_syllables)))

        syllable_judgements: list[SyllableJudgement] = []
        for i, syl in enumerate(expected_syllables):
            seg = segments[i] if i < len(segments) else Segment(0.0, utt.duration, syl.display)
            ref_seg = ref_segments[i] if i < len(ref_segments) else None
            syllable_judgements.append(
                self._judge_syllable(
                    utt=utt,
                    ref_utt=ref_utt,
                    seg=seg,
                    ref_seg=ref_seg,
                    syl=syl,
                    index=i,
                    is_focus=i in focus_indices,
                    stimulus=stimulus,
                    contrast_refs=contrast_refs or {},
                )
            )

        verdict = self._decide(syllable_judgements, focus_indices)
        return ClipJudgement(
            item_id=stimulus["id"],
            scorer=self.name,
            predicted_correct=verdict["correct"],
            predicted_dim=verdict["dim"],
            predicted_unit=verdict["unit"],
            confidence=verdict["confidence"],
            syllables=syllable_judgements,
            latency_s=time.time() - started,
            evidence={
                "alignment_cost": round(float(alignment_cost), 4)
                if np.isfinite(alignment_cost) else None,
                "n_syllables": n_syllables,
                **verdict["evidence"],
            },
        )

    # -- per-syllable analysis ------------------------------------------
    def _judge_syllable(
        self,
        utt: Utterance,
        ref_utt: Utterance,
        seg: Segment,
        ref_seg: Segment | None,
        syl: Syllable,
        index: int,
        is_focus: bool,
        stimulus: dict,
        contrast_refs: dict[str, Path],
    ) -> SyllableJudgement:
        notes: list[str] = []
        onset = measure_onset(utt, seg)

        # --- tone ------------------------------------------------------
        tone_judgement = None
        if syl.tone in (1, 2, 3, 4):
            from .f0 import extract_contour

            _, values = extract_contour(utt.f0, seg.start, seg.end, trim=0.04)
            decision = self.tone_model.classify_hz(values, self.learner_range)
            if decision.tone == 0:
                notes.extend(decision.notes)
                tone_judgement = ToneJudgement(
                    expected_tone=syl.tone, observed_tone=0, ok=False, margin=float("inf"),
                    scores={}, distances={}, features={}, contour=[],
                    range_source=decision.range_source, notes=decision.notes,
                )
            else:
                tone_judgement = ToneJudgement(
                    expected_tone=syl.tone,
                    observed_tone=decision.tone,
                    ok=decision.tone == syl.tone,
                    margin=self._tone_margin(decision, syl.tone),
                    scores=decision.scores,
                    distances=decision.distances,
                    features=decision.features,
                    contour=decision.contour,
                    range_source=decision.range_source,
                    notes=decision.notes,
                )

        # --- calibrated segmental cue ----------------------------------
        segmental = None
        boundary = self.boundaries.get(stimulus.get("contrast", ""))
        if boundary is not None and is_focus and stimulus.get("expected_unit"):
            value = measure_feature(
                boundary.feature_family,
                boundary.feature,
                utt,
                seg,
                onset,
                formant_cache=self._formant_cache,
            )
            segmental = judge_segmental(
                boundary,
                expected_unit=stimulus["expected_unit"],
                spoken_unit=stimulus.get("spoken_unit") or "",
                value=value,
            )

        # --- nearest-reference template comparison ---------------------
        template = self._compare_templates(
            utt, seg, contrast_refs, stimulus, self._ref_cache
        )

        # --- general alignment cost for this syllable ------------------
        dtw_cost = self._syllable_cost(utt, ref_utt, seg, ref_seg)

        return SyllableJudgement(
            index=index,
            char=syl.char,
            expected_pinyin=syl.display,
            window=(seg.start, seg.end),
            tone=tone_judgement,
            segmental=segmental,
            template=template,
            dtw_cost=dtw_cost,
            onset=onset.as_dict(),
            notes=notes,
        )

    def _tone_margin(self, decision, expected_tone: int) -> float:
        """Positive when some other tone fits better than the expected one."""
        if not decision.scores or expected_tone not in decision.scores:
            return float("inf")
        expected_score = decision.scores[expected_tone]
        best_wrong = min(
            (s for t, s in decision.scores.items() if t != expected_tone), default=float("inf")
        )
        return float(expected_score - best_wrong)

    def _compare_templates(
        self,
        utt: Utterance,
        seg: Segment,
        contrast_refs: dict[str, Path],
        stimulus: dict,
        cache: dict,
    ) -> dict | None:
        """DTW the test syllable against each contrast member's reference.

        The classic minimal-pair test, and the fallback for contrasts with no
        dedicated physical cue (r/l, n/l, uo/e).
        """
        if not contrast_refs:
            return None
        times = utt.feats.times
        idx = np.flatnonzero(seg.slice(times))
        if idx.size < 3:
            return None
        test = cmn(utt.feats.mfcc[idx, :26].astype(np.float64))

        costs: dict[str, float] = {}
        for unit, path in contrast_refs.items():
            ref_utt = cache.get(str(path))
            if ref_utt is None:
                signal, sr = load_audio(path)
                ref_utt = prepare(signal, sr)
                cache[str(path)] = ref_utt
            ref_idx = self._primary_segment_indices(ref_utt)
            ref = cmn(ref_utt.feats.mfcc[ref_idx, :26].astype(np.float64))
            if ref.shape[0] < 3:
                continue
            costs[unit] = float(dtw(test, ref, metric="euclidean").normalized)

        if not costs:
            return None
        expected_unit = stimulus.get("expected_unit")
        # Which axis this contrast lives on, so a template mismatch can be
        # reported as an initial vs final substitution. Correct items carry no
        # `error_dim`, so fall back to the family.
        dim = stimulus.get("error_dim") or (
            stimulus.get("family")
            if stimulus.get("family") in ("initial", "final")
            else None
        )
        if len(costs) == 1:
            return {"costs": {k: round(v, 3) for k, v in costs.items()},
                    "nearest": next(iter(costs)), "ratio": 1.0, "dim": dim}
        nearest = min(costs, key=lambda u: costs[u])
        expected_cost = costs.get(expected_unit, float("nan"))
        nearest_cost = costs[nearest]
        gap = (
            float(expected_cost - nearest_cost) if np.isfinite(expected_cost) else float("nan")
        )
        return {
            "costs": {k: round(v, 3) for k, v in costs.items()},
            "nearest": nearest,
            "expected_cost": round(float(expected_cost), 3),
            "nearest_cost": round(float(nearest_cost), 3),
            "gap": round(gap, 3),
            "fires": self._template_is_evidence(nearest, expected_unit, expected_cost, nearest_cost),
            "dim": dim,
        }

    @staticmethod
    def _template_is_evidence(
        nearest: str, expected_unit: str | None, expected_cost: float, nearest_cost: float
    ) -> bool:
        """Is "the nearest template is not the expected one" real evidence?

        A ratio is the wrong statistic here: a *perfect* match to the
        substituted member gives nearest_cost == 0, which is the strongest
        possible evidence, not a division-by-zero to be swallowed.

        The test is a gap, with an absolute floor (0.35) so that two tiny costs
        in a noisy short segment do not fire, and a relative floor (25% of the
        expected cost) so the call scales with how far off the expected member
        already is.
        """
        if expected_unit is None or nearest == expected_unit:
            return False
        if not (np.isfinite(expected_cost) and np.isfinite(nearest_cost)):
            return False
        gap = expected_cost - nearest_cost
        return gap > max(0.35, 0.25 * abs(expected_cost))

    @staticmethod
    def _primary_segment_indices(utt: Utterance) -> np.ndarray:
        """Frames of the voiced span - a single-syllable reference template."""
        from .align import voiced_span

        start, end = voiced_span(utt.feats)
        return np.flatnonzero((utt.feats.times >= start) & (utt.feats.times <= end))

    def _syllable_cost(
        self, utt: Utterance, ref_utt: Utterance, seg: Segment, ref_seg: Segment | None
    ) -> float:
        """DTW cost of this test syllable against the *same* syllable's reference.

        Using the aligned reference window (rather than the whole reference) is
        what makes this meaningful for sentence-length items.
        """
        idx = np.flatnonzero(seg.slice(utt.feats.times))
        if idx.size < 3:
            return float("nan")
        test = cmn(utt.feats.mfcc[idx, :26].astype(np.float64))

        if ref_seg is None:
            ref_idx = self._primary_segment_indices(ref_utt)
        else:
            ref_idx = np.flatnonzero(ref_seg.slice(ref_utt.feats.times))
        if ref_idx.size < 3:
            return float("nan")
        ref = cmn(ref_utt.feats.mfcc[ref_idx, :26].astype(np.float64))
        return float(dtw(test, ref, metric="euclidean").normalized)

    # -- verdict aggregation --------------------------------------------
    def _decide(self, syllables: list[SyllableJudgement], focus_indices: set[int]) -> dict:
        """Pick the single strongest error across the whole clip.

        Priority follows how trustworthy each signal is: a calibrated physical
        cue that clearly crossed its boundary beats a tone mismatch, which in
        turn beats a template-cost ratio.
        """
        candidates = []

        for s in syllables:
            if s.segmental is not None and s.segmental.expected not in ("", "?"):
                confidence = abs(s.segmental.score - 50.0) / 50.0
                if not s.segmental.ok and confidence >= self.segmental_confidence:
                    candidates.append(
                        {
                            "rank": 0,
                            "score": 1.0 + confidence,
                            "dim": s.segmental.dim,
                            "unit": s.segmental.observed,
                            "confidence": confidence,
                            "why": f"segmental cue {s.segmental.method}: observed "
                                   f"{s.segmental.observed}, expected {s.segmental.expected}",
                            "index": s.index,
                        }
                    )

            if s.tone is not None and s.tone.margin > self.tone_margin:
                confidence = float(np.clip(s.tone.margin, 0.0, 3.0) / 3.0)
                candidates.append(
                    {
                        "rank": 1,
                        "score": confidence,
                        "dim": "tone",
                        "unit": f"T{s.tone.observed_tone}",
                        "confidence": confidence,
                        "why": f"F0 contour matches T{s.tone.observed_tone} better than "
                               f"the expected T{s.tone.expected_tone} "
                               f"(margin {s.tone.margin:.2f})",
                        "index": s.index,
                    }
                )

            if s.template and s.template.get("fires"):
                gap = s.template.get("gap", 0.0)
                confidence = float(np.clip(gap / 3.0, 0.0, 1.0))
                candidates.append(
                    {
                        "rank": 2,
                        "score": confidence,
                        "dim": s.template.get("dim"),
                        "unit": s.template["nearest"],
                        "confidence": confidence,
                        "why": f"closer to the reference for {s.template['nearest']} "
                               f"(cost {s.template['nearest_cost']} vs "
                               f"{s.template['expected_cost']} for the expected member)",
                        "index": s.index,
                    }
                )

        if not candidates:
            worst = max(
                (abs(s.segmental.score - 50.0) / 50.0 for s in syllables
                 if s.segmental is not None and np.isfinite(s.segmental.score)),
                default=0.0,
            )
            return {
                "correct": True, "dim": None, "unit": None,
                "confidence": float(np.clip(worst, 0.0, 1.0)),
                "evidence": {"decision": "no error evidence above threshold"},
            }

        best = min(candidates, key=lambda c: (c["rank"], -c["score"]))
        return {
            "correct": False,
            "dim": best["dim"],
            "unit": best["unit"],
            "confidence": best["confidence"],
            "evidence": {
                "decision": best["why"],
                "decided_on_syllable": best["index"],
                "n_candidates": len(candidates),
            },
        }
