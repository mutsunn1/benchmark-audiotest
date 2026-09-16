"""Benchmark driver: calibrate the traditional engine, then score every item.

Calibration discipline, which is what makes the accuracy numbers mean anything:

* Tone prototypes and segmental boundaries are fitted on **teacher** readings
  only. Scored audio is **learner** takes, so no item is ever compared against
  itself.
* On top of that, the tone model for an item excludes the item's own contrast
  (leave-one-contrast-out), so a 妈 item is judged by prototypes fitted on
  shi/bao/cai/tang/mao. That measures cross-syllable generalisation rather than
  memorisation of the syllable under test.
* The learner's pitch range comes from a small enrolment set - correct
  productions in the learner's own voice - never from the clip being judged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .align import Segment, segment_syllables, voiced_span
from .audio import load_audio
from .config import Settings
from .corpus import Manifest
from .features import measure_onset, prepare
from .f0 import F0Range
from .g2p import analyze
from .segmental import ContrastBoundary, calibrate_boundary
from .tone_model import ToneModel, describe_model, fit_tone_model
from .traditional import TraditionalScorer


@dataclass
class ScorerSuite:
    """Everything the traditional engine needs, calibrated once."""

    manifest: Manifest
    data_dir: Path
    learner_ranges: dict[str, F0Range]
    tone_models: dict[str, ToneModel] = field(default_factory=dict)
    boundaries: dict[str, ContrastBoundary] = field(default_factory=dict)
    default_tone_model: ToneModel | None = None
    sentence_tone_model: ToneModel | None = None
    _ref_cache: dict = field(default_factory=dict)
    _formant_cache: dict = field(default_factory=dict)
    _scorers: dict[str, TraditionalScorer] = field(default_factory=dict)

    def _scorer(
        self, key: str, model: ToneModel | None, f0_range: F0Range
    ) -> TraditionalScorer:
        if key not in self._scorers:
            scorer = TraditionalScorer(
                tone_model=model,
                boundaries=self.boundaries,
                learner_range=f0_range,
            )
            # Share caches so references are loaded once across contrasts.
            scorer._ref_cache = self._ref_cache
            scorer._formant_cache = self._formant_cache
            self._scorers[key] = scorer
        return self._scorers[key]

    def scorer_for(self, item: dict) -> TraditionalScorer:
        """Pick the tone model matching the item's prosodic context.

        Tone realisations in connected speech differ systematically from
        isolated citation forms, so the two contexts get their own prototypes.
        """
        n_syllables = len(analyze(item.get("expected_text", "") or "", sandhi=True))
        if n_syllables > 1 and self.sentence_tone_model is not None:
            return self._scorer(
                f"sentence::{item.get('contrast', '')}",
                self.sentence_tone_model,
                self.learner_ranges["sentence"],
            )
        model = self.tone_models.get(item.get("contrast", "")) or self.default_tone_model
        return self._scorer(
            f"word::{item.get('contrast', '')}", model, self.learner_ranges["word"]
        )


def _tone_samples_from_utterance(utt, windows) -> list[tuple[np.ndarray, int]]:
    """Pull one voiced F0 sample per labelled window."""
    from .f0 import extract_contour

    samples = []
    for start, end, tone in windows:
        _, values = extract_contour(utt.f0, start, end, trim=0.04)
        if values.size >= 4 and tone in (1, 2, 3, 4):
            samples.append((values, tone))
    return samples


def learner_f0_ranges(manifest: Manifest, data_dir: Path) -> dict[str, F0Range]:
    """Estimate the learner's pitch range, separately per prosodic context.

    The range must match the context it will be used in. Isolated words sit in
    a narrow band; connected speech swings much wider, so one pooled range makes
    the test contour and the prototypes land in different T-value spaces and
    systematically misfires (a level 阴平 projected with a sentence-wide range
    looks like a fall, and gets classified as 去声).

    The teacher side is split the same way: word prototypes are fitted on
    isolated word readings, sentence prototypes on the calibration sentences.
    """
    pooled: dict[str, list[np.ndarray]] = {"word": [], "sentence": []}
    for entry in manifest.enrollment:
        path = data_dir / entry["audio"]
        if not path.exists():
            continue
        signal, sr = load_audio(path)
        utt = prepare(signal, sr)
        values = utt.f0.f0[np.isfinite(utt.f0.f0)]
        if values.size < 4:
            continue
        context = "sentence" if entry.get("contrast") == "calib_sentence" else "word"
        pooled[context].append(values)

    ranges: dict[str, F0Range] = {}
    for context, chunks in pooled.items():
        if chunks:
            ranges[context] = F0Range.from_values(
                np.concatenate(chunks), 2.0, 98.0, source=f"learner-enrolment:{context}"
            )
    # Fall back to whichever context we do have.
    fallback = ranges.get("word") or ranges.get("sentence") or F0Range(90.0, 240.0, "default")
    return {
        "word": ranges.get("word", fallback),
        "sentence": ranges.get("sentence", fallback),
    }


def build_suite(
    settings: Settings,
    manifest: Manifest,
    verbose: bool = True,
) -> ScorerSuite:
    data_dir = settings.data_dir
    ranges = learner_f0_ranges(manifest, data_dir)
    suite = ScorerSuite(
        manifest=manifest,
        data_dir=data_dir,
        learner_ranges=ranges,
    )
    if verbose:
        for context, rng in ranges.items():
            print(f"learner pitch range [{context}]: {rng.fmin:.0f}-{rng.fmax:.0f} Hz")

    tone_refs = [r for r in manifest.references.values() if r.get("tone")]
    contrasts = sorted({r["contrast"] for r in tone_refs})

    # Leave-one-contrast-out tone models.
    for held_out in contrasts:
        samples = []
        for ref in tone_refs:
            if ref["contrast"] == held_out:
                continue
            path = data_dir / ref["audio"]
            if not path.exists():
                continue
            signal, sr = load_audio(path)
            utt = prepare(signal, sr)
            values = utt.f0.f0[np.isfinite(utt.f0.f0)]
            if values.size >= 4:
                samples.append((values, int(ref["tone"])))
        if samples:
            suite.tone_models[held_out] = fit_tone_model(samples)

    # Default model: every tone reference, for contrasts outside the tone sets.
    all_samples = []
    for ref in tone_refs:
        path = data_dir / ref["audio"]
        if not path.exists():
            continue
        signal, sr = load_audio(path)
        utt = prepare(signal, sr)
        values = utt.f0.f0[np.isfinite(utt.f0.f0)]
        if values.size >= 4:
            all_samples.append((values, int(ref["tone"])))
    if all_samples:
        suite.default_tone_model = fit_tone_model(all_samples)

    # Sentence-context prototypes, from the calibration sentences only.
    sentence_samples: list[tuple[np.ndarray, int]] = []
    for entry in manifest.sentence_calibration:
        path = data_dir / entry["audio"]
        if not path.exists():
            continue
        signal, sr = load_audio(path)
        utt = prepare(signal, sr)
        tones = entry["tones"]
        segments = segment_syllables(utt.feats, len(tones))
        sentence_samples.extend(
            _tone_samples_from_utterance(
                utt,
                [(seg.start, seg.end, tone) for seg, tone in zip(segments, tones)],
            )
        )
    if sentence_samples:
        suite.sentence_tone_model = fit_tone_model(sentence_samples)
        if verbose:
            print(f"sentence-context tone prototypes from "
                  f"{len(sentence_samples)} syllables")
            print(describe_model(suite.sentence_tone_model))

    if verbose and suite.default_tone_model is not None:
        print(describe_model(suite.default_tone_model))

    suite.boundaries = calibrate_boundaries(manifest, data_dir, verbose=verbose)
    return suite


def calibrate_boundaries(
    manifest: Manifest, data_dir: Path, verbose: bool = True
) -> dict[str, ContrastBoundary]:
    """Measure each contrast's decision boundary on its two reference readings."""
    required = {
        "sibilant": ("zh_z", "ch_c", "sh_s", "sh_s2"),
        "vot": ("b_p", "d_t", "g_k", "j_q"),
        "rounding": ("v_i", "v_i2"),
        "nasal": ("an_ang", "en_eng", "in_ing"),
    }
    # Which two units define each contrast's axis.
    axes = {
        "zh_z": ("zh", "z"), "ch_c": ("ch", "c"), "sh_s": ("sh", "s"), "sh_s2": ("sh", "s"),
        "b_p": ("p", "b"), "d_t": ("t", "d"), "g_k": ("k", "g"), "j_q": ("q", "j"),
        "v_i": ("i", "v"), "v_i2": ("i", "v"),
        "an_ang": ("ang", "an"), "en_eng": ("eng", "en"), "in_ing": ("ing", "in"),
    }

    boundaries: dict[str, ContrastBoundary] = {}
    _utt_cache: dict[str, object] = {}

    def _load(path: Path):
        key = str(path)
        if key not in _utt_cache:
            signal, sr = load_audio(path)
            _utt_cache[key] = prepare(signal, sr)
        return _utt_cache[key]

    for family, contrast_ids in required.items():
        for contrast in contrast_ids:
            refs = manifest.references_for(contrast)
            unit_high, unit_low = axes[contrast]
            if unit_high not in refs or unit_low not in refs:
                if verbose:
                    print(f"  boundary {contrast}: missing reference "
                          f"({unit_high}/{unit_low}) - skipped")
                continue
            utt_high = _load(data_dir / refs[unit_high]["audio"])
            utt_low = _load(data_dir / refs[unit_low]["audio"])
            seg_high = Segment(*voiced_span(utt_high.feats))
            seg_low = Segment(*voiced_span(utt_low.feats))
            onset_high = measure_onset(utt_high, seg_high)
            onset_low = measure_onset(utt_low, seg_low)

            boundary = calibrate_boundary(
                contrast, utt_high, seg_high, onset_high, utt_low, seg_low, onset_low
            )
            if boundary is not None:
                boundaries[contrast] = boundary
                if verbose:
                    print(
                        f"  boundary {contrast:7} {boundary.feature:16} "
                        f"thr={boundary.threshold:9.2f} "
                        f"({boundary.value_high:.2f} vs {boundary.value_low:.2f}, "
                        f"sep={boundary.separation:.2f})"
                    )
    return boundaries


def run_traditional(
    suite: ScorerSuite,
    limit: int | None = None,
    item_ids: set[str] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Score every manifest item with the traditional engine."""
    items = suite.manifest.scored_items()
    if item_ids:
        items = [i for i in items if i["id"] in item_ids]
    if limit:
        items = items[:limit]

    results: list[dict] = []
    started = time.time()
    for n, item in enumerate(items, start=1):
        if verbose and (n % 20 == 0 or n == len(items)):
            print(f"  scoring [{n}/{len(items)}]", flush=True)
        scorer = suite.scorer_for(item)
        contrast_refs = {
            unit: suite.data_dir / ref["audio"]
            for unit, ref in suite.manifest.references_for(item.get("contrast", "")).items()
        }
        try:
            judgement = scorer.score(
                stimulus=item,
                audio_path=suite.data_dir / item["audio"],
                ref_path=suite.data_dir / item["ref_audio"],
                contrast_refs=contrast_refs,
            )
            results.append(judgement.as_dict())
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the run
            results.append(
                {
                    "item_id": item["id"],
                    "scorer": "traditional",
                    "error": f"{type(exc).__name__}: {exc}",
                    "predicted_correct": None,
                    "latency_s": 0.0,
                }
            )
    if verbose:
        print(f"traditional: scored {len(results)} items in {time.time() - started:.1f}s")
    return results
