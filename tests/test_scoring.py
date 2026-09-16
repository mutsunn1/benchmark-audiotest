"""Inventory consistency, segmental decisions, tone model and metrics."""

import numpy as np
import pytest

from captbench.g2p import analyze
from captbench.inventory import (
    CALIBRATION_SENTENCES,
    build_inventory,
    segmental_items,
    tone_items,
    trap_items,
)
from captbench.metrics import (
    Record,
    consistency,
    detection,
    localisation,
    scorecard,
    tone_identification,
    trap_analysis,
)
from captbench.f0 import F0Range
from captbench.segmental import ContrastBoundary, judge_segmental, unit_to_class
from captbench.tone_model import canonical_model, fit_tone_model

# ------------------------------------------------------------- inventory
def test_inventory_has_no_duplicate_ids():
    items = build_inventory()
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids))


def test_every_error_item_carries_a_label_and_dimension():
    for item in build_inventory():
        if item.is_correct:
            assert item.error_dim is None
            assert item.error_label is None
        else:
            assert item.error_dim, item.id
            assert item.error_label, item.id


def test_correct_items_have_identical_expected_and_spoken_text():
    for item in build_inventory():
        if item.is_correct:
            assert item.expected_text == item.spoken_text, item.id


def test_tone_items_cover_the_full_substitution_matrix():
    items = tone_items()
    for contrast in {i.contrast for i in items}:
        group = [i for i in items if i.contrast == contrast]
        pairs = {(i.expected_unit, i.spoken_unit) for i in group}
        assert len(pairs) == 16, f"{contrast} should cover all 4x4 tone pairs"
        assert sum(1 for i in group if i.is_correct) == 4


def test_segmental_items_include_both_correct_readings_and_both_substitutions():
    for contrast in {i.contrast for i in segmental_items()}:
        group = [i for i in segmental_items() if i.contrast == contrast]
        assert len(group) == 4, contrast
        assert sum(1 for i in group if i.is_correct) == 2
        assert {i.expected_text for i in group} == {g.expected_text for g in group[:2]} | {
            g.spoken_text for g in group[:2]
        }


def test_trap_items_mark_the_focus_word_and_a_substitution():
    for contrast in {i.contrast for i in trap_items()}:
        group = [i for i in trap_items() if i.contrast == contrast]
        assert len(group) == 2
        correct = next(i for i in group if i.is_correct)
        wrong = next(i for i in group if not i.is_correct)
        assert correct.expected_text == correct.spoken_text
        assert wrong.expected_text != wrong.spoken_text
        # The substituted reading must be a real minimal-pair swap on the focus.
        exp = analyze(wrong.expected_text, sandhi=True)[wrong.focus_index]
        got = analyze(wrong.spoken_text, sandhi=True)[wrong.focus_index]
        assert exp.tone != got.tone
        assert exp.initial == got.initial and exp.final == got.final


def test_calibration_sentences_are_multisyllabic():
    for sentence in CALIBRATION_SENTENCES:
        assert len(analyze(sentence, sandhi=True)) >= 4


# ------------------------------------------------------------- segmental
@pytest.mark.parametrize(
    "family,unit,expected_class",
    [
        ("sibilant", "zh", "retroflex"),
        ("sibilant", "s", "dental"),
        ("vot", "p", "aspirated"),
        ("vot", "b", "unaspirated"),
        ("rounding", "v", "v"),
        ("rounding", "i", "i"),
        ("nasal", "ang", "velar"),
        ("nasal", "an", "alveolar"),
    ],
)
def test_unit_to_class(family, unit, expected_class):
    assert unit_to_class(family, unit) == expected_class


def _boundary(**overrides) -> ContrastBoundary:
    base = dict(
        contrast="sh_s",
        feature_family="sibilant",
        feature="high_low_db",
        threshold=0.0,
        value_high=5.0,  # dental reference, higher centroid
        value_low=-5.0,  # retroflex reference
        separation=0.5,
        labels=("dental", "retroflex"),
    )
    base.update(overrides)
    return ContrastBoundary(**base)


def test_boundary_classifies_and_scores_sides():
    boundary = _boundary()
    assert boundary.classify(3.0) == "dental"
    assert boundary.classify(-3.0) == "retroflex"
    assert boundary.confidence(5.0) > boundary.confidence(0.1)


def test_judge_segmental_accepts_a_correct_retroflex_and_flags_a_dental():
    boundary = _boundary()
    ok = judge_segmental(boundary, expected_unit="sh", spoken_unit="sh", value=-4.0)
    assert ok.ok and ok.dim == "initial" and ok.score > 50

    wrong = judge_segmental(boundary, expected_unit="sh", spoken_unit="s", value=4.0)
    assert not wrong.ok
    assert wrong.observed == "dental"
    assert any("substituted phone" in n for n in wrong.notes)


def test_judge_segmental_without_a_boundary_is_neutral():
    verdict = judge_segmental(None, expected_unit="sh", spoken_unit="sh", value=0.0)
    assert verdict.dim == "none"
    assert np.isnan(verdict.score)


# ------------------------------------------------------------- tone model
def _contour(values):
    return np.asarray(values, dtype=float)


def test_fit_tone_model_learns_prototype_shapes():
    samples = []
    for _ in range(4):
        samples.append((np.geomspace(220, 220, 20), 1))  # level high
        samples.append((np.geomspace(180, 260, 20), 2))  # rising
        samples.append((np.geomspace(200, 130, 20), 3))  # falling low
        samples.append((np.geomspace(260, 140, 20), 4))  # falling from high
    model = fit_tone_model(samples)
    assert set(model.available_tones()) == {1, 2, 3, 4}
    assert model.prototypes[2].contour[-1] > model.prototypes[2].contour[0]
    assert model.prototypes[1].contour[-1] == pytest.approx(
        model.prototypes[1].contour[0], abs=1.0
    )


def test_canonical_model_is_available_without_calibration():
    model = canonical_model(F0Range(90.0, 220.0))
    assert set(model.available_tones()) == {1, 2, 3, 4}
    decision = model.classify(_contour(np.linspace(5, 5, 20)))
    assert decision.tone == 1, "a flat high contour is 阴平"


def test_classify_hz_projects_with_the_learner_range():
    samples = []
    for _ in range(4):
        samples.append((np.geomspace(200, 200, 20), 1))
        samples.append((np.geomspace(150, 280, 20), 2))
        samples.append((np.geomspace(180, 120, 20), 3))
        samples.append((np.geomspace(300, 140, 20), 4))
    model = fit_tone_model(samples)

    # A rising contour in a *different* speaker's Hz range should still be T2.
    learner_range = F0Range(220.0, 420.0)
    decision = model.classify_hz(np.geomspace(250, 400, 25), learner_range)
    assert decision.tone == 2
    assert decision.range_source == "calibration"


def test_classify_hz_reports_tone_zero_when_there_is_no_voicing():
    model = canonical_model(F0Range(90.0, 220.0))
    decision = model.classify_hz(np.array([np.nan, np.nan]))
    assert decision.tone == 0
    assert decision.notes


# ------------------------------------------------------------- metrics
def _record(**overrides) -> Record:
    base = dict(
        item_id="x", scorer="s", mode="m", family="tone", contrast="ma",
        truth_correct=True, truth_dim=None, truth_label=None, truth_tone=1,
        pred_correct=True, pred_dim=None, pred_unit=None, pred_tone=1,
        confidence=0.9, latency_s=0.1,
    )
    base.update(overrides)
    return Record(**base)


def test_detection_confusion_and_error_rates():
    records = [
        _record(item_id="tn"),
        _record(item_id="fp", truth_correct=True, pred_correct=False),
        _record(item_id="fn", truth_correct=False, pred_correct=True),
        _record(item_id="tp", truth_correct=False, pred_correct=False),
    ]
    det = detection(records)
    assert (det["tp"], det["tn"], det["fp"], det["fn"]) == (1, 1, 1, 1)
    assert det["accuracy"] == pytest.approx(0.5)
    assert det["false_accept_rate"] == pytest.approx(0.5)
    assert det["false_reject_rate"] == pytest.approx(0.5)


def test_detection_ignores_failed_calls():
    records = [_record(item_id="ok"), _record(item_id="bad", error="boom", pred_correct=None)]
    det = detection(records)
    assert det["n"] == 1


def test_tone_identification_counts_only_known_truth():
    records = [
        _record(item_id="a", truth_tone=3, pred_tone=3),
        _record(item_id="b", truth_tone=3, pred_tone=4),
        _record(item_id="c", truth_tone=0, pred_tone=2),  # unlabelled, excluded
    ]
    tone = tone_identification(records)
    assert tone["n"] == 2
    assert tone["accuracy"] == pytest.approx(0.5)
    assert tone["confusion"]["T3_as_T4"] == 1


def test_localisation_scores_only_flagged_errors():
    records = [
        _record(item_id="a", truth_correct=False, truth_dim="tone",
                pred_correct=False, pred_dim="tone"),
        _record(item_id="b", truth_correct=False, truth_dim="initial",
                pred_correct=False, pred_dim="tone"),
        _record(item_id="c", truth_correct=False, truth_dim="tone",
                pred_correct=True),  # not flagged, excluded
    ]
    loc = localisation(records)
    assert loc["n"] == 2
    assert loc["accuracy"] == pytest.approx(0.5)
    assert loc["per_dimension"]["initial"]["accuracy"] == 0.0


def test_trap_analysis_separates_substituted_from_control():
    records = [
        _record(item_id="s1", family="trap", truth_correct=False, pred_correct=True),
        _record(item_id="s2", family="trap", truth_correct=False, pred_correct=False),
        _record(item_id="c1", family="trap", truth_correct=True, pred_correct=True),
    ]
    trap = trap_analysis(records)
    assert trap["n_substituted"] == 2
    assert trap["false_accept_rate"] == pytest.approx(0.5)
    assert trap["false_reject_rate"] == pytest.approx(0.0)


def test_consistency_measures_agreement_across_runs():
    run_a = [_record(item_id="a"), _record(item_id="b", pred_tone=2)]
    run_b = [_record(item_id="a"), _record(item_id="b", pred_tone=3)]
    info = consistency([run_a, run_b])
    assert info["n_runs"] == 2
    assert info["n_items"] == 2
    assert info["verdict_agreement"] == pytest.approx(1.0)
    assert info["tone_agreement"] == pytest.approx(0.5)


def test_scorecard_has_every_section():
    records = [_record(item_id=str(i)) for i in range(4)]
    card = scorecard(records)
    for key in ("detection", "tone_identification", "localisation",
                "by_family", "by_contrast", "trap", "latency"):
        assert key in card
