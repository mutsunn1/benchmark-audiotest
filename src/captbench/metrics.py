"""Metrics: turning two scorers' raw judgements into comparable numbers.

Everything is reduced to a common `Record` first, so neither scorer gets
special treatment downstream. The measures that matter for this benchmark:

* **detection** - can the system tell a correct production from a wrong one?
  Reported as accuracy plus the two error rates that matter in teaching:
  false accept (a wrong pronunciation passed as correct - the failure mode the
  "linguistic trap" predicts) and false reject.
* **tone identification** - over items where the audio genuinely contains a
  known tone, how often is that tone named correctly? This is the fairest
  head-to-head, because both engines produce a tone estimate for the same audio.
* **localisation** - when a system does flag an error, does it name the right
  dimension (tone vs initial vs final)?
* **consistency** - the same clip scored repeatedly: do the answers agree?
  The traditional engine is deterministic; a sampled omni model is not.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Record:
    item_id: str
    scorer: str
    mode: str
    family: str
    contrast: str
    truth_correct: bool
    truth_dim: str | None
    truth_label: str | None
    truth_tone: int
    pred_correct: bool | None
    pred_dim: str | None
    pred_unit: str | None
    pred_tone: int
    confidence: float | None
    latency_s: float
    error: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.error is None and self.pred_correct is not None


def _tone_of_unit(unit: str | None) -> int:
    if unit and unit.startswith("T") and unit[1:].isdigit():
        return int(unit[1:])
    return 0


def record_from_traditional(result: dict, item: dict, mode: str = "scripted") -> Record:
    """Extract the comparable fields from a traditional-scorer judgement."""
    focus = item.get("focus_index", 0)
    span = item.get("focus_span", 1)
    syllables = result.get("syllables") or []
    focus_syllables = [
        s for s in syllables if focus <= s.get("index", -1) < focus + span
    ] or syllables

    pred_tone = 0
    confidence = None
    if focus_syllables:
        # Prefer the syllable with the largest tone margin inside the focus span.
        best = max(
            focus_syllables,
            key=lambda s: abs(((s.get("tone") or {}).get("margin") or 0.0)),
        )
        tone = best.get("tone") or {}
        pred_tone = int(tone.get("observed_tone") or 0)
        confidence = result.get("confidence")

    return Record(
        item_id=result["item_id"],
        scorer=result.get("scorer", "traditional"),
        mode=mode,
        family=item.get("family", ""),
        contrast=item.get("contrast", ""),
        truth_correct=bool(item.get("is_correct")),
        truth_dim=item.get("error_dim"),
        truth_label=item.get("error_label"),
        truth_tone=_tone_of_unit(item.get("spoken_unit")),
        pred_correct=result.get("predicted_correct"),
        pred_dim=result.get("predicted_dim"),
        pred_unit=result.get("predicted_unit"),
        pred_tone=pred_tone,
        confidence=confidence,
        latency_s=float(result.get("latency_s") or 0.0),
        error=result.get("error"),
        raw=result,
    )


def record_from_omni(result: dict, item: dict, mode: str) -> Record:
    """Extract the comparable fields from an omni response."""
    verdict = result.get("verdict") or {}
    parsed = result.get("parsed") or {}
    heard = result.get("heard_tone") or 0
    if not heard:
        heard = verdict.get("pred_tone") or 0
    return Record(
        item_id=result["item_id"],
        scorer=result.get("scorer", "omni"),
        mode=mode,
        family=item.get("family", ""),
        contrast=item.get("contrast", ""),
        truth_correct=bool(item.get("is_correct")),
        truth_dim=item.get("error_dim"),
        truth_label=item.get("error_label"),
        truth_tone=_tone_of_unit(item.get("spoken_unit")),
        pred_correct=verdict.get("predicted_correct"),
        pred_dim=verdict.get("predicted_dim"),
        pred_unit=verdict.get("predicted_unit") or result.get("pred_unit"),
        pred_tone=int(heard or 0),
        confidence=parsed.get("confidence"),
        latency_s=float(result.get("latency_s") or 0.0),
        error=result.get("error"),
        raw=result,
    )


def to_records(results: list[dict], items: dict[str, dict], scorer: str) -> list[Record]:
    records: list[Record] = []
    for result in results:
        item = items.get(result["item_id"])
        if item is None:
            continue
        if scorer == "traditional":
            records.append(record_from_traditional(result, item))
        else:
            records.append(record_from_omni(result, item, result.get("mode", "scripted")))
    return records


# --------------------------------------------------------------------------
# Individual measures
# --------------------------------------------------------------------------
def detection(records: list[Record]) -> dict:
    """Correct-vs-error detection quality, including both error rates."""
    usable = [r for r in records if r.usable]
    if not usable:
        return {"n": 0}

    tp = sum(1 for r in usable if not r.truth_correct and not r.pred_correct)
    tn = sum(1 for r in usable if r.truth_correct and r.pred_correct)
    fp = sum(1 for r in usable if r.truth_correct and not r.pred_correct)
    fn = sum(1 for r in usable if not r.truth_correct and r.pred_correct)

    n_err = tp + fn
    n_ok = tn + fp
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / n_err if n_err else float("nan")
    specificity = tn / n_ok if n_ok else float("nan")
    return {
        "n": len(usable),
        "n_correct_items": n_ok,
        "n_error_items": n_err,
        "accuracy": (tp + tn) / len(usable),
        "balanced_accuracy": float(np.nanmean([recall, specificity])),
        "false_accept_rate": (fn / n_err) if n_err else float("nan"),
        "false_reject_rate": (fp / n_ok) if n_ok else float("nan"),
        "precision_error": precision,
        "recall_error": recall,
        "specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def tone_identification(records: list[Record]) -> dict:
    """Over items whose audio genuinely contains a tone, how often is it right?"""
    usable = [r for r in records if r.usable and r.truth_tone and r.pred_tone]
    if not usable:
        return {"n": 0}
    hits = sum(1 for r in usable if r.pred_tone == r.truth_tone)
    return {
        "n": len(usable),
        "accuracy": hits / len(usable),
        "hits": hits,
        "confusion": {
            f"T{t}_as_T{p}": c
            for (t, p), c in sorted(Counter(
                (r.truth_tone, r.pred_tone) for r in usable
            ).items())
        },
    }


def localisation(records: list[Record]) -> dict:
    """Of the errors a system did flag, how many name the right dimension?"""
    flagged = [
        r for r in records if r.usable and not r.pred_correct and r.truth_dim
    ]
    if not flagged:
        return {"n": 0}
    right = sum(1 for r in flagged if r.pred_dim == r.truth_dim)
    per_dim: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "right": 0})
    for r in flagged:
        per_dim[r.truth_dim]["n"] += 1
        if r.pred_dim == r.truth_dim:
            per_dim[r.truth_dim]["right"] += 1
    return {
        "n": len(flagged),
        "accuracy": right / len(flagged),
        "per_dimension": {
            dim: {**vals, "accuracy": vals["right"] / vals["n"]}
            for dim, vals in sorted(per_dim.items())
        },
    }


def by_family(records: list[Record]) -> dict:
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        groups[r.family].append(r)
    return {
        fam: {
            "n": len(recs),
            "accuracy": _accuracy(recs),
            "false_accept_rate": detection(recs).get("false_accept_rate"),
        }
        for fam, recs in sorted(groups.items())
    }


def by_contrast(records: list[Record]) -> dict:
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        groups[r.contrast].append(r)
    out = {}
    for contrast, recs in sorted(groups.items()):
        det = detection(recs)
        out[contrast] = {
            "n": len(recs),
            "accuracy": det.get("accuracy"),
            "false_accept_rate": det.get("false_accept_rate"),
        }
    return out


def _accuracy(records: list[Record]) -> float | None:
    usable = [r for r in records if r.usable]
    if not usable:
        return None
    return sum(1 for r in usable if r.pred_correct == r.truth_correct) / len(usable)


def trap_analysis(records: list[Record]) -> dict:
    """The linguistic-trap subset: did context or the expected text win?"""
    traps = [r for r in records if r.family == "trap" and r.usable]
    if not traps:
        return {"n": 0}
    errors = [r for r in traps if not r.truth_correct]
    controls = [r for r in traps if r.truth_correct]
    return {
        "n": len(traps),
        "n_substituted": len(errors),
        "n_control": len(controls),
        "false_accept_rate": (
            sum(1 for r in errors if r.pred_correct) / len(errors) if errors else None
        ),
        "false_reject_rate": (
            sum(1 for r in controls if not r.pred_correct) / len(controls)
            if controls
            else None
        ),
        "tone_accuracy": tone_identification(traps).get("accuracy"),
    }


def latency(records: list[Record]) -> dict:
    values = np.array([r.latency_s for r in records if r.latency_s > 0])
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean_s": float(values.mean()),
        "median_s": float(np.median(values)),
        "p95_s": float(np.percentile(values, 95)),
        "max_s": float(values.max()),
    }


def consistency(runs: list[list[Record]]) -> dict:
    """Agreement across repeated scorings of the same items.

    `runs` is a list of per-run record lists covering the same item ids.
    """
    if len(runs) < 2:
        return {"n_runs": len(runs), "note": "need at least two runs"}
    by_item: dict[str, list[Record]] = defaultdict(list)
    for run in runs:
        for r in run:
            by_item[r.item_id].append(r)

    complete = {k: v for k, v in by_item.items() if len(v) == len(runs)}
    if not complete:
        return {"n_runs": len(runs), "n_items": 0}

    verdict_agree = 0
    tone_agree = 0
    tone_eligible = 0
    for records in complete.values():
        verdicts = {(r.pred_correct, r.pred_dim) for r in records}
        if len(verdicts) == 1:
            verdict_agree += 1
        tones = {r.pred_tone for r in records if r.pred_tone}
        if len(tones) >= 1:
            tone_eligible += 1
            if len(tones) == 1:
                tone_agree += 1
    n = len(complete)
    return {
        "n_runs": len(runs),
        "n_items": n,
        "verdict_agreement": verdict_agree / n,
        "tone_agreement": (tone_agree / tone_eligible) if tone_eligible else None,
    }


def scorecard(records: list[Record]) -> dict:
    """Everything above, for one scorer + mode."""
    return {
        "n_records": len(records),
        "detection": detection(records),
        "tone_identification": tone_identification(records),
        "localisation": localisation(records),
        "by_family": by_family(records),
        "by_contrast": by_contrast(records),
        "trap": trap_analysis(records),
        "latency": latency(records),
    }


def compare(cards: dict[str, dict]) -> list[dict]:
    """A flat table of the headline numbers, one row per scorer+mode."""
    rows = []
    for name, card in sorted(cards.items()):
        det = card.get("detection", {})
        tone = card.get("tone_identification", {})
        rows.append(
            {
                "system": name,
                "n": det.get("n", 0),
                "detection_accuracy": det.get("accuracy"),
                "balanced_accuracy": det.get("balanced_accuracy"),
                "false_accept_rate": det.get("false_accept_rate"),
                "false_reject_rate": det.get("false_reject_rate"),
                "tone_accuracy": tone.get("accuracy"),
                "tone_n": tone.get("n", 0),
                "localisation_accuracy": card.get("localisation", {}).get("accuracy"),
                "trap_far": card.get("trap", {}).get("false_accept_rate"),
                "median_latency_s": card.get("latency", {}).get("median_s"),
            }
        )
    return rows
