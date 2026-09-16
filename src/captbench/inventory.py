"""Stimulus inventory: what we synthesise, and the ground truth for each clip.

The benchmark works by *controlled substitution*. For every minimal pair we
synthesise both members with the same voice, then label one of them `expected`
and, when they differ, the other one as the error the learner committed:

    expected=妈 (mā)  spoken=妈  -> is_correct=True
    expected=妈 (mā)  spoken=麻  -> is_correct=False, error_dim=tone  T1->T2

Because the substitution is categorical and known by construction, accuracy can
be measured exactly. This is the strength of a synthetic benchmark and also its
main limitation: TTS substitutions are clean, whereas real learner errors are
graded - a learner who half-achieves a rising tone is neither correct nor a
clean substitution.

Nothing here bridges that gap. Generating graded errors would mean warping a
correct clip's F0 contour part-way toward another tone (Praat's Manipulation
can do it), which would give a correctness threshold to sweep instead of a
binary label. It is not implemented; every item in this inventory is either
wholly correct or a categorical substitution.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from .g2p import analyze, minimal_pair_diff


@dataclass(frozen=True)
class Stimulus:
    id: str
    family: str  # "tone" | "initial" | "final" | "trap" | "calibration"
    contrast: str  # minimal-pair group, e.g. "ma" or "zh_z"
    expected_text: str  # what the learner was supposed to say
    spoken_text: str  # what the audio actually says
    is_correct: bool
    error_dim: str | None  # "tone" | "initial" | "final" | "word"
    error_label: str | None  # human-readable, e.g. "T3->T2"
    expected_unit: str | None  # "T3" | "zh" | "v"
    spoken_unit: str | None  # "T2" | "z" | "i"
    contrast_units: tuple[str, ...] = ()  # every member of this contrast
    focus_index: int = 0  # which syllable carries the contrast
    focus_span: int = 1
    context: str | None = None  # communicative scenario, for trap items
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Tone contrasts: one syllable realised in all four tones.
# --------------------------------------------------------------------------
#: (contrast id, characters indexed by tone 1..4)
TONE_SETS: list[tuple[str, list[str]]] = [
    ("ma", ["妈", "麻", "马", "骂"]),
    ("shi", ["诗", "时", "使", "是"]),
    ("bao", ["包", "薄", "饱", "抱"]),
    ("cai", ["猜", "才", "采", "菜"]),
]

#: Extra tone realisations that are synthesised purely to fit the speaker's
#: tone prototypes. They are never scored, so the tone model can be calibrated
#: on syllables the test items do not use.
CALIBRATION_TONE_SETS: list[tuple[str, list[str]]] = [
    ("tang", ["汤", "糖", "躺", "烫"]),
    ("mao", ["猫", "毛", "卯", "冒"]),
]

#: Calibration sentences, read by both roles. They exist because connected
#: speech has a wider pitch range than isolated words - an enrolment drawn only
#: from monosyllables under-estimates the speaker's range and saturates the
#: five-level transform on sentences. Their tones (after sandhi) also give the
#: sentence-context tone prototypes something to be fitted on.
CALIBRATION_SENTENCES: list[str] = [
    "今天天气非常好",
    "我不知道你在说什么",
]

# --------------------------------------------------------------------------
# Segmental contrasts: minimal pairs differing in exactly one phone.
# --------------------------------------------------------------------------
#: (contrast id, char A, char B, expected dim, label A->B)
SEGMENTAL_PAIRS: list[tuple[str, str, str, str, str]] = [
    # Retroflex vs dental sibilants - the classic 平翘舌 confusion.
    ("zh_z", "知", "资", "initial", "zh->z"),
    ("ch_c", "插", "擦", "initial", "ch->c"),
    ("sh_s", "诗", "丝", "initial", "sh->s"),
    ("sh_s2", "书", "苏", "initial", "sh->s"),
    # r/l and n/l.
    ("r_l", "热", "乐", "initial", "r->l"),
    ("n_l", "脑", "老", "initial", "n->l"),
    # Aspiration: the Mandarin contrast is VOT, not voicing.
    ("b_p", "八", "趴", "initial", "b->p (aspiration)"),
    ("d_t", "大", "踏", "initial", "d->t (aspiration)"),
    ("g_k", "哥", "科", "initial", "g->k (aspiration)"),
    ("j_q", "鸡", "七", "initial", "j->q (aspiration)"),
    # Nasal codas.
    ("an_ang", "班", "帮", "final", "an->ang"),
    ("en_eng", "陈", "成", "final", "en->eng"),
    ("in_ing", "新", "星", "final", "in->ing"),
    # Rounding: ü vs i (the F3 test) and uo vs e (openness/rounding).
    ("v_i", "女", "你", "final", "ü->i (rounding)"),
    ("v_i2", "绿", "力", "final", "ü->i (rounding)"),
    ("uo_e", "阔", "客", "final", "uo->e"),
]


# --------------------------------------------------------------------------
# Linguistic-trap sentences: the context predicts one word, the audio says
# its minimal-pair partner. A scorer that trusts the expected text (or the
# semantics) over the acoustics will accept them.
# --------------------------------------------------------------------------
TRAPS: list[dict] = [
    {
        "contrast": "shuijiao",
        "expected": "我想吃水饺",
        "spoken": "我想吃睡觉",
        "focus_index": 3,
        "focus_span": 2,
        "label": "T3->T4 (水饺→睡觉)",
        "unit_expected": "T3",
        "unit_spoken": "T4",
        "context": "在饭馆点菜 — a learner ordering food at a restaurant",
    },
    {
        "contrast": "maimai",
        "expected": "我要买这本书",
        "spoken": "我要卖这本书",
        "focus_index": 2,
        "label": "T3->T4 (买→卖)",
        "unit_expected": "T3",
        "unit_spoken": "T4",
        "context": "在书店买书 — a learner buying a book in a bookstore",
    },
    {
        "contrast": "kankan",
        "expected": "他在图书馆看书",
        "spoken": "他在图书馆砍书",
        "focus_index": 5,
        "label": "T4->T3 (看→砍)",
        "unit_expected": "T4",
        "unit_spoken": "T3",
        "context": "描述某人读书 — describing someone reading in a library",
    },
    {
        "contrast": "guanshang",
        "expected": "请把门关上",
        "spoken": "请把门灌上",
        "focus_index": 3,
        "label": "T1->T4 (关→灌)",
        "unit_expected": "T1",
        "unit_spoken": "T4",
        "context": "请人关门 — asking someone to close the door",
    },
]


def _tone_unit(tone: int) -> str:
    return f"T{tone}"


def tone_items() -> list[Stimulus]:
    """All 4x3 tone substitutions for each tone set, plus the 4 correct ones."""
    items: list[Stimulus] = []
    for contrast, chars in TONE_SETS:
        for expected_tone in (1, 2, 3, 4):
            for spoken_tone in (1, 2, 3, 4):
                expected_char = chars[expected_tone - 1]
                spoken_char = chars[spoken_tone - 1]
                correct = expected_tone == spoken_tone
                suffix = "correct" if correct else f"T{expected_tone}_as_T{spoken_tone}"
                items.append(
                    Stimulus(
                        id=f"tone__{contrast}__{suffix}",
                        family="tone",
                        contrast=contrast,
                        expected_text=expected_char,
                        spoken_text=spoken_char,
                        is_correct=correct,
                        error_dim=None if correct else "tone",
                        error_label=None if correct else f"T{expected_tone}->T{spoken_tone}",
                        expected_unit=_tone_unit(expected_tone),
                        spoken_unit=_tone_unit(spoken_tone),
                        contrast_units=("T1", "T2", "T3", "T4"),
                    )
                )
    return items


def calibration_items() -> list[Stimulus]:
    """Productions used only to fit the speaker's tone prototypes.

    Neither the words nor the sentences here are ever scored.
    """
    items: list[Stimulus] = []
    for contrast, chars in CALIBRATION_TONE_SETS:
        for tone, char in enumerate(chars, start=1):
            items.append(
                Stimulus(
                    id=f"calibration__{contrast}__T{tone}",
                    family="calibration",
                    contrast=contrast,
                    expected_text=char,
                    spoken_text=char,
                    is_correct=True,
                    error_dim=None,
                    error_label=None,
                    expected_unit=_tone_unit(tone),
                    spoken_unit=_tone_unit(tone),
                    contrast_units=("T1", "T2", "T3", "T4"),
                    notes="calibration only - never scored",
                )
            )
    for i, sentence in enumerate(CALIBRATION_SENTENCES, start=1):
        items.append(
            Stimulus(
                id=f"calibration__sentence{i}",
                family="calibration",
                contrast="calib_sentence",
                expected_text=sentence,
                spoken_text=sentence,
                is_correct=True,
                error_dim=None,
                error_label=None,
                # No single unit: these carry several tones, one per syllable.
                expected_unit=None,
                spoken_unit=None,
                notes="sentence calibration only - never scored",
            )
        )
    return items


def segmental_items() -> list[Stimulus]:
    """Both correct readings of each minimal pair, plus both substitutions."""
    items: list[Stimulus] = []
    for contrast, char_a, char_b, dim, label_ab in SEGMENTAL_PAIRS:
        syl_a, syl_b = analyze(char_a)[0], analyze(char_b)[0]
        assert minimal_pair_diff(syl_a, syl_b) == dim, (
            f"{char_a}/{char_b} differ by {minimal_pair_diff(syl_a, syl_b)}, "
            f"declared as {dim}"
        )
        unit_a = syl_a.initial if dim == "initial" else syl_a.final
        unit_b = syl_b.initial if dim == "initial" else syl_b.final
        label_ba = label_ab.split(" (")[0].split("->")[1] + "->" + label_ab.split("->")[0]
        if " (" in label_ab:
            label_ba += " (" + label_ab.split(" (")[1]

        for expected_char, spoken_char, expected_unit, spoken_unit, label in (
            (char_a, char_a, unit_a, unit_a, None),
            (char_b, char_b, unit_b, unit_b, None),
            (char_a, char_b, unit_a, unit_b, label_ab),
            (char_b, char_a, unit_b, unit_a, label_ba),
        ):
            correct = spoken_char == expected_char
            slug = f"{syl_a.pinyin}_as_{syl_b.pinyin}" if not correct else syl_a.pinyin
            if expected_char == char_b and not correct:
                slug = f"{syl_b.pinyin}_as_{syl_a.pinyin}"
            if correct:
                slug = analyze(expected_char)[0].pinyin
            items.append(
                Stimulus(
                    id=f"{dim}__{contrast}__{slug}",
                    family=dim,
                    contrast=contrast,
                    expected_text=expected_char,
                    spoken_text=spoken_char,
                    is_correct=correct,
                    error_dim=None if correct else dim,
                    error_label=label,
                    expected_unit=expected_unit,
                    spoken_unit=spoken_unit,
                    contrast_units=(unit_a, unit_b),
                )
            )
    return items


def trap_items() -> list[Stimulus]:
    """Correct and substituted readings of each trap sentence."""
    items: list[Stimulus] = []
    for trap in TRAPS:
        for correct in (True, False):
            text = trap["expected"] if correct else trap["spoken"]
            items.append(
                Stimulus(
                    id=f"trap__{trap['contrast']}__{'correct' if correct else 'substituted'}",
                    family="trap",
                    contrast=trap["contrast"],
                    expected_text=trap["expected"],
                    spoken_text=text,
                    is_correct=correct,
                    error_dim=None if correct else "word",
                    error_label=None if correct else trap["label"],
                    expected_unit=trap["unit_expected"],
                    spoken_unit=trap["unit_spoken"] if not correct else trap["unit_expected"],
                    focus_index=trap["focus_index"],
                    focus_span=trap.get("focus_span", 1),
                    context=trap["context"],
                )
            )
    return items


def build_inventory(
    families: Iterable[str] = ("tone", "initial", "final", "trap"),
    include_calibration: bool = True,
) -> list[Stimulus]:
    """Build the full stimulus list for the requested families.

    `calibration` items are always added when requested: they are synthesised
    and used to fit the tone model, but never scored.
    """
    families = set(families)
    items: list[Stimulus] = []
    if "tone" in families:
        items.extend(tone_items())
    if "initial" in families or "final" in families:
        items.extend(
            s for s in segmental_items() if s.family in families or s.error_dim in families
        )
    if "trap" in families:
        items.extend(trap_items())
    if include_calibration:
        items.extend(calibration_items())
    seen: set[str] = set()
    unique: list[Stimulus] = []
    for item in items:
        if item.id in seen:
            continue
        seen.add(item.id)
        unique.append(item)
    return unique
