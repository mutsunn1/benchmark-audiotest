"""Mandarin G2P: syllable decomposition, tone sandhi, canonical tone templates.

pypinyin's raw output needs normalising before it is phonologically meaningful:

* `y` / `w` are reported as initials but are glides, not consonants
  (一 = `y` + `i` is really /i/; 我 = `w` + `o` is really /uo/).
* `ü` is written `v`, and `j/q/x/y` + `u` is always really `ü`
  (居 `ju` is /tɕy/).
* Abbreviated finals `ui` / `iu` / `un` are really `uei` / `iou` / `uen`.

`normalise_final` undoes all of that so that two syllables differing only in
the property we care about (rounding, nasality) compare equal elsewhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from pypinyin import Style
from pypinyin import pinyin as _pypinyin

#: Consonants that can occupy the initial slot (zero-initial syllables have "").
INITIALS = (
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s",
)

#: Every legal Mandarin final, in the normalised spelling `normalize_final`
#: produces. Used to reject model output that is not a pinyin syllable at all:
#: without it, "gibberish" parses as the syllable `g` + `ibberish` and gets
#: scored as a pronunciation error rather than as an unusable reply.
MANDARIN_FINALS = frozenset({
    "a", "o", "e", "i", "u", "v", "er", "io",
    "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong",
    "ia", "ie", "iao", "iou", "ian", "in", "iang", "ing", "iong",
    "ua", "uo", "uai", "uei", "uan", "uen", "uang", "ueng",
    "ve", "van", "vn",
})

#: Tones 1-4 plus 5 for the neutral tone.
TONE_NUMBERS = (1, 2, 3, 4, 5)

_TONE_MARK_VOWELS = {
    "a": "āáǎàa",
    "o": "ōóǒòo",
    "e": "ēéěèe",
    "i": "īíǐìi",
    "u": "ūúǔùu",
    "v": "ǖǘǚǜü",
}

#: Canonical tone contours in Shi Feng five-level T-value space (0-5).
#: Each tone maps to one or more acceptable variants; DTW matches against all
#: of them and keeps the best. The 3rd tone carries a 21 "half third" variant
#: because the full 214 dip only surfaces in isolation or before a pause.
TONE_TEMPLATES: dict[int, list[list[tuple[float, float]]]] = {
    1: [[(0.0, 5.0), (1.0, 5.0)]],                              # 阴平 55
    2: [[(0.0, 3.0), (0.18, 2.8), (1.0, 5.0)]],                 # 阳平 35
    3: [
        [(0.0, 2.2), (0.35, 0.7), (0.72, 1.1), (1.0, 4.0)],     # 上声 214
        [(0.0, 2.2), (0.5, 0.6), (1.0, 1.2)],                   # 半上 21
    ],
    4: [[(0.0, 5.0), (1.0, 0.6)]],                              # 去声 51
    5: [[(0.0, 3.0), (1.0, 2.2)]],                              # 轻声
}

TONE_NAMES_ZH = {1: "阴平", 2: "阳平", 3: "上声", 4: "去声", 5: "轻声"}


@dataclass(frozen=True)
class Syllable:
    """One syllable of running Mandarin text."""

    char: str
    pinyin: str      # pypinyin TONE3 form, e.g. "zhang1", "nv3"
    initial: str     # normalised: "" for zero-initial, "zh", "b", ...
    final: str       # normalised: "v" = ü, full finals (uei not ui)
    tone: int        # 1-4, or 5 for neutral
    surface_final: str  # what pypinyin reported, before normalisation
    surface_initial: str

    @property
    def display(self) -> str:
        """Syllable with a tone diacritic: `ma3` -> `mǎ`."""
        return tone_mark(self.pinyin)

    @property
    def tone_name(self) -> str:
        return TONE_NAMES_ZH.get(self.tone, "?")

    def with_tone(self, tone: int) -> "Syllable":
        base = self.pinyin[:-1] if self.pinyin[-1].isdigit() else self.pinyin
        return Syllable(
            char=self.char,
            pinyin=f"{base}{tone}",
            initial=self.initial,
            final=self.final,
            tone=tone,
            surface_final=self.surface_final,
            surface_initial=self.surface_initial,
        )


def tone_mark(pinyin_tone3: str) -> str:
    """`zhang1` -> `zhāng`, `nv3` -> `nǚ`, `ma5` -> `ma`."""
    if not pinyin_tone3 or not pinyin_tone3[-1].isdigit():
        return pinyin_tone3
    body, tone = pinyin_tone3[:-1], int(pinyin_tone3[-1])
    if tone == 5:
        return body.replace("v", "ü")
    # The tone mark goes on the first of a, o, e; else the last of i, u, ü.
    for vowel in ("a", "o", "e"):
        if vowel in body:
            idx = body.index(vowel)
            return body[:idx] + _TONE_MARK_VOWELS[vowel][tone - 1] + body[idx + 1 :].replace("v", "ü")
    for idx in range(len(body) - 1, -1, -1):
        if body[idx] in _TONE_MARK_VOWELS:
            v = body[idx]
            return body[:idx] + _TONE_MARK_VOWELS[v][tone - 1] + body[idx + 1 :].replace("v", "ü")
    return body.replace("v", "ü")


def normalize_final(surface_initial: str, surface_final: str) -> str:
    """Turn pypinyin's surface final into a phonological final (ü -> `v`)."""
    initial, final = surface_initial, surface_final

    if initial == "y":
        if final.startswith("i"):
            pass                                  # yi -> i, yin -> in, ying -> ing
        elif final.startswith("u"):
            final = "v" + final[1:]               # yu -> ü, yue -> üe, yun -> ün
        elif final == "ong":
            final = "iong"                        # yong
        else:
            final = "i" + final                   # ya -> ia, ye -> ie, you -> iou
    elif initial == "w":
        final = "u" if final == "u" else "u" + final   # wu -> u, wo -> uo, wen -> uen

    if initial in ("j", "q", "x") and final.startswith("u"):
        final = "v" + final[1:]                   # ju -> jü, jue -> jüe, jun -> jün

    # Expand abbreviated finals (never after j/q/x/y, where u was really ü).
    final = {"ui": "uei", "iu": "iou", "un": "uen"}.get(final, final)
    return final


def normalize_initial(surface_initial: str) -> str:
    """`y` / `w` are glides, not initials."""
    return "" if surface_initial in ("y", "w") else surface_initial


@lru_cache(maxsize=4096)
def _lookup(char: str) -> tuple[str, str, str]:
    surface_initial = _pypinyin(char, style=Style.INITIALS, strict=False)[0][0]
    surface_final = _pypinyin(char, style=Style.FINALS, strict=False)[0][0]
    tone3 = _pypinyin(char, style=Style.TONE3, neutral_tone_with_five=True)[0][0]
    return surface_initial, surface_final, tone3


def analyze_char(char: str) -> Syllable | None:
    """Decompose a single Han character. Returns None for non-Han input."""
    surface_initial, surface_final, tone3 = _lookup(char)
    if not tone3 or not tone3[-1].isdigit():
        return None
    return Syllable(
        char=char,
        pinyin=tone3,
        initial=normalize_initial(surface_initial),
        final=normalize_final(surface_initial, surface_final),
        tone=int(tone3[-1]),
        surface_final=surface_final,
        surface_initial=surface_initial,
    )


def analyze(text: str, sandhi: bool = False) -> list[Syllable]:
    """Decompose running text into syllables, optionally applying tone sandhi.

    Runs pypinyin over the whole string so its phrase dictionary resolves
    heteronyms correctly (睡觉 is shuì jiào, not shuì jué).
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return []
    joined = "".join(chars)
    inis = _pypinyin(joined, style=Style.INITIALS, strict=False)
    fins = _pypinyin(joined, style=Style.FINALS, strict=False)
    t3s = _pypinyin(joined, style=Style.TONE3, neutral_tone_with_five=True)

    if not (len(inis) == len(fins) == len(t3s) == len(chars)):
        # Segmentation mismatch (should not happen) - fall back to per-char.
        syllables = [s for s in (analyze_char(c) for c in chars) if s]
        return apply_tone_sandhi(syllables) if sandhi else syllables

    syllables = []
    for char, (surface_initial,), (surface_final,), (tone3,) in zip(chars, inis, fins, t3s):
        if not tone3 or not tone3[-1].isdigit():
            continue
        syllables.append(
            Syllable(
                char=char,
                pinyin=tone3,
                initial=normalize_initial(surface_initial),
                final=normalize_final(surface_initial, surface_final),
                tone=int(tone3[-1]),
                surface_final=surface_final,
                surface_initial=surface_initial,
            )
        )
    return apply_tone_sandhi(syllables) if sandhi else syllables


def apply_tone_sandhi(syllables: list[Syllable]) -> list[Syllable]:
    """Apply the three sandhi rules a learner is expected to produce.

    * 3rd + 3rd -> 2nd + 3rd (你好 ní hǎo)
    * 不 bù -> bú before a 4th tone
    * 一 yī -> yì before 1/2/3, yí before 4
    """
    out = list(syllables)
    for i, syl in enumerate(out):
        nxt = out[i + 1] if i + 1 < len(out) else None
        if syl.char == "不" and nxt is not None and nxt.tone == 4:
            out[i] = syl.with_tone(2)
        elif syl.char == "一" and nxt is not None:
            out[i] = syl.with_tone(2 if nxt.tone == 4 else 4)
        elif syl.tone == 3 and nxt is not None and nxt.tone == 3:
            out[i] = syl.with_tone(2)
    return out


def tones_of(text: str, sandhi: bool = False) -> list[int]:
    return [s.tone for s in analyze(text, sandhi=sandhi)]


def to_pinyin(text: str, sandhi: bool = False, tone_marks: bool = True) -> str:
    syllables = analyze(text, sandhi=sandhi)
    return " ".join(s.display if tone_marks else s.pinyin for s in syllables)


def minimal_pair_diff(a: Syllable, b: Syllable) -> str:
    """Name the single dimension on which two syllables differ.

    Returns one of `initial`, `final`, `tone`, `none`, or `multiple`.
    """
    return diff_dimension(
        (a.initial, a.final, a.tone), (b.initial, b.final, b.tone)
    )


def diff_dimension(
    a: tuple[str, str, int], b: tuple[str, str, int]
) -> str:
    """Compare two (initial, final, tone) triples."""
    diffs = []
    if a[0] != b[0]:
        diffs.append("initial")
    if a[1] != b[1]:
        diffs.append("final")
    if a[2] != b[2]:
        diffs.append("tone")
    if not diffs:
        return "none"
    return diffs[0] if len(diffs) == 1 else "multiple"


_INITIALS_BY_LENGTH = tuple(sorted(INITIALS, key=len, reverse=True))


def split_pinyin(value: str) -> tuple[str, str, int] | None:
    """Parse a pinyin syllable into (initial, final, tone).

    Accepts the spellings a model is likely to emit: `ma3`, `zhang1`, `nv3`,
    `nü3`, `lu:4`, `ma`, with or without spaces. Returns None if nothing
    parseable is found. The output uses the same conventions as `analyze`, so
    a model's transcription and a reference syllable compare directly.
    """
    if not value:
        return None
    text = value.strip().lower().replace("u:", "v").replace("ü", "v")
    text = text.replace("'", "").replace(" ", "")
    if not text:
        return None

    tone = 5
    if text[-1].isdigit():
        tone = int(text[-1])
        text = text[:-1]
    if not text:
        return None

    surface_initial = ""
    for candidate in _INITIALS_BY_LENGTH:
        if text.startswith(candidate):
            surface_initial = candidate
            text = text[len(candidate) :]
            break
    else:
        # No consonant initial: a leading y/w is a glide spelling, not part of
        # the final. Feeding it through as a surface initial lets the same
        # normalisation apply as for `analyze` (wo -> uo, yi -> i, yu -> ü).
        if text[0] in ("y", "w"):
            surface_initial = text[0]
            text = text[1:]

    if not text:
        return None
    final = normalize_final(surface_initial, text)
    if final not in MANDARIN_FINALS:
        return None
    return normalize_initial(surface_initial), final, tone


def syllable_signature(text: str, index: int = 0) -> tuple[str, str, int] | None:
    """(initial, final, tone) of the syllable at `index` in Han text."""
    syllables = analyze(text, sandhi=True)
    if not syllables or index >= len(syllables):
        return None
    syllable = syllables[index]
    return syllable.initial, syllable.final, syllable.tone


def parse_pinyin_sequence(value: str) -> list[tuple[str, str, int]]:
    """Parse a possibly multi-syllable transcription into syllables.

    Models emit anything from `ma3` to `wo3 xiang3 chi1 shui4 jiao4` to
    `wo3xiang3chi1shui4jiao4`, so both the spaced and the run-together forms
    are handled.
    """
    if not value:
        return []
    tokens = [t for t in re.split(r"[\s,;/|]+", value.strip()) if t]
    if len(tokens) <= 1:
        raw = (tokens[0] if tokens else value).lower()
        # Split a run-together string after each tone digit.
        tokens = re.findall(r"[a-zü:]+[1-5]?", raw) or tokens
    out = []
    for token in tokens:
        syllable = split_pinyin(token)
        if syllable:
            out.append(syllable)
    return out


def compare_transcription(
    heard_pinyin: str, expected_text: str, index: int = 0
) -> dict:
    """Score a model's pinyin transcription against the expected syllable.

    Used for the open/context prompts, where the model is not told the target
    and simply transcribes what it hears. `expected_text` must be the text the
    *audio* actually contains (the ground truth), not what the learner was
    supposed to say - otherwise this would measure deference, not perception.
    """
    truth = syllable_signature(expected_text, index)
    heard_all = parse_pinyin_sequence(heard_pinyin)
    if truth is None or not heard_all:
        return {"comparable": False, "match": None, "dim": None}
    heard = heard_all[index] if index < len(heard_all) else heard_all[-1]

    dim = diff_dimension(heard, truth)
    return {
        "comparable": True,
        "match": dim == "none",
        "dim": None if dim == "none" else dim,
        "n_heard": len(heard_all),
        "heard": {"initial": heard[0], "final": heard[1], "tone": heard[2]},
        "truth": {"initial": truth[0], "final": truth[1], "tone": truth[2]},
    }
