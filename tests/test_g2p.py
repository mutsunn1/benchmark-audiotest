"""G2P: syllable decomposition, normalisation and tone sandhi."""

import pytest

from captbench.f0 import template_curve
from captbench.g2p import (
    TONE_TEMPLATES,
    analyze,
    minimal_pair_diff,
    normalize_final,
    to_pinyin,
    tone_mark,
    tones_of,
)


@pytest.mark.parametrize(
    "char,initial,final,tone",
    [
        ("妈", "m", "a", 1),
        ("知", "zh", "i", 1),
        ("资", "z", "i", 1),
        # y/w are glides, not initials.
        ("一", "", "i", 1),
        ("五", "", "u", 3),
        ("我", "", "uo", 3),
        ("也", "", "ie", 3),
        ("有", "", "iou", 3),
        # y + u is really ü.
        ("鱼", "", "v", 2),
        ("月", "", "ve", 4),
        ("云", "", "vn", 2),
        ("用", "", "iong", 4),
        ("女", "n", "v", 3),
        # j/q/x + u is ü too.
        ("居", "j", "v", 1),
        ("军", "j", "vn", 1),
        ("学", "x", "ve", 2),
        # Abbreviated finals expand.
        ("水", "sh", "uei", 3),
        ("六", "l", "iou", 4),
        ("论", "l", "uen", 4),
    ],
)
def test_syllable_decomposition(char, initial, final, tone):
    syllable = analyze(char)[0]
    assert (syllable.initial, syllable.final, syllable.tone) == (initial, final, tone)


def test_phrase_disambiguation_beats_char_by_char():
    """睡觉 is shuì jiào; read character by character 觉 would be jué."""
    syllables = analyze("睡觉")
    assert [s.pinyin for s in syllables] == ["shui4", "jiao4"]


def test_tone_marks():
    assert tone_mark("ma1") == "mā"
    assert tone_mark("ma3") == "mǎ"
    assert tone_mark("nv3") == "nǚ"
    assert tone_mark("shui4") == "shuì"
    assert tone_mark("ma5") == "ma"  # neutral tone carries no mark


def test_tone_sandhi_third_third():
    assert tones_of("你好", sandhi=True) == [2, 3]
    assert tones_of("你好", sandhi=False) == [3, 3]


def test_tone_sandhi_bu_and_yi():
    assert tones_of("不是", sandhi=True) == [2, 4]
    assert tones_of("一起", sandhi=True) == [4, 3]
    assert tones_of("一个", sandhi=True) == [2, 4]
    # 一 before a non-4th tone becomes 4th, not 2nd.
    assert tones_of("一天", sandhi=True) == [4, 1]


def test_sandhi_leaves_other_tones_alone():
    assert tones_of("妈妈", sandhi=True) == [1, 1]


def test_minimal_pair_diff_names_the_single_axis():
    assert minimal_pair_diff(analyze("知")[0], analyze("资")[0]) == "initial"
    assert minimal_pair_diff(analyze("女")[0], analyze("你")[0]) == "final"
    assert minimal_pair_diff(analyze("水")[0], analyze("睡")[0]) == "tone"
    assert minimal_pair_diff(analyze("妈")[0], analyze("妈")[0]) == "none"
    # 知 vs 女 differs on more than one axis.
    assert minimal_pair_diff(analyze("知")[0], analyze("女")[0]) == "multiple"


def test_all_declared_minimal_pairs_differ_on_one_axis_only():
    """The inventory's segmental pairs must really be minimal pairs."""
    from captbench.inventory import SEGMENTAL_PAIRS

    for contrast, char_a, char_b, dim, _ in SEGMENTAL_PAIRS:
        a, b = analyze(char_a)[0], analyze(char_b)[0]
        assert minimal_pair_diff(a, b) == dim, f"{contrast}: {char_a}/{char_b}"


def test_tone_templates_span_the_five_level_scale():
    for tone, variants in TONE_TEMPLATES.items():
        for variant in range(len(variants)):
            curve = template_curve(tone, variant, 20)
            assert len(curve) == 20
            assert all(-0.5 <= v <= 5.5 for v in curve), f"T{tone} out of range"


def test_tone_template_shapes_are_phonologically_right():
    def ends(tone):
        c = template_curve(tone, 0, 20)
        return c[0], c[-1]

    t1_start, t1_end = ends(1)
    assert abs(t1_end - t1_start) < 0.5, "阴平 should be level"
    assert min(t1_start, t1_end) > 4.0, "阴平 should sit high"

    t2_start, t2_end = ends(2)
    assert t2_end > t2_start + 1.5, "阳平 should rise"

    t3 = template_curve(3, 0, 20)
    assert min(t3) < 1.0, "上声 must dip low"

    t4_start, t4_end = ends(4)
    assert t4_start > 4.0 and t4_end < 1.5, "去声 should fall high to low"


def test_to_pinyin_roundtrip():
    assert to_pinyin("你好", sandhi=True) == "ní hǎo"
    assert to_pinyin("我想吃水饺", sandhi=True) == "wó xiǎng chī shuí jiǎo"


def test_normalize_final_is_idempotent_on_plain_finals():
    for final in ("a", "ang", "uo", "ei", "ao"):
        assert normalize_final("b", final) == final


# ---------------------------------------------------- transcription parsing
@pytest.mark.parametrize(
    "char,pinyin",
    [
        ("我", "wo3"), ("一", "yi1"), ("鱼", "yu2"), ("也", "ye3"),
        ("五", "wu3"), ("英", "ying1"), ("水", "shui3"), ("女", "nv3"),
        ("学", "xue2"), ("用", "yong4"), ("知", "zhi1"), ("安", "an1"),
    ],
)
def test_split_pinyin_agrees_with_analyze(char, pinyin):
    """A model's transcription must land in the same space as our reference."""
    from captbench.g2p import split_pinyin

    syllable = analyze(char)[0]
    assert split_pinyin(pinyin) == (syllable.initial, syllable.final, syllable.tone)


def test_split_pinyin_accepts_alternative_umlaut_spellings():
    from captbench.g2p import split_pinyin

    assert split_pinyin("nü3") == split_pinyin("nv3") == ("n", "v", 3)
    assert split_pinyin("lu:4") == ("l", "v", 4)


def test_split_pinyin_rejects_unparseable_input():
    from captbench.g2p import split_pinyin

    assert split_pinyin("") is None
    assert split_pinyin("!!!") is None
    assert split_pinyin("3") is None


def test_split_pinyin_decodes_tone_diacritics():
    """Models answer in `shuì` as readily as `shui4`."""
    from captbench.g2p import split_pinyin

    assert split_pinyin("mǎ") == ("m", "a", 3)
    assert split_pinyin("shuì") == ("sh", "uei", 4)
    assert split_pinyin("wǒ") == ("", "uo", 3)
    assert split_pinyin("chī") == ("ch", "i", 1)
    assert split_pinyin("nǚ") == ("n", "v", 3)


def test_bare_umlaut_does_not_swallow_the_tone_digit():
    """ü is a vowel spelling, not a tone mark - `nü3` still carries tone 3."""
    from captbench.g2p import split_pinyin

    assert split_pinyin("nü3") == ("n", "v", 3)
    assert split_pinyin("lu:4") == ("l", "v", 4)


def test_parse_pinyin_sequence_handles_diacritics_with_trailing_digits():
    """A real reply: diacritics on the syllables, a bare digit run after."""
    from captbench.g2p import parse_pinyin_sequence

    seq = parse_pinyin_sequence("wǒ xiǎng chī shuì jiào 3 3 1 4")
    assert len(seq) == 5
    assert seq[3] == ("sh", "uei", 4)


def test_parse_pinyin_sequence_handles_spaced_and_runtogether():
    from captbench.g2p import parse_pinyin_sequence

    spaced = parse_pinyin_sequence("wo3 xiang3 chi1 shui4 jiao4")
    runon = parse_pinyin_sequence("wo3xiang3chi1shui4jiao4")
    assert len(spaced) == 5
    assert spaced == runon
    assert spaced[3] == ("sh", "uei", 4)  # 睡


def test_compare_transcription_scores_each_dimension():
    from captbench.g2p import compare_transcription

    right = compare_transcription("ma2", "麻")
    assert right["comparable"] and right["match"] and right["dim"] is None

    tone = compare_transcription("ma1", "麻")
    assert not tone["match"] and tone["dim"] == "tone"

    initial = compare_transcription("zi1", "知")
    assert not initial["match"] and initial["dim"] == "initial"

    final = compare_transcription("ni3", "女")
    assert not final["match"] and final["dim"] == "final"


def test_compare_transcription_picks_the_focus_syllable_in_a_sentence():
    from captbench.g2p import compare_transcription

    # 睡觉 substituted for 水饺: the focus is index 3.
    heard = "wo3 xiang3 chi1 shui4 jiao4"
    substituted = compare_transcription(heard, "我想吃睡觉", 3)
    assert substituted["match"], "the model transcribed what was actually said"
    expected = compare_transcription(heard, "我想吃水饺", 3)
    assert not expected["match"] and expected["dim"] == "tone"


def test_compare_transcription_reports_incomparable_input():
    from captbench.g2p import compare_transcription

    assert compare_transcription("", "麻")["comparable"] is False
    assert compare_transcription("gibberish", "麻")["comparable"] is False
