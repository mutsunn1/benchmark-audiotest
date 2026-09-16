"""Report generation: markdown tables from the scorecards.

Deliberately text-first. A confusion matrix is more readable as a grid of
numbers in a table than as a heatmap, and the report needs to be diffable
between runs so threshold changes can be tracked.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .corpus import Manifest
from .metrics import Record

SYSTEM_LABELS = {
    "traditional": "Traditional CAPT (acoustic-phonetic)",
    "omni:open": "Qwen-Omni - open transcription",
    "omni:context": "Qwen-Omni - scenario context",
    "omni:scripted": "Qwen-Omni - scripted (target text given)",
}


def _fmt(value, digits: int = 3, percent: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        if percent:
            return f"{value * 100:.1f}%"
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def render_report(
    manifest: Manifest,
    cards: dict[str, dict],
    comparison_rows: list[dict],
    records: dict[str, list[Record]],
    consistency_info: dict | None = None,
    extra_notes: list[str] | None = None,
) -> str:
    n_scored = len(manifest.scored_items())
    n_correct = sum(1 for i in manifest.scored_items() if i["is_correct"])
    parts: list[str] = []

    parts.append("# 中文口语评测基准：传统声学引擎 vs Qwen-Omni\n")
    parts.append(
        f"*生成时间: {datetime.now(timezone.utc).isoformat(timespec='seconds')}*\n"
    )

    parts.append("## 1. 语料\n")
    parts.append(
        _table(
            ["项", "值"],
            [
                ["TTS 后端", f"`{manifest.backend}`"],
                ["学习者音色 (learner)", f"`{manifest.voice}`"],
                ["参考音色 (teacher)", f"`{manifest.teacher_voice}`"],
                ["计分条目", f"{n_scored}（正确 {n_correct} / 错误 {n_scored - n_correct}）"],
                ["校准参考", str(len(manifest.references))],
                ["学习者校准条目", str(len(manifest.enrollment))],
            ],
        )
        + "\n"
    )
    parts.append(
        "每个条目都有**构造性真值**：错误音频是用同一音色朗读最小对立对的另一侧生成的，"
        "因此“正确/错误”及其维度（声调 / 声母 / 韵母）都是已知的，不需要人工标注。\n"
    )

    parts.append("## 2. 总览对比\n")
    parts.append(
        _table(
            ["系统", "n", "判定准确率", "平衡准确率", "误收率 (FAR)", "误拒率 (FRR)",
             "声调识别率", "声调 n", "错误定位率", "中位延迟"],
            [
                [
                    SYSTEM_LABELS.get(r["system"], r["system"]),
                    r["n"],
                    _fmt(r["detection_accuracy"], percent=True),
                    _fmt(r["balanced_accuracy"], percent=True),
                    _fmt(r["false_accept_rate"], percent=True),
                    _fmt(r["false_reject_rate"], percent=True),
                    _fmt(r["tone_accuracy"], percent=True),
                    r["tone_n"],
                    _fmt(r["localisation_accuracy"], percent=True),
                    f"{_fmt(r['median_latency_s'], 2)}s",
                ]
                for r in comparison_rows
            ],
        )
        + "\n"
    )
    parts.append(
        "**误收率 (false accept rate)** 是关键指标：把错误发音判成正确的比例。"
        "报告预测 Omni 模型会因为语言先验而“脑补纠正”，直接体现为误收率升高。\n"
    )

    parts.append("## 3. 判定混淆矩阵\n")
    for name, card in sorted(cards.items()):
        det = card.get("detection", {})
        if not det.get("n"):
            continue
        parts.append(f"### {SYSTEM_LABELS.get(name, name)}\n")
        parts.append(
            _table(
                ["真值 \\ 判定", "判为正确", "判为错误"],
                [
                    ["**实际正确**", f"{det['tn']} (TN)", f"{det['fp']} (FP, 误拒)"],
                    ["**实际错误**", f"{det['fn']} (FN, 误收)", f"{det['tp']} (TP)"],
                ],
            )
            + "\n"
        )

    parts.append("## 4. 声调识别\n")
    parts.append(
        "只统计音频中确实含有某个已知声调的条目，比较各系统给出的调类。"
        "这是最公平的正面对比，因为两个系统面对的是同一段音频。\n"
    )
    for name, card in sorted(cards.items()):
        tone = card.get("tone_identification", {})
        if not tone.get("n"):
            continue
        parts.append(f"### {SYSTEM_LABELS.get(name, name)}\n")
        parts.append(
            f"准确率 {_fmt(tone['accuracy'], percent=True)} "
            f"({tone['hits']}/{tone['n']})\n"
        )
        rows = [[f"T{t}"] for t in (1, 2, 3, 4)]
        for i, t in enumerate((1, 2, 3, 4)):
            for p in (1, 2, 3, 4):
                rows[i].append(str(tone["confusion"].get(f"T{t}_as_T{p}", 0)))
        parts.append(
            _table(["实际 \\ 识别", "T1", "T2", "T3", "T4"], rows) + "\n"
        )

    parts.append("## 5. 分维度与分对立组\n")
    for name, card in sorted(cards.items()):
        fam = card.get("by_family") or {}
        if not fam:
            continue
        parts.append(f"### {SYSTEM_LABELS.get(name, name)}\n")
        parts.append(
            _table(
                ["题型", "n", "准确率", "误收率"],
                [
                    [k, v["n"], _fmt(v["accuracy"], percent=True),
                     _fmt(v["false_accept_rate"], percent=True)]
                    for k, v in fam.items()
                ],
            )
            + "\n"
        )

    parts.append("## 6. 语言陷阱实验\n")
    parts.append(
        "陷阱句的语境强烈预测某个词（如“我想吃水饺”），但音频实际说的是其最小对立伙伴"
        "（“我想吃睡觉”）。信赖预期文本或语义连贯性的系统会把它们判成正确。\n"
    )
    trap_rows = []
    for name, card in sorted(cards.items()):
        trap = card.get("trap", {})
        if not trap.get("n"):
            continue
        trap_rows.append(
            [
                SYSTEM_LABELS.get(name, name),
                trap["n"],
                _fmt(trap["false_accept_rate"], percent=True),
                _fmt(trap["false_reject_rate"], percent=True),
                _fmt(trap["tone_accuracy"], percent=True),
            ]
        )
    if trap_rows:
        parts.append(
            _table(["系统", "n", "陷阱句误收率", "对照句误拒率", "声调识别率"], trap_rows)
            + "\n"
        )
    else:
        parts.append("_（本次运行未包含 trap 题型的 omni 结果）_\n")

    if consistency_info:
        parts.append("## 7. 一致性与延迟\n")
        rows = []
        for name, info in sorted(consistency_info.items()):
            rows.append(
                [
                    SYSTEM_LABELS.get(name, name),
                    info.get("n_runs", "-"),
                    info.get("n_items", "-"),
                    _fmt(info.get("verdict_agreement"), percent=True),
                    _fmt(info.get("tone_agreement"), percent=True),
                ]
            )
        parts.append(
            _table(["系统", "重复次数", "条目数", "判定一致率", "声调一致率"], rows) + "\n"
        )
        parts.append(
            "传统引擎是确定性映射，重复评测结果必然一致；Omni 为自回归采样，"
            "需要实测其稳定性。\n"
        )

    if extra_notes:
        parts.append("## 8. 说明与局限\n")
        for note in extra_notes:
            parts.append(f"- {note}\n")

    return "\n".join(parts)


def write_report(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path
