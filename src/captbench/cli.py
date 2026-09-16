"""Command line interface.

    captbench build     synthesise the corpus (needs a TTS backend)
    captbench score     run the scorers over the manifest
    captbench report    render the markdown scorecard
    captbench run       build + score + report
    captbench info      show configuration and corpus status

A fully offline run works today with `--tts say --scorers traditional`. The
Qwen-Omni side needs DASHSCOPE_API_KEY in `.env`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import build_suite, run_traditional
from .config import Settings, get_settings
from .corpus import Manifest, build_corpus
from .metrics import compare, consistency, scorecard, to_records
from .omni import OmniScorer, interpret
from .report import render_report, write_json, write_report
from .traditional import TraditionalScorer

ALL_FAMILIES = ("tone", "initial", "final", "trap")
DEFAULT_MODES = ("open", "scripted")


def _load_manifest(settings: Settings, require_audio: bool = True) -> Manifest:
    if not settings.manifest_path.exists():
        raise SystemExit(
            f"no manifest at {settings.manifest_path}\n"
            "Run `captbench build` first."
        )
    manifest = Manifest.load(settings.manifest_path)
    if require_audio:
        missing = manifest.missing_audio(settings.data_dir)
        if missing:
            raise SystemExit(
                f"{len(missing)} audio file(s) named by the manifest are missing, "
                f"e.g. {missing[0]}\n"
                "The corpus and the manifest have drifted apart. Rebuild with "
                "`captbench build --force`."
            )
    return manifest


def _items_index(manifest: Manifest) -> dict[str, dict]:
    return {i["id"]: i for i in manifest.scored_items()}


def _select(items: list[dict], args) -> list[dict]:
    if getattr(args, "items", None):
        wanted = {i.strip() for i in args.items.split(",") if i.strip()}
        items = [i for i in items if i["id"] in wanted]
    if getattr(args, "families", None):
        wanted = {f.strip() for f in args.families.split(",") if f.strip()}
        items = [i for i in items if i["family"] in wanted]
    if getattr(args, "limit", None):
        items = items[: args.limit]
    return items


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_build(args, settings: Settings) -> int:
    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    build_corpus(
        settings,
        backend=args.tts,
        families=families,
        limit=args.limit,
        force=args.force,
        voice=args.voice,
        teacher_voice=args.teacher_voice,
    )
    return 0


def cmd_score(args, settings: Settings) -> int:
    manifest = _load_manifest(settings)
    items = _select(manifest.scored_items(), args)
    scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]
    results_dir = settings.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if "traditional" in scorers:
        print(f"calibrating traditional engine over {len(items)} items")
        suite = build_suite(settings, manifest)
        results = run_traditional(suite, item_ids={i["id"] for i in items})
        path = results_dir / "traditional.jsonl"
        _write_jsonl(path, results)
        written.append(path)
        print(f"  -> {path}")

    if "omni" in scorers:
        if not settings.has_api_key():
            print(
                "skipping omni: DASHSCOPE_API_KEY is not set.\n"
                "  Add it to .env (see .env.example), then re-run with "
                "`--scorers omni`.",
                file=sys.stderr,
            )
        else:
            modes = [m.strip() for m in args.modes.split(",") if m.strip()]
            scorer = OmniScorer(settings, temperature=args.temperature)
            results = []
            total = len(items) * len(modes)
            done = 0
            for item in items:
                for mode in modes:
                    # Context prompts only make sense where a scenario exists.
                    if mode == "context" and not item.get("context"):
                        continue
                    done += 1
                    print(f"  omni [{done}/{total}] {item['id']} ({mode})", flush=True)
                    response = scorer.score(
                        item, settings.data_dir / item["audio"], mode=mode,
                        use_cache=not args.no_cache,
                    )
                    verdict = interpret(response, item, mode)
                    results.append(
                        {
                            "item_id": item["id"],
                            "scorer": "omni",
                            "mode": mode,
                            "model": scorer.model,
                            "heard_pinyin": verdict["heard_pinyin"],
                            "heard_tone": verdict["heard_tone"],
                            "truth_tone": verdict["truth_tone"],
                            "tone_correct": verdict["tone_correct"],
                            "verdict": verdict,
                            "parsed": response.parsed,
                            "latency_s": response.latency_s,
                            "usage": response.usage,
                            "error": response.error,
                        }
                    )
            path = results_dir / "omni.jsonl"
            _write_jsonl(path, results)
            written.append(path)
            print(f"  -> {path}")
            print(f"  calls={scorer.stats['calls']} cache_hits={scorer.stats['cache_hits']} "
                  f"errors={scorer.stats['errors']} "
                  f"total_latency={scorer.stats['total_latency']:.1f}s")

    if args.repeats > 1:
        _run_consistency(args, settings, manifest, items, results_dir)

    for path in written:
        print(f"wrote {path}")
    return 0


def _run_consistency(args, settings: Settings, manifest: Manifest, items, results_dir: Path):
    """Repeat a small sample to measure run-to-run agreement."""
    if not settings.has_api_key():
        print("skipping consistency runs: no API key")
        return
    sample = items[: args.consistency_n]
    scorer = OmniScorer(settings, temperature=args.temperature)
    runs = []
    for run_index in range(args.repeats):
        records = []
        for item in sample:
            response = scorer.score(
                item, settings.data_dir / item["audio"], mode="scripted", use_cache=False
            )
            verdict = interpret(response, item, "scripted")
            records.append(
                {
                    "item_id": item["id"],
                    "scorer": "omni",
                    "mode": "scripted",
                    "verdict": verdict,
                    "heard_tone": verdict["heard_tone"],
                    "latency_s": response.latency_s,
                    "error": response.error,
                }
            )
        runs.append(
            to_records(records, _items_index(manifest), "omni")
        )
        print(f"  consistency run {run_index + 1}/{args.repeats} done")
    info = consistency(runs)
    path = results_dir / "consistency.json"
    write_json(path, info)
    print(f"  consistency -> {path}  {info}")


def cmd_report(args, settings: Settings) -> int:
    manifest = _load_manifest(settings)
    items = _items_index(manifest)
    results_dir = settings.results_dir
    records: dict[str, list] = {}
    cards: dict[str, dict] = {}

    trad_path = results_dir / "traditional.jsonl"
    if trad_path.exists():
        results = _read_jsonl(trad_path)
        recs = to_records(results, items, "traditional")
        records["traditional"] = recs
        cards["traditional"] = scorecard(recs)

    omni_path = results_dir / "omni.jsonl"
    if omni_path.exists():
        results = _read_jsonl(omni_path)
        by_mode: dict[str, list] = {}
        for entry in results:
            by_mode.setdefault(entry.get("mode", "scripted"), []).append(entry)
        for mode, entries in by_mode.items():
            recs = to_records(entries, items, "omni")
            key = f"omni:{mode}"
            records[key] = recs
            cards[key] = scorecard(recs)

    if not cards:
        raise SystemExit(
            f"no results in {results_dir}. Run `captbench score` first."
        )

    consistency_info = {}
    cons_path = results_dir / "consistency.json"
    if cons_path.exists():
        payload = json.loads(cons_path.read_text(encoding="utf-8"))
        consistency_info["omni:scripted"] = payload

    rows = compare(cards)
    notes = _limitations(manifest)
    text = render_report(manifest, cards, rows, records, consistency_info, notes)
    out = write_report(results_dir / "report.md", text)
    write_json(
        results_dir / "scorecards.json",
        {"comparison": rows, "cards": cards},
    )
    print(text)
    print(f"\nreport -> {out}")
    return 0


def cmd_run(args, settings: Settings) -> int:
    needs_build = args.force or not settings.manifest_path.exists()
    if not needs_build:
        existing = Manifest.load(settings.manifest_path)
        if existing.missing_audio(settings.data_dir):
            print("manifest references missing audio - rebuilding the corpus")
            needs_build = True
    if needs_build:
        cmd_build(args, settings)
    cmd_score(args, settings)
    return cmd_report(args, settings)


def cmd_info(args, settings: Settings) -> int:
    print("captbench configuration")
    print(f"  project root     {settings.data_dir.parent}")
    print(f"  data dir         {settings.data_dir}")
    print(f"  API key          {'set' if settings.has_api_key() else 'NOT SET'}")
    print(f"  TTS model        {settings.tts_model} (voice {settings.tts_voice}, "
          f"teacher {settings.tts_teacher_voice})")
    print(f"  Omni model       {settings.omni_model}")
    print(f"  native base      {settings.native_base}")
    print(f"  compat base      {settings.compat_base}")
    if settings.manifest_path.exists():
        manifest = Manifest.load(settings.manifest_path)
        print(f"  manifest         {settings.manifest_path}")
        print(f"    backend={manifest.backend} voice={manifest.voice} "
              f"teacher={manifest.teacher_voice} created={manifest.created}")
        print(f"    {len(manifest.scored_items())} scored items, "
              f"{len(manifest.references)} references")
        print(f"    stats {manifest.stats}")
        print(f"  audio cache      {settings.audio_dir / 'tts_cache'}")
    else:
        print("  manifest         (none - run `captbench build`)")
    return 0


def _limitations(manifest: Manifest) -> list[str]:
    notes = [
        "**合成语料而非真实学习者**：错误项由 TTS 朗读最小对立对的另一侧构造，"
        "是干净的范畴替换；真实学习者的偏误是连续渐变的。因此本基准测的是"
        "“能否区分范畴替换”，不是“能否给出口音分数”。",
        "**传统引擎为参考实现**：这里没有使用讯飞/SpeechSuper 等商业引擎，"
        "而是按报告描述的 GOP / 五度 T 值 / DTW 路线自行实现，"
        "其绝对水平不代表商业引擎的水平。",
        "**声调原型采用了说话人自适应**：原型由 teacher 音色的正确读音拟合，"
        "并且对每个条目留出该条目所属的最小对立组（leave-one-contrast-out），"
        "因此不构成对被测音频的记忆。",
        "**误收率的分母是错误条目数**，误拒率的分母是正确条目数。",
    ]
    if manifest.backend == "say":
        notes.append(
            "**本次使用 macOS `say` 本地音色**：learner 与 teacher 为同一音色的"
            "不同语速读数。换成 `--tts qwen` 后 teacher 使用不同音色，"
            "跨说话人条件下的表现需要重新跑一遍才能比较。"
        )
    return notes


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captbench",
        description="Benchmark traditional CAPT scoring against Qwen-Omni scoring "
                    "for Mandarin pronunciation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--families", default=",".join(ALL_FAMILIES),
                       help="comma-separated: tone,initial,final,trap")
        p.add_argument("--limit", type=int, default=None,
                       help="cap the number of scored items")

    p_build = sub.add_parser("build", help="synthesise the corpus")
    add_common(p_build)
    p_build.add_argument("--tts", default="say", choices=["say", "qwen"])
    p_build.add_argument("--voice", default=None)
    p_build.add_argument("--teacher-voice", default=None)
    p_build.add_argument("--force", action="store_true", help="re-render cached clips")
    p_build.set_defaults(func=cmd_build)

    p_score = sub.add_parser("score", help="run the scorers")
    add_common(p_score)
    p_score.add_argument("--scorers", default="traditional",
                         help="comma-separated: traditional,omni")
    p_score.add_argument("--modes", default=",".join(DEFAULT_MODES),
                         help="omni prompt modes: open,context,scripted")
    p_score.add_argument("--items", default=None, help="comma-separated item ids")
    p_score.add_argument("--temperature", type=float, default=None)
    p_score.add_argument("--no-cache", action="store_true")
    p_score.add_argument("--repeats", type=int, default=1,
                         help="repeat the sample N times for a consistency estimate")
    p_score.add_argument("--consistency-n", type=int, default=10,
                         help="how many items to repeat")
    p_score.set_defaults(func=cmd_score)

    p_report = sub.add_parser("report", help="render the scorecard")
    p_report.set_defaults(func=cmd_report)

    p_run = sub.add_parser("run", help="build + score + report")
    add_common(p_run)
    p_run.add_argument("--tts", default="say", choices=["say", "qwen"])
    p_run.add_argument("--voice", default=None)
    p_run.add_argument("--teacher-voice", default=None)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--scorers", default="traditional")
    p_run.add_argument("--modes", default=",".join(DEFAULT_MODES))
    p_run.add_argument("--items", default=None)
    p_run.add_argument("--temperature", type=float, default=None)
    p_run.add_argument("--no-cache", action="store_true")
    p_run.add_argument("--repeats", type=int, default=1)
    p_run.add_argument("--consistency-n", type=int, default=10)
    p_run.set_defaults(func=cmd_run)

    p_info = sub.add_parser("info", help="show configuration")
    p_info.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        return args.func(args, settings)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
