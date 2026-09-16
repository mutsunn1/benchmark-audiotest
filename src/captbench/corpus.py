"""Corpus build: synthesise every clip the benchmark needs and record the truth.

Two kinds of artefact come out of a build:

* **items** - the clips to be scored, each carrying its ground truth (`expected`
  vs `spoken`, so `is_correct` is known by construction).
* **references** - clean readings of each contrast member, used to calibrate
  the traditional scorer (tone prototypes, segmental decision boundaries,
  nearest-reference comparison). References are never scored.

Everything is written to 16 kHz mono WAV in one cache directory, and the
manifest stores paths relative to the data directory so the corpus can be moved.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .g2p import analyze
from .inventory import Stimulus, build_inventory
from .tts import LEARNER, TEACHER, TTSEngine, make_engine

MANIFEST_VERSION = 1


@dataclass
class Manifest:
    backend: str
    voice: str
    created: str
    teacher_voice: str = ""
    items: list[dict] = field(default_factory=list)
    references: dict[str, dict] = field(default_factory=dict)
    enrollment: list[dict] = field(default_factory=list)
    sentence_calibration: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    version: int = MANIFEST_VERSION

    # -- lookup helpers --------------------------------------------------
    def scored_items(self) -> list[dict]:
        """Items to be scored - calibration clips are synthesised but excluded."""
        return [i for i in self.items if i["family"] != "calibration"]

    def references_for(self, contrast: str) -> dict[str, dict]:
        """All reference readings belonging to a contrast, keyed by unit."""
        return {
            ref["unit"]: ref
            for key, ref in self.references.items()
            if ref["contrast"] == contrast
        }

    def missing_audio(self, data_dir: Path, sample: int | None = None) -> list[str]:
        """Audio paths named by the manifest that are no longer on disk.

        The manifest and the audio cache can drift apart - a `rm -rf` of the
        cache, a corpus copied without its clips. Scoring against dangling paths
        would silently produce a run of errors, so callers check first.
        """
        paths: list[str] = []
        for item in self.items:
            paths.append(item["audio"])
            if item.get("ref_audio"):
                paths.append(item["ref_audio"])
        checked = paths if sample is None else paths[:sample]
        return [p for p in checked if not (data_dir / p).exists()]

    def tone_references(self, exclude_contrasts: set[str] | None = None) -> list[dict]:
        """Teacher readings usable for fitting tone prototypes."""
        exclude = exclude_contrasts or set()
        return [
            ref
            for ref in self.references.values()
            if ref.get("tone") and ref["contrast"] not in exclude
        ]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "backend": self.backend,
            "voice": self.voice,
            "teacher_voice": self.teacher_voice,
            "created": self.created,
            "stats": self.stats,
            "items": self.items,
            "references": self.references,
            "enrollment": self.enrollment,
            "sentence_calibration": self.sentence_calibration,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            backend=raw["backend"],
            voice=raw["voice"],
            created=raw.get("created", ""),
            teacher_voice=raw.get("teacher_voice", ""),
            items=raw.get("items", []),
            references=raw.get("references", {}),
            enrollment=raw.get("enrollment", []),
            sentence_calibration=raw.get("sentence_calibration", []),
            stats=raw.get("stats", {}),
            version=raw.get("version", MANIFEST_VERSION),
        )


def _same_audio(a: Path, b: Path) -> bool:
    """True when two clips are the same recording (same bytes)."""
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
        return a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def build_corpus(
    settings: Settings,
    backend: str = "say",
    families: tuple[str, ...] = ("tone", "initial", "final", "trap"),
    limit: int | None = None,
    force: bool = False,
    voice: str | None = None,
    teacher_voice: str | None = None,
    progress: bool = True,
) -> Manifest:
    """Synthesise the whole corpus (both roles) and return the manifest."""
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    learner = make_engine(backend, settings, role=LEARNER, voice=voice)
    teacher = make_engine(backend, settings, role=TEACHER, voice=teacher_voice)

    stimuli: list[Stimulus] = build_inventory(families)
    if limit:
        scored = [s for s in stimuli if s.family != "calibration"][:limit]
        keep = {s.id for s in scored}
        stimuli = [s for s in stimuli if s.id in keep or s.family == "calibration"]

    if progress:
        n_scored = sum(1 for s in stimuli if s.family != "calibration")
        print(f"building corpus: {len(stimuli)} stimuli ({n_scored} scored)")
        print(f"  learner: {learner.describe()}")
        print(f"  teacher: {teacher.describe()}")

    started = time.time()
    items: list[dict] = []
    references: dict[str, dict] = {}
    enrollment: list[dict] = []
    sentence_calibration: list[dict] = []
    failures: list[dict] = []
    identical_takes: list[str] = []

    for n, stim in enumerate(stimuli, start=1):
        if progress and (n % 20 == 0 or n == len(stimuli)):
            print(f"  [{n}/{len(stimuli)}] {stim.id}", flush=True)

        try:
            spoken = learner.synth(stim.spoken_text, force=force)
            expected = teacher.synth(stim.expected_text, force=force)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the build
            failures.append({"id": stim.id, "text": stim.spoken_text,
                             "error": f"{type(exc).__name__}: {exc}"})
            print(f"  !! {stim.id}: {exc}", flush=True)
            continue

        # Guard against a silently degenerate corpus: if the learner take and
        # the reference are the same recording, every DTW comparison is a
        # perfect self-match and the benchmark flatters itself.
        if stim.expected_text == stim.spoken_text and _same_audio(spoken.path, expected.path):
            identical_takes.append(stim.id)

        entry = stim.to_dict()
        entry["contrast_units"] = list(stim.contrast_units)
        entry.update(
            {
                "audio": _rel(spoken.path, data_dir),
                "ref_audio": _rel(expected.path, data_dir),
            }
        )
        items.append(entry)

        # Calibration items double as the learner's enrolment set: they are
        # correct productions in the learner's own voice, used only to estimate
        # that speaker's pitch range for the five-level transform.
        if stim.family == "calibration":
            enrollment.append(
                {
                    "id": stim.id,
                    "contrast": stim.contrast,
                    "unit": stim.expected_unit,
                    "tone": _tone_of(stim.expected_unit or ""),
                    "text": stim.spoken_text,
                    "audio": _rel(spoken.path, data_dir),
                }
            )
            # Sentences additionally give the tone model connected-speech
            # instances, with the tones the speaker is expected to produce
            # after sandhi.
            if stim.contrast == "calib_sentence":
                sentence_calibration.append(
                    {
                        "id": stim.id,
                        "text": stim.spoken_text,
                        "audio": _rel(expected.path, data_dir),
                        "tones": [s.tone for s in analyze(stim.spoken_text, sandhi=True)],
                    }
                )

        # Contrast references come from the teacher's correct productions.
        if stim.is_correct and stim.contrast_units and stim.expected_unit:
            references.setdefault(
                f"{stim.contrast}|{stim.expected_unit}",
                {
                    "contrast": stim.contrast,
                    "unit": stim.expected_unit,
                    "text": stim.expected_text,
                    "tone": _tone_of(stim.expected_unit),
                    "family": "calibration" if stim.family == "calibration" else stim.family,
                    "audio": _rel(expected.path, data_dir),
                },
            )

    def _merged(engine) -> dict:
        return {
            "calls": engine.stats["calls"],
            "cache_hits": engine.stats["cache_hits"],
            "errors": engine.stats["errors"],
            "latency_s": round(engine.stats["total_latency"], 2),
            "characters": engine.stats["chars"],
        }

    stats = {
        "n_items": len(items),
        "n_scored": sum(1 for i in items if i["family"] != "calibration"),
        "n_references": len(references),
        "n_failures": len(failures),
        "failures": failures,
        "n_identical_takes": len(identical_takes),
        "identical_takes": identical_takes[:20],
        "learner": _merged(learner),
        "teacher": _merged(teacher),
        "n_synth_calls": learner.stats["calls"] + teacher.stats["calls"],
        "n_cache_hits": learner.stats["cache_hits"] + teacher.stats["cache_hits"],
        "total_synth_latency_s": round(
            learner.stats["total_latency"] + teacher.stats["total_latency"], 2
        ),
        "wall_clock_s": round(time.time() - started, 2),
    }

    manifest = Manifest(
        backend=learner.name,
        voice=learner.voice,
        teacher_voice=teacher.voice,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        items=items,
        references=references,
        enrollment=enrollment,
        sentence_calibration=sentence_calibration,
        stats=stats,
    )
    manifest.save(settings.manifest_path)
    if identical_takes and progress:
        print(
            f"  WARNING: {len(identical_takes)} correct items have a learner take "
            "identical to the teacher reference.\n"
            "  DTW comparisons against those are self-matches and will look "
            "perfect. Put the two TTS roles in different rate buckets "
            "(see SAY_RATES) or use different voices."
        )
    if progress:
        print(f"manifest -> {settings.manifest_path}")
        print(f"  {len(items)} items, {len(references)} references, "
              f"{stats['n_synth_calls']} synth calls, "
              f"{stats['total_synth_latency_s']}s synthesis")
    return manifest


def _tone_of(unit: str) -> int | None:
    if unit and unit.startswith("T") and unit[1:].isdigit():
        return int(unit[1:])
    return None
