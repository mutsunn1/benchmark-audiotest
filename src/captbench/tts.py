"""Text-to-speech backends used to synthesise the stimulus corpus.

Two backends, one interface:

* `qwen` - Qwen3-TTS-Flash over the DashScope REST API. Synthesises the real
  corpus. Needs DASHSCOPE_API_KEY.
* `say`  - the macOS `say` command with a zh_CN voice. Lets the whole pipeline
  build and run end to end with no credentials, which is how the traditional
  scorer was developed and validated.

Every text is rendered in two **roles**, and the distinction matters:

* `learner` - the clip that gets scored.
* `teacher` - a separate take used as the model pronunciation for alignment,
  tone prototypes and decision boundaries.

They must not be the same recording. If the learner's "correct" clip were
byte-identical to the reference, every DTW-based check would score a perfect
zero and the benchmark would flatter itself. `say` varies the speaking rate
between roles; Qwen TTS uses a different voice, which additionally makes the
comparison cross-speaker - the situation the five-level T-value transform
exists to handle.

Both roles write 16 kHz mono WAV, so nothing downstream knows which voice
produced a clip. Results are cached by (backend, role, voice, text), so
re-running a build is free and an interrupted run resumes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .audio import load_audio, normalize_peak, save_audio, trim_silence
from .config import Settings

TARGET_SR = 16000
LEARNER = "learner"
TEACHER = "teacher"

#: `say` speaking rates per role.
#:
#: These must land in *different quantisation buckets*. `say` snaps the rate to
#: a coarse internal grid, so nearby values render byte-identical audio: 160 and
#: 180 produce the same file for short text. Roles in the same bucket make the
#: learner's "correct" clip bit-identical to the reference, which silently
#: invalidates every DTW-based check (each correct item scores a perfect zero).
#: `build_corpus` verifies the two takes actually differ.
SAY_RATES = {LEARNER: 180, TEACHER: 120}


@dataclass
class SynthResult:
    path: Path
    text: str
    backend: str
    role: str
    voice: str
    cached: bool
    latency_s: float = 0.0
    chars: int = 0


def cache_key(backend: str, role: str, voice: str, text: str, variant: str = "") -> str:
    """Cache identity of a clip.

    `variant` carries whatever else changes the audio for a given (backend,
    role, voice, text) - the speaking rate, for instance. Without it, changing
    a rate would silently serve the previously rendered clip.
    """
    digest = hashlib.sha1(
        f"{backend}|{role}|{voice}|{variant}|{text}".encode("utf-8")
    ).hexdigest()[:16]
    safe = "".join(c for c in text if c.isalnum())[:12] or "clip"
    return f"{backend}_{role}_{voice}_{safe}_{digest}"


class TTSEngine:
    name = "base"

    def __init__(self, voice: str, cache_dir: Path, role: str = LEARNER, variant: str = ""):
        self.voice = voice
        self.role = role
        self.variant = variant
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"calls": 0, "cache_hits": 0, "total_latency": 0.0, "chars": 0, "errors": 0}

    def synth(self, text: str, force: bool = False) -> SynthResult:
        out = self.cache_dir / f"{cache_key(self.name, self.role, self.voice, text, self.variant)}.wav"
        if out.exists() and not force:
            self.stats["cache_hits"] += 1
            return SynthResult(out, text, self.name, self.role, self.voice, cached=True)

        started = time.time()
        raw = self._render(text)
        signal, sr = load_audio(raw, target_sr=TARGET_SR)
        signal = normalize_peak(trim_silence(signal, sr), peak=0.9)
        save_audio(out, signal, TARGET_SR)
        latency = time.time() - started

        self.stats["calls"] += 1
        self.stats["total_latency"] += latency
        self.stats["chars"] += len(text)
        if raw.exists() and raw.parent != out.parent:
            shutil.rmtree(raw.parent, ignore_errors=True)
        return SynthResult(out, text, self.name, self.role, self.voice, cached=False,
                           latency_s=latency, chars=len(text))

    def _render(self, text: str) -> Path:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}/{self.role} (voice={self.voice})"


class SayTTS(TTSEngine):
    """macOS local Mandarin voice. No credentials, fully offline."""

    name = "say"
    #: `say` occasionally blocks forever instead of returning - seen in
    #: practice, hung on a single syllable for 20+ minutes. Every call is
    #: bounded and retried rather than allowed to stall a whole build.
    timeout_s = 30.0
    max_retries = 2

    def __init__(self, voice: str, cache_dir: Path, role: str = LEARNER, rate: int | None = None):
        self.rate = rate or SAY_RATES.get(role, 180)
        super().__init__(voice, cache_dir, role, variant=f"r{self.rate}")

    def _render(self, text: str) -> Path:
        last_error: str = ""
        for attempt in range(self.max_retries):
            tmpdir = Path(tempfile.mkdtemp(prefix="captbench_say_"))
            aiff = tmpdir / "out.aiff"
            try:
                proc = subprocess.run(
                    ["say", "-v", self.voice, "-r", str(self.rate), "-o", str(aiff), text],
                    capture_output=True, text=True, timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired:
                last_error = f"`say` timed out after {self.timeout_s:.0f}s (attempt {attempt + 1})"
                shutil.rmtree(tmpdir, ignore_errors=True)
                continue
            if proc.returncode == 0 and aiff.exists() and aiff.stat().st_size > 0:
                return aiff
            last_error = (proc.stderr or f"exit {proc.returncode}").strip()
            shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"`say` failed for {text!r}: {last_error}")

    @staticmethod
    def available_voices() -> list[str]:
        try:
            out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
        except FileNotFoundError:
            return []
        return [line.split()[0] for line in out.splitlines() if "zh_CN" in line or "zh_TW" in line]


class QwenTTS(TTSEngine):
    """Qwen3-TTS-Flash via DashScope's multimodal-generation endpoint.

    Returns a URL to the rendered audio, which is downloaded and normalised.
    """

    name = "qwen"

    def __init__(self, voice: str, cache_dir: Path, settings: Settings,
                 role: str = LEARNER, model: str | None = None):
        super().__init__(voice, cache_dir, role)
        self.settings = settings
        self.model = model or settings.tts_model
        self.max_retries = 4

    def _render(self, text: str) -> Path:
        api_key = self.settings.require_api_key()
        payload = {
            "model": self.model,
            "input": {"text": text, "voice": self.voice, "language_type": "Chinese"},
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.settings.request_timeout) as client:
                    response = client.post(self.settings.tts_url, json=payload, headers=headers)
                    if response.status_code == 429 or response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"retryable {response.status_code}", request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    body = response.json()
                    audio = (body.get("output") or {}).get("audio") or {}
                    url = audio.get("url")
                    if not url:
                        raise RuntimeError(f"no audio URL in response: {json.dumps(body)[:400]}")
                    tmpdir = Path(tempfile.mkdtemp(prefix="captbench_qwen_"))
                    suffix = Path(url.split("?")[0]).suffix or ".wav"
                    raw = tmpdir / f"out{suffix}"
                    with client.stream("GET", url) as stream:
                        stream.raise_for_status()
                        with open(raw, "wb") as fh:
                            for chunk in stream.iter_bytes():
                                fh.write(chunk)
                    return raw
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
                self.stats["errors"] += 1
                if attempt < self.max_retries - 1:
                    time.sleep(1.5 * (2**attempt))
        raise RuntimeError(
            f"Qwen TTS failed for {text!r} after {self.max_retries} tries: {last_error}"
        )


def make_engine(
    backend: str,
    settings: Settings,
    role: str = LEARNER,
    voice: str | None = None,
    cache_dir: Path | None = None,
) -> TTSEngine:
    """Build a TTS engine for one role."""
    cache_dir = cache_dir or (settings.audio_dir / "tts_cache")
    if backend == "qwen":
        default_voice = settings.tts_voice if role == LEARNER else settings.tts_teacher_voice
        return QwenTTS(voice or default_voice, cache_dir, settings, role=role)
    if backend == "say":
        if voice is None:
            preferred = "Tingting"
            available = SayTTS.available_voices()
            voice = preferred if preferred in available else (available[0] if available else preferred)
        return SayTTS(voice, cache_dir, role=role)
    raise ValueError(f"unknown TTS backend: {backend!r} (expected 'qwen' or 'say')")
