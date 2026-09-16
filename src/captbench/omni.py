"""The Qwen-Omni scorer.

Where the traditional engine measures physics, this one asks a model. That
makes prompt design the whole experiment, so the scorer runs three modes over
the *same* audio:

* `open`     - "transcribe this Mandarin syllable in pinyin with a tone
               number". Nothing about the expected answer. Measures raw
               perception, and is directly comparable to the traditional
               engine's tone identification.
* `context`  - the communicative scenario, no target text (used for the trap
               sentences). Measures whether semantics pull the transcript
               toward the plausible word.
* `scripted` - the standard CAPT setting: the expected text is supplied and the
               model judges correctness. This is where the "linguistic trap"
               bites: a model that trusts the prompt over the audio will accept
               a substitution it should have caught.

Running all three is what separates "the model misheard" from "the model heard
fine but deferred to the prompt" - two failures that look identical if you only
run the scripted prompt.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import Settings
from .g2p import compare_transcription, to_pinyin

SYSTEM_PROMPT = (
    "You are a meticulous Mandarin pronunciation assessment engine used by a "
    "language school. You judge what the audio actually contains, never what it "
    "was supposed to contain. You always reply with a single JSON object and "
    "nothing else - no markdown fences, no commentary."
)

OPEN_PROMPT = """Listen to this recording of one Mandarin syllable or word.

Transcribe it exactly as you hear it, as pinyin with a tone number (1-4, or 5 for neutral).

Reply with JSON only:
{"pinyin": "<syllable(s) with tone digits, e.g. ma3>", "tone": <1-4 or 5>, "confidence": <0.0-1.0>}"""

CONTEXT_PROMPT = """A Mandarin learner is speaking in this situation: {context}

Listen to the recording. Transcribe what they actually said - do not correct it toward what would make sense in the situation.

Reply with JSON only:
{{"pinyin": "<what you heard, with tone digits>", "tone": <1-4 or 5>, "confidence": <0.0-1.0>}}"""

SCRIPTED_PROMPT = """A Mandarin learner was asked to read this aloud: {text} ({pinyin})

Listen to the recording and judge how they actually pronounced it. Report what you heard, not what was expected.

Reply with JSON only:
{{"heard_pinyin": "<what you actually heard, with tone digits>", "heard_tone": <1-4 or 5>,
  "matches_target": <true if the pronunciation matches the target syllable and tone>,
  "error_type": "<none|tone|initial|final|other>",
  "observed_unit": "<the tone number or phone you heard, e.g. 2 or sh>",
  "confidence": <0.0-1.0>,
  "explanation": "<one short sentence>"}}"""


@dataclass
class OmniResponse:
    mode: str
    raw_text: str
    parsed: dict = field(default_factory=dict)
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    error: str | None = None
    cache_hit: bool = False

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "parsed": self.parsed,
            "latency_s": round(self.latency_s, 4),
            "usage": self.usage,
            "error": self.error,
            "cache_hit": self.cache_hit,
            "raw_text": self.raw_text[:2000],
        }


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply."""
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    # Fall back to brace matching so nested objects survive.
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : i + 1])
                start = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}


def normalise_tone(value) -> int:
    """Coerce whatever the model returned for a tone into 1-5, else 0."""
    if isinstance(value, int) and 1 <= value <= 5:
        return value
    if isinstance(value, str):
        m = re.search(r"[1-5]", value)
        if m:
            return int(m.group())
    return 0


class OmniScorer:
    """Qwen-Omni over DashScope's OpenAI-compatible endpoint."""

    name = "omni"

    def __init__(
        self,
        settings: Settings,
        model: str | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
        cache_dir: Path | None = None,
    ):
        self.settings = settings
        self.model = model or settings.omni_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_dir = cache_dir or (settings.data_dir / "omni_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"calls": 0, "errors": 0, "total_latency": 0.0, "cache_hits": 0}

    # -- prompt construction --------------------------------------------
    def build_prompt(self, stimulus: dict, mode: str) -> str:
        if mode == "open":
            return OPEN_PROMPT
        if mode == "context":
            context = stimulus.get("context") or "a Mandarin pronunciation exercise"
            return CONTEXT_PROMPT.format(context=context)
        if mode == "scripted":
            return SCRIPTED_PROMPT.format(
                text=stimulus["expected_text"],
                pinyin=to_pinyin(stimulus["expected_text"]),
            )
        raise ValueError(f"unknown omni prompt mode: {mode!r}")

    # -- transport -------------------------------------------------------
    def _encode_audio(self, path: Path) -> str:
        data = Path(path).read_bytes()
        if len(data) > 9 * 1024 * 1024:
            raise ValueError(f"audio too large for base64 input: {len(data)} bytes")
        return "data:;base64," + base64.b64encode(data).decode("ascii")

    def _cache_path(self, audio_b64: str, prompt: str) -> Path:
        import hashlib

        digest = hashlib.sha1(
            f"{self.model}|{self.temperature}|{prompt}|{audio_b64}".encode("utf-8")
        ).hexdigest()[:20]
        return self.cache_dir / f"{digest}.json"

    def _call(self, audio_b64: str, prompt: str) -> tuple[str, dict]:
        api_key = self.settings.require_api_key()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "modalities": ["text"],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._stream(payload, headers)
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
                self.stats["errors"] += 1
                if attempt < self.max_retries - 1:
                    time.sleep(1.5 * (2**attempt))
        raise RuntimeError(f"Qwen-Omni call failed after {self.max_retries} tries: {last_error}")

    def _stream(self, payload: dict, headers: dict) -> tuple[str, dict]:
        parts: list[str] = []
        usage: dict = {}
        with httpx.Client(timeout=self.settings.request_timeout) as client:
            with client.stream(
                "POST", self.settings.omni_url, json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", "replace")
                    raise httpx.HTTPStatusError(
                        f"{response.status_code}: {body[:400]}",
                        request=response.request,
                        response=response,
                    )
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", "replace")
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data in ("", "[DONE]"):
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if isinstance(content, str):
                            parts.append(content)
                        elif isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("text"):
                                    parts.append(part["text"])
        return "".join(parts), usage

    # -- public API ------------------------------------------------------
    def score(
        self,
        stimulus: dict,
        audio_path: Path,
        mode: str = "scripted",
        use_cache: bool = True,
    ) -> OmniResponse:
        prompt = self.build_prompt(stimulus, mode)
        audio_b64 = self._encode_audio(audio_path)
        cache_path = self._cache_path(audio_b64, prompt)

        if use_cache and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.stats["cache_hits"] += 1
            return OmniResponse(
                mode=mode,
                raw_text=cached.get("raw_text", ""),
                parsed=cached.get("parsed", {}),
                # Replay the latency measured when the call was actually made,
                # rather than reporting 0 - a cached re-score still needs to
                # report how slow the model is.
                latency_s=float(cached.get("latency_s") or 0.0),
                usage=cached.get("usage", {}),
                error=cached.get("error"),
                cache_hit=True,
            )

        started = time.time()
        try:
            raw, usage = self._call(audio_b64, prompt)
            parsed = extract_json(raw)
            response = OmniResponse(mode=mode, raw_text=raw, parsed=parsed,
                                    latency_s=time.time() - started, usage=usage)
        except Exception as exc:  # noqa: BLE001 - surfaced in the result
            response = OmniResponse(mode=mode, raw_text="", parsed={},
                                    latency_s=time.time() - started,
                                    error=f"{type(exc).__name__}: {exc}")

        self.stats["calls"] += 1
        self.stats["total_latency"] += response.latency_s
        # Always refresh the cache, even when the caller bypassed reads with
        # use_cache=False: the reply is a pure function of (model, prompt,
        # audio), so re-recording it only ever makes the cache more complete.
        if response.error is None:
            cache_path.write_text(
                json.dumps(
                    {
                        "raw_text": response.raw_text,
                        "parsed": response.parsed,
                        "usage": response.usage,
                        "latency_s": response.latency_s,
                        "error": response.error,
                        "model": self.model,
                        "mode": mode,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return response

    def score_repeated(
        self,
        stimulus: dict,
        audio_path: Path,
        n: int = 3,
        mode: str = "scripted",
        use_cache: bool = False,
    ) -> list[OmniResponse]:
        """Same clip, same prompt, n times - for measuring self-consistency."""
        return [
            self.score(stimulus, audio_path, mode=mode, use_cache=use_cache)
            for _ in range(n)
        ]


def interpret(response: OmniResponse, stimulus: dict, mode: str) -> dict:
    """Turn a raw model reply into the same vocabulary the traditional scorer uses.

    Keeping the two scorers' outputs in one shape is what lets `metrics` compare
    them without special-casing either.
    """
    parsed = response.parsed or {}
    heard_tone = normalise_tone(parsed.get("tone") or parsed.get("heard_tone"))
    heard_pinyin = str(parsed.get("pinyin") or parsed.get("heard_pinyin") or "").strip()

    # The tone the audio actually contains, per the ground truth.
    truth_tone = _tone_of_unit(stimulus.get("spoken_unit") or "")

    result = {
        "mode": mode,
        "heard_pinyin": heard_pinyin,
        "heard_tone": heard_tone,
        "truth_tone": truth_tone,
        "tone_correct": (heard_tone == truth_tone) if (heard_tone and truth_tone) else None,
        "confidence": parsed.get("confidence"),
        "error": response.error,
        "latency_s": response.latency_s,
        "usage": response.usage,
        "model_error_type": parsed.get("error_type"),
        "explanation": parsed.get("explanation"),
    }

    if mode == "scripted":
        matches = parsed.get("matches_target")
        if isinstance(matches, bool):
            result["predicted_correct"] = matches
        elif heard_tone and truth_tone:
            result["predicted_correct"] = heard_tone == truth_tone
        else:
            result["predicted_correct"] = None
        error_type = str(parsed.get("error_type") or "none").lower()
        result["predicted_dim"] = None if error_type in ("none", "") else error_type
        result["predicted_unit"] = str(parsed.get("observed_unit") or "") or None
    else:
        # Open/context: the model transcribed without being told the target.
        # Two different questions can be asked of that transcript, and
        # conflating them corrupts the metrics:
        #
        #   perception - does the transcript match what the audio actually
        #                contains? Measures how well the model hears.
        #   verdict    - does the transcript match what the learner was
        #                supposed to say? This is the correctness judgement a
        #                transcription-based CAPT system would make.
        #
        # Scoring the *verdict* against `spoken_text` would mark every
        # correctly-heard error item as "accepted as correct" - inflating the
        # false accept rate with what is actually a correct perception.
        perception = compare_transcription(
            heard_pinyin, stimulus.get("spoken_text", ""), stimulus.get("focus_index", 0)
        )
        verdict = compare_transcription(
            heard_pinyin, stimulus.get("expected_text", ""), stimulus.get("focus_index", 0)
        )
        result["perception"] = perception
        result["transcription"] = verdict
        if verdict.get("comparable"):
            result["predicted_correct"] = verdict["match"]
            result["predicted_dim"] = verdict["dim"]
            heard_units = verdict.get("heard") or {}
            result["predicted_unit"] = _unit_label(verdict["dim"], heard_units, heard_tone)
        else:
            result["predicted_correct"] = None
            result["predicted_dim"] = None
            result["predicted_unit"] = None

    return result


def _unit_label(dim: str | None, heard: dict, heard_tone: int) -> str | None:
    """Name the unit the model reported, in the vocabulary the metrics expect."""
    if dim == "initial":
        return heard.get("initial") or None
    if dim == "final":
        return heard.get("final") or None
    if dim == "tone":
        return f"T{heard.get('tone') or heard_tone}"
    return None


def _tone_of_unit(unit: str) -> int:
    if unit and unit.startswith("T") and unit[1:].isdigit():
        return int(unit[1:])
    return 0
