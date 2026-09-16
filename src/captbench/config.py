"""Configuration: .env loading, API endpoints, model names, data paths.

Everything that a user is expected to supply lives in `.env` (see `.env.example`).
Nothing here ever prints or logs the API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_dotenv(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Parse a minimal KEY=VALUE .env file into os.environ.

    Deliberately dependency-free. Supports `#` comments, blank lines, and
    single/double quoted values. Existing environment variables win unless
    `override=True`.
    """
    path = path or DOTENV_PATH
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


@dataclass(frozen=True)
class Settings:
    """Runtime settings, resolved from environment variables."""

    api_key: str | None
    native_base: str
    compat_base: str
    tts_model: str
    tts_voice: str
    tts_teacher_voice: str
    omni_model: str
    data_dir: Path
    request_timeout: float

    # --- derived paths -------------------------------------------------
    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def ref_dir(self) -> Path:
        return self.data_dir / "audio" / "ref"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def calibration_path(self) -> Path:
        return self.data_dir / "calibration.json"

    @property
    def tts_url(self) -> str:
        return f"{self.native_base.rstrip('/')}/services/aigc/multimodal-generation/generation"

    @property
    def omni_url(self) -> str:
        return f"{self.compat_base.rstrip('/')}/chat/completions"

    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def require_api_key(self) -> str:
        if not self.api_key:
            raise MissingCredential(
                "DASHSCOPE_API_KEY is not set.\n"
                f"Add it to {DOTENV_PATH} (copy .env.example) or export it:\n"
                "    DASHSCOPE_API_KEY=sk-...\n"
                "Until then, use `--tts say` for a fully offline run "
                "(macOS local Mandarin voice, traditional scorer only)."
            )
        return self.api_key

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            api_key=os.environ.get("DASHSCOPE_API_KEY") or None,
            native_base=os.environ.get(
                "DASHSCOPE_NATIVE_BASE", "https://dashscope.aliyuncs.com/api/v1"
            ),
            compat_base=os.environ.get(
                "DASHSCOPE_COMPAT_BASE",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            tts_model=os.environ.get("QWEN_TTS_MODEL", "qwen3-tts-flash"),
            tts_voice=os.environ.get("QWEN_TTS_VOICE", "Cherry"),
            tts_teacher_voice=os.environ.get("QWEN_TTS_TEACHER_VOICE", "Ethan"),
            omni_model=os.environ.get("QWEN_OMNI_MODEL", "qwen3-omni-flash"),
            data_dir=Path(os.environ.get("CAPTBENCH_DATA_DIR", str(DEFAULT_DATA_DIR))),
            request_timeout=float(os.environ.get("CAPTBENCH_TIMEOUT", "120")),
        )


class MissingCredential(RuntimeError):
    """Raised when a cloud call is attempted without a configured API key."""


def get_settings() -> Settings:
    return Settings.from_env()
