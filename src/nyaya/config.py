"""Settings: config/settings.toml, build/bands.json, and secrets from the environment.

No code writes config/settings.toml. `nyaya-build calibrate` writes
build/bands.json; when it exists, load_settings() merges it over the [bands]
defaults and suffixes versions.config with the first 12 hex digits of its
SHA-256, so new bands mean a new config version.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = Path("config") / "settings.toml"
BANDS_FILE = Path("build") / "bands.json"


class ConfigError(ValueError):
    """A missing or invalid setting. The message names the key."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _empty_band_to_none(v: Any) -> Any:
    if v is None or (isinstance(v, list | tuple) and len(v) == 0):
        return None
    return v


Band = Annotated[tuple[float, float] | None, BeforeValidator(_empty_band_to_none)]


class Paths(_Section):
    tar_root: str
    metadata_root: str
    build: str


class VersionsCfg(_Section):
    clean_text: str
    chunker: str
    config: str
    prompt: str


class Calibrate(_Section):
    sample_size: int
    min_era_c: int
    seed: int


class EraCBands(_Section):
    left_x: Band = None
    right_x: Band = None


class Bands(_Section):
    era_c: EraCBands = EraCBands()
    top_y: Band = None
    bottom_y: Band = None


class EraMarkersD(_Section):
    labels: list[str]
    min_count: int


class EraMarkersC(_Section):
    margin_line_regex: str
    min_margin_lines: int
    held_regex: str
    year_from: int
    year_to: int


class EraMarkers(_Section):
    d: EraMarkersD
    c: EraMarkersC


class Chunk(_Section):
    target_words: int
    min_words: int
    max_words: int
    abbreviations: list[str]


class OpinionCfg(_Section):
    # insertion order is priority order (dissenting before concurring)
    phrases: dict[str, list[str]]
    phrase_window_chars: int


class Acts(_Section):
    alias_file: str
    no_year_allowlist: list[str]


class Quote(_Section):
    min_words: int
    max_words: int


class Embed(_Section):
    model: str
    revision: str
    dim: int
    query_prefix: str


class Rerank(_Section):
    model: str
    revision: str
    instruction: str
    fallback_model: str


class Retrieval(_Section):
    keyword_k: int
    vector_k: int
    rrf_k: int
    rerank_candidates: int
    final_k: int
    ef_search: int
    expand_rank_offset: float
    lookup_max_editorial: int


class Refusal(_Section):
    threshold_qwen: float
    threshold_cohere: float
    max_wrong_refusal_share: float


class Llm(_Section):
    max_input_tokens: int
    max_output_tokens: int
    repair_min_seconds_left: int


class Timeouts(_Section):
    request_s: int
    llm_s: int
    answer_deadline_s: int


class Load(_Section):
    smoke_min_hits: int


class G6(_Section):
    min_paragraph_integrity: float
    min_size_compliance: float
    holdout_size: int
    holdout_seed: int


class G4(_Section):
    max_drop_points: float


class Devset(_Section):
    size: int
    handcheck_size: int
    unanswerable_size: int
    max_shared_words: int
    seed: int


class Judge(_Section):
    min_kappa: float


class Bm25Trigger(_Section):
    min_gap_points: float


class EvalCfg(_Section):
    bootstrap_samples: int
    seed: int


class Latency(_Section):
    p95_target_s: float


class Lawyer(_Section):
    sample_size: int
    seed: int


class Build(_Section):
    max_error_share: float


class EncoderCheck(_Section):
    min_cosine: float


class OutcomeMap(_Section):
    default: str
    values: dict[str, list[str]]


class Secrets(BaseSettings):
    """Secrets from the environment and `.env`. Empty values count as unset."""

    model_config = SettingsConfigDict(
        extra="ignore", env_ignore_empty=True, case_sensitive=False, frozen=True
    )

    database_url: str | None = None
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    judge_base_url: str | None = None
    judge_api_key: SecretStr | None = None
    judge_model: str | None = None
    openrouter_api_key: SecretStr | None = None
    hf_home: str | None = None


class Settings(_Section):
    snapshot_date: date
    paths: Paths
    versions: VersionsCfg
    calibrate: Calibrate
    bands: Bands = Bands()
    running_head_patterns: list[str]
    era_markers: EraMarkers
    chunk: Chunk
    opinion: OpinionCfg
    acts: Acts
    quote: Quote
    embed: Embed
    rerank: Rerank
    retrieval: Retrieval
    refusal: Refusal
    llm: Llm
    timeouts: Timeouts
    load: Load
    g6: G6
    g4: G4
    devset: Devset
    judge: Judge
    bm25_trigger: Bm25Trigger
    eval: EvalCfg
    latency: Latency
    lawyer: Lawyer
    build: Build
    encoder_check: EncoderCheck
    outcome_map: OutcomeMap
    notices: list[str] = Field(min_length=1)
    secrets: Secrets = Field(default_factory=Secrets)

    @field_validator("notices")
    @classmethod
    def _notices_not_blank(cls, v: list[str]) -> list[str]:
        if any(not n.strip() for n in v):
            raise ValueError("notices must not be blank")
        return v


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _format_errors(exc: ValidationError, prefix: str = "") -> str:
    lines = []
    for err in exc.errors():
        key = ".".join(str(p) for p in err["loc"])
        if err["type"] == "missing":
            lines.append(f"{prefix}{key}: missing")
        else:
            lines.append(f"{prefix}{key}: {err['msg']}")
    return "invalid settings:\n  " + "\n  ".join(lines)


_DEFAULT_ENV = object()


def load_settings(
    root: Path | str | None = None,
    env_file: Path | str | object | None = _DEFAULT_ENV,
) -> Settings:
    """Read, merge and validate all settings.

    `root` is the repository root (default: this checkout). `env_file` defaults
    to `<root>/.env`; pass None to read secrets from the process environment only.
    """
    root = Path(root) if root is not None else REPO_ROOT
    settings_path = root / SETTINGS_FILE
    try:
        raw: dict[str, Any] = tomllib.loads(settings_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"settings file not found: {settings_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{settings_path}: {exc}") from exc

    bands_path = root / BANDS_FILE
    if bands_path.exists():
        blob = bands_path.read_bytes()
        try:
            bands = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{bands_path}: {exc}") from exc
        if not isinstance(bands, dict):
            raise ConfigError(f"{bands_path}: bands: expected a JSON object")
        raw["bands"] = _deep_merge(raw.get("bands", {}), bands)
        digest = hashlib.sha256(blob).hexdigest()[:12]
        versions = raw.get("versions")
        if isinstance(versions, dict) and isinstance(versions.get("config"), str):
            versions["config"] = f"{versions['config']}-{digest}"

    if env_file is _DEFAULT_ENV:
        env_file = root / ".env"
    try:
        secrets = Secrets(_env_file=env_file)  # type: ignore[call-arg]
        return Settings.model_validate({**raw, "secrets": secrets})
    except ValidationError as exc:
        raise ConfigError(_format_errors(exc)) from None
