import hashlib
import json
import shutil
from datetime import date

import pytest

from nyaya.config import ConfigError, load_settings


@pytest.fixture
def root(tmp_path, repo_root):
    (tmp_path / "config").mkdir()
    shutil.copy(repo_root / "config" / "settings.toml", tmp_path / "config" / "settings.toml")
    return tmp_path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "DATABASE_URL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "JUDGE_BASE_URL",
        "JUDGE_API_KEY",
        "JUDGE_MODEL",
        "OPENROUTER_API_KEY",
        "HF_HOME",
    ):
        monkeypatch.delenv(key, raising=False)


def test_repo_settings_load(repo_root):
    s = load_settings(repo_root, env_file=None)
    assert s.snapshot_date == date(2026, 11, 11)
    assert s.versions.config == "cfg1" or s.versions.config.startswith("cfg1-")
    assert s.retrieval.final_k == 8
    assert s.embed.dim == 1024
    assert s.embed.query_prefix.endswith("\nQuery: ")
    assert s.quote.min_words == 8 and s.quote.max_words == 40
    assert len(s.notices) == 4
    assert list(s.opinion.phrases) == ["dissenting", "concurring"]
    assert s.era_markers.d.min_count == 2 and len(s.era_markers.d.labels) == 5
    assert s.outcome_map.default == "other"
    assert "Appeal(s) allowed" in s.outcome_map.values["allowed"]


def test_bands_default_empty(root):
    s = load_settings(root, env_file=None)
    assert s.bands.era_c.left_x is None and s.bands.top_y is None
    assert s.versions.config == "cfg1"


def test_bands_json_merged_and_hashed(root):
    (root / "build").mkdir()
    bands = {"era_c": {"left_x": [20.0, 45.5], "right_x": [550.0, 580.0]}, "top_y": [0, 60]}
    raw = json.dumps(bands).encode()
    (root / "build" / "bands.json").write_bytes(raw)
    s = load_settings(root, env_file=None)
    assert s.bands.era_c.left_x == (20.0, 45.5)
    assert s.bands.era_c.right_x == (550.0, 580.0)
    assert s.bands.top_y == (0.0, 60.0)
    assert s.bands.bottom_y is None
    digest = hashlib.sha256(raw).hexdigest()[:12]
    assert s.versions.config == f"cfg1-{digest}"


def test_bad_band_length_names_key(root):
    (root / "build").mkdir()
    (root / "build" / "bands.json").write_text('{"top_y": [1, 2, 3]}')
    with pytest.raises(ConfigError, match=r"bands\.top_y"):
        load_settings(root, env_file=None)


def test_missing_key_is_named(root):
    path = root / "config" / "settings.toml"
    text = path.read_text(encoding="utf-8").replace("final_k = 8\n", "")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"retrieval\.final_k"):
        load_settings(root, env_file=None)


def test_invalid_value_is_named(root):
    path = root / "config" / "settings.toml"
    text = path.read_text(encoding="utf-8").replace("rrf_k = 60", 'rrf_k = "sixty"')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"retrieval\.rrf_k"):
        load_settings(root, env_file=None)


def test_unknown_key_is_named(root):
    path = root / "config" / "settings.toml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("[retrieval]\n", "[retrieval]\nfinal_kk = 3\n"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"retrieval\.final_kk"):
        load_settings(root, env_file=None)


def test_missing_settings_file(tmp_path):
    with pytest.raises(ConfigError, match=r"settings\.toml"):
        load_settings(tmp_path, env_file=None)


def test_secrets_from_env_file(root, monkeypatch):
    env = root / ".env"
    env.write_text("DATABASE_URL=postgresql://u:p@127.0.0.1:5432/db\nLLM_API_KEY=sk-test\n")
    monkeypatch.setenv("LLM_MODEL", "from-environment")
    s = load_settings(root, env_file=env)
    assert s.secrets.database_url == "postgresql://u:p@127.0.0.1:5432/db"
    assert s.secrets.llm_api_key is not None
    assert s.secrets.llm_api_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(s.secrets)
    assert s.secrets.llm_model == "from-environment"
    assert s.secrets.judge_model is None


def test_empty_env_values_are_none(root):
    env = root / ".env"
    env.write_text("DATABASE_URL=\nHF_HOME=\n")
    s = load_settings(root, env_file=env)
    assert s.secrets.database_url is None and s.secrets.hf_home is None
