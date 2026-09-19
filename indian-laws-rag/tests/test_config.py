"""The env-var-driven backend selection that makes one image run anywhere."""

import importlib

import pytest

import app.core.config


def reload_config(monkeypatch, **env):
    """Re-import config with a specific environment, as a fresh process would."""
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    return importlib.reload(app.core.config)


@pytest.fixture(autouse=True)
def restore_config():
    yield
    importlib.reload(app.core.config)


# --- vector store -----------------------------------------------------------


def test_qdrant_is_embedded_when_no_url_is_given(monkeypatch):
    config = reload_config(monkeypatch, QDRANT_URL=None)

    assert config.settings.uses_qdrant_server is False


def test_a_qdrant_url_selects_the_server_backend(monkeypatch):
    config = reload_config(monkeypatch, QDRANT_URL="http://qdrant:6333")

    assert config.settings.uses_qdrant_server is True
    assert config.settings.qdrant_url == "http://qdrant:6333"


# --- memory -----------------------------------------------------------------


def test_memory_is_sqlite_when_no_database_url_is_given(monkeypatch):
    config = reload_config(monkeypatch, DATABASE_URL=None)

    assert config.settings.uses_postgres is False


def test_a_postgres_url_selects_the_postgres_backend(monkeypatch):
    config = reload_config(monkeypatch, DATABASE_URL="postgresql://user:pw@db:5432/laws")

    assert config.settings.uses_postgres is True


def test_a_non_postgres_database_url_does_not_select_postgres(monkeypatch):
    config = reload_config(monkeypatch, DATABASE_URL="mysql://db/laws")

    assert config.settings.uses_postgres is False


# --- auth -------------------------------------------------------------------


def test_auth_is_off_when_no_api_key_is_set(monkeypatch):
    config = reload_config(monkeypatch, API_KEY=None)

    assert config.settings.auth_required is False


def test_auth_is_on_once_an_api_key_is_set(monkeypatch):
    config = reload_config(monkeypatch, API_KEY="a-real-key")

    assert config.settings.auth_required is True


def test_a_placeholder_api_key_does_not_enable_auth(monkeypatch):
    # Half-configured is worse than off: it would reject every request.
    config = reload_config(monkeypatch, API_KEY="paste_your_key_here")

    assert config.settings.auth_required is False


# --- parsing ----------------------------------------------------------------


def test_cors_origins_are_parsed_from_a_comma_separated_list(monkeypatch):
    config = reload_config(monkeypatch, CORS_ORIGINS="https://a.test, https://b.test")

    assert config.settings.cors_origins == ["https://a.test", "https://b.test"]


def test_cors_defaults_to_allowing_any_origin(monkeypatch):
    config = reload_config(monkeypatch, CORS_ORIGINS=None)

    assert config.settings.cors_origins == ["*"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("1", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ],
)
def test_boolean_env_vars_accept_common_spellings(monkeypatch, value, expected):
    config = reload_config(monkeypatch, RATE_LIMIT_ENABLED=value)

    assert config.settings.rate_limit_enabled is expected


def test_an_invalid_numeric_setting_falls_back_to_its_default(monkeypatch):
    # A typo in an env var should not stop the service from starting.
    config = reload_config(monkeypatch, RERANK_TOP_N="not-a-number")

    assert config.settings.rerank_top_n == 6


def test_numeric_tunables_are_read_from_the_environment(monkeypatch):
    config = reload_config(monkeypatch, RERANK_TOP_N="10", CONFIDENCE_THRESHOLD="0.5")

    assert config.settings.rerank_top_n == 10
    assert config.settings.confidence_threshold == 0.5
