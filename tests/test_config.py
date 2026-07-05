"""YAML config loading and the auto-loaded default config."""

from pathlib import Path

import pytest

from gnosis.settings import Settings

# Non-flag settings this suite needs to construct a valid Settings.
_REQUIRED = {
    "GNOSIS_TOKEN": "x",
    "GNOSIS_READ_OPERATOR_TOKEN": "x",
    "GNOSIS_EXPORT_OPERATOR_TOKEN": "x",
    "GNOSIS_WRITE_OPERATOR_TOKEN": "x",
    "GNOSIS_ADMIN_OPERATOR_TOKEN": "x",
    "NEO4J_URI": "bolt://x",
    "NEO4J_USERNAME": "x",
    "NEO4J_PASSWORD": "x",
    "LITELLM_BASE_URL": "x",
    "LITELLM_API_KEY": "x",
    "GNOSIS_LLM": "openai/gpt-5.5",
}

_RUN_18_FLAGS = (
    "gnosis_fact_extraction_enabled",
    "gnosis_entity_graph_enabled",
    "gnosis_adaptive_routing_enabled",
    "gnosis_chain_of_note_enabled",
)


def _base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _REQUIRED.items():
        monkeypatch.setenv(key, value)


def test_auto_loads_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unset GNOSIS_CONFIG_FILE -> the shipped configs/default.yaml auto-loads.
    _base_env(monkeypatch)
    monkeypatch.delenv("GNOSIS_CONFIG_FILE", raising=False)
    settings = Settings()
    for flag in _RUN_18_FLAGS:
        assert getattr(settings, flag) is True, flag
    # Only the Run 18 stack; rejected / unmeasured levers stay off.
    assert settings.gnosis_hybrid_retrieval_enabled is False
    assert settings.gnosis_rerank_enabled is False
    assert settings.gnosis_read_supersession_enabled is False
    assert settings.gnosis_coverage_budget_multiplier == 1


def test_empty_config_opts_out_to_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("GNOSIS_CONFIG_FILE", "")
    settings = Settings()
    for flag in _RUN_18_FLAGS:
        assert getattr(settings, flag) is False, flag


def test_explicit_config_file_loads_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _base_env(monkeypatch)
    config = tmp_path / "c.yaml"
    yaml_lines = (
        "gnosis_fact_extraction_enabled: true",
        "gnosis_adaptive_routing_enabled: true",
    )
    _ = config.write_text("\n".join(yaml_lines) + "\n")
    monkeypatch.setenv("GNOSIS_CONFIG_FILE", str(config))
    settings = Settings()
    assert settings.gnosis_fact_extraction_enabled is True
    assert settings.gnosis_adaptive_routing_enabled is True
    # A flag the config omits keeps the safe default.
    assert settings.gnosis_chain_of_note_enabled is False


def test_env_overrides_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _base_env(monkeypatch)
    config = tmp_path / "c.yaml"
    _ = config.write_text("gnosis_fact_extraction_enabled: true\n")
    monkeypatch.setenv("GNOSIS_CONFIG_FILE", str(config))
    monkeypatch.setenv("GNOSIS_FACT_EXTRACTION_ENABLED", "false")
    assert Settings().gnosis_fact_extraction_enabled is False


def test_shipped_default_is_the_run_18_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _base_env(monkeypatch)
    default = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    monkeypatch.setenv("GNOSIS_CONFIG_FILE", str(default))
    settings = Settings()
    for flag in _RUN_18_FLAGS:
        assert getattr(settings, flag) is True, flag
    assert settings.gnosis_hybrid_retrieval_enabled is False
