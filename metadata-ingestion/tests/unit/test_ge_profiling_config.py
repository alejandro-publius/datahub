import logging

import pytest

from datahub.configuration.common import ConfigurationWarning
from datahub.ingestion.source.ge_profiling_config import GEProfilingConfig


def test_profile_table_level_only():
    config = GEProfilingConfig.model_validate(
        {"enabled": True, "profile_table_level_only": True}
    )
    assert config.any_field_level_metrics_enabled() is False

    config = GEProfilingConfig.model_validate(
        {
            "enabled": True,
            "profile_table_level_only": True,
            "include_field_max_value": False,
        }
    )
    assert config.any_field_level_metrics_enabled() is False


def test_profile_table_level_only_fails_with_field_metric_enabled():
    with pytest.raises(
        ValueError,
        match="Cannot enable field-level metrics if profile_table_level_only is set",
    ):
        GEProfilingConfig.model_validate(
            {
                "enabled": True,
                "profile_table_level_only": True,
                "include_field_max_value": True,
            }
        )


def test_profiling_method_field_removed() -> None:
    # `method` was removed together with the Great Expectations profiler.
    # SQLAlchemy is the only SQL profiler; recipes that still set `method` are
    # accepted (the field is dropped) with a deprecation warning.
    with pytest.warns(ConfigurationWarning, match="method was removed"):
        config = GEProfilingConfig.model_validate({"enabled": True, "method": "ge"})
    assert not hasattr(config, "method")


def test_profile_use_autocommit_default_on(monkeypatch):
    # No recipe key, no env var → the built-in default (True) via default_factory.
    monkeypatch.delenv("DATAHUB_PROFILE_USE_AUTOCOMMIT", raising=False)
    config = GEProfilingConfig.model_validate({"enabled": True})
    assert config.profile_use_autocommit is True


def test_profile_use_autocommit_recipe_false_disables(monkeypatch):
    # An explicit recipe value beats the env var and the built-in default.
    monkeypatch.delenv("DATAHUB_PROFILE_USE_AUTOCOMMIT", raising=False)
    config = GEProfilingConfig.model_validate(
        {"enabled": True, "profile_use_autocommit": False}
    )
    assert config.profile_use_autocommit is False


def test_profile_use_autocommit_env_false_disables(monkeypatch):
    # With no recipe key, the env var supplies the default via default_factory.
    monkeypatch.setenv("DATAHUB_PROFILE_USE_AUTOCOMMIT", "false")
    config = GEProfilingConfig.model_validate({"enabled": True})
    assert config.profile_use_autocommit is False


def test_profile_use_autocommit_recipe_true_beats_env_false(monkeypatch):
    # The most valuable case: recipe true wins over env false. Replacing
    # default_factory with validator-based merging would silently break this.
    monkeypatch.setenv("DATAHUB_PROFILE_USE_AUTOCOMMIT", "false")
    config = GEProfilingConfig.model_validate(
        {"enabled": True, "profile_use_autocommit": True}
    )
    assert config.profile_use_autocommit is True


def test_profile_use_autocommit_malformed_env_falls_back(monkeypatch, caplog):
    # A malformed fleet-wide setting must not prevent ingestion from starting
    # and must not silently flip the flag: it falls back to the default with
    # a warning naming the variable and the value.
    monkeypatch.setenv("DATAHUB_PROFILE_USE_AUTOCOMMIT", "banana")
    with caplog.at_level(
        logging.WARNING, logger="datahub.ingestion.source.ge_profiling_config"
    ):
        config = GEProfilingConfig.model_validate({"enabled": True})
    assert config.profile_use_autocommit is True
    assert any(
        "DATAHUB_PROFILE_USE_AUTOCOMMIT" in record.getMessage()
        and "banana" in record.getMessage()
        for record in caplog.records
    )
