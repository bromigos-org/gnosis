"""Opt the test suite out of the auto-loaded default config.

gnosis auto-loads ``configs/default.yaml`` (the Run 18 stack) when
``GNOSIS_CONFIG_FILE`` is unset, so out of the box it runs its best config.
Tests exercise features in isolation over the safe code-default baseline, so
disable the auto-load here (empty string = opt out). A test that needs a config
sets ``GNOSIS_CONFIG_FILE`` itself.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNOSIS_CONFIG_FILE", "")
