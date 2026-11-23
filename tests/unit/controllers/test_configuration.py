"""Versioned controller configuration loading and provenance."""

from __future__ import annotations

from dataclasses import replace

import pytest

from trailer_docking.controllers.configuration import (
    DEFAULT_CONTROLLER_CONFIG_PATH,
    ControllerConfigurations,
)


def test_default_file_matches_controller_defaults() -> None:
    loaded = ControllerConfigurations.load(DEFAULT_CONTROLLER_CONFIG_PATH)
    defaults = ControllerConfigurations()
    assert loaded == defaults
    assert loaded.fingerprint == defaults.fingerprint
    assert len(loaded.fingerprint) == 64


def test_controller_configuration_round_trip(tmp_path) -> None:
    chosen = replace(
        ControllerConfigurations(),
        specification_id="test-v1",
        nmpc=replace(ControllerConfigurations().nmpc, horizon_steps=12),
    )
    path = chosen.save(tmp_path / "controllers.yaml")
    assert ControllerConfigurations.load(path) == chosen


def test_controller_configuration_is_strict() -> None:
    data = ControllerConfigurations().to_dict()
    data["nmpc"]["mystery"] = 1
    with pytest.raises(ValueError, match="unknown"):
        ControllerConfigurations.from_dict(data)
    data = ControllerConfigurations().to_dict()
    del data["lqr"]["lateral_weight"]
    with pytest.raises(ValueError, match="missing"):
        ControllerConfigurations.from_dict(data)
