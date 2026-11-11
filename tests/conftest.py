from pathlib import Path

import pytest

from trailer_docking.core.config import (
    DockConfiguration,
    VehicleConfiguration,
    load_dock_configuration,
    load_vehicle_configuration,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VEHICLE_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "vehicles" / "compact_semitrailer_v1.yaml"
)
DOCK_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "environments" / "nominal_dock_v1.yaml"


@pytest.fixture(scope="session")
def vehicle_config() -> VehicleConfiguration:
    return load_vehicle_configuration(VEHICLE_CONFIG_PATH)


@pytest.fixture(scope="session")
def dock_config() -> DockConfiguration:
    return load_dock_configuration(DOCK_CONFIG_PATH)
