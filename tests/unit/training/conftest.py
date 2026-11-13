from pathlib import Path

import pytest

from trailer_docking.evaluation.scenarios import (
    DEFAULT_DOCK_PATH,
    DEFAULT_VEHICLE_PATH,
    Scenario,
    StartDistribution,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def aligned_scenario() -> Scenario:
    """Start 4 m out, perfectly aligned: reversing straight docks."""

    return Scenario(
        name="aligned",
        vehicle_path=REPOSITORY_ROOT / DEFAULT_VEHICLE_PATH,
        dock_path=REPOSITORY_ROOT / DEFAULT_DOCK_PATH,
        start=StartDistribution((4, 4), (0, 0), (0, 0), (0, 0)),
    )
