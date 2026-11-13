from pathlib import Path

import pytest

from trailer_docking.core.config import (
    DockConfiguration,
    DockingTaskConfiguration,
    VehicleConfiguration,
    load_dock_configuration,
    load_docking_task_configuration,
    load_vehicle_configuration,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VEHICLE_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "vehicles" / "compact_semitrailer_v1.yaml"
)
DOCK_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "environments" / "nominal_dock_v1.yaml"
TASK_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "environments" / "gymnasium_docking_v1.yaml"
)


@pytest.fixture(scope="session")
def vehicle_config() -> VehicleConfiguration:
    return load_vehicle_configuration(VEHICLE_CONFIG_PATH)


@pytest.fixture(scope="session")
def dock_config() -> DockConfiguration:
    return load_dock_configuration(DOCK_CONFIG_PATH)


@pytest.fixture(scope="session")
def task_config() -> DockingTaskConfiguration:
    return load_docking_task_configuration(TASK_CONFIG_PATH)


@pytest.fixture(scope="session")
def policy_checkpoint() -> bytes:
    """Bytes of a valid, untrained PPO checkpoint for the 21-value observation.

    It holds real network weights in the layout the framework saves, so
    bundle and loading code can be exercised without training.
    """

    pytest.importorskip("stable_baselines3")
    import io
    import zipfile

    import numpy as np
    import torch
    from gymnasium import spaces
    from stable_baselines3.common.policies import ActorCriticPolicy

    torch.manual_seed(0)
    policy = ActorCriticPolicy(
        spaces.Box(-1.0, 1.0, (21,), dtype=np.float32),
        spaces.Box(-1.0, 1.0, (1,), dtype=np.float32),
        lambda _: 0.0,
    )
    weights = io.BytesIO()
    torch.save(policy.state_dict(), weights)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("policy.pth", weights.getvalue())
    return archive.getvalue()
