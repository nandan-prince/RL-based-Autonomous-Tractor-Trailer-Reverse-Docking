import math
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from trailer_docking.core.config import (
    ConfigurationError,
    DockConfiguration,
    DockingTaskConfiguration,
    VehicleConfiguration,
    load_dock_configuration,
    load_docking_task_configuration,
    load_vehicle_configuration,
)


def _load_yaml(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(
    tmp_path: Path, data: dict[str, object], name: str = "config.yaml"
) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_nominal_vehicle_configuration_values(
    vehicle_config: VehicleConfiguration,
) -> None:
    assert vehicle_config.specification_id == "compact-semitrailer-v1"
    assert vehicle_config.schema_version == 1
    assert vehicle_config.tractor.wheelbase == pytest.approx(1.9)
    assert vehicle_config.trailer.hitch_to_axle == pytest.approx(4.0)
    assert vehicle_config.limits.steering_angle_abs_max == pytest.approx(0.5)
    assert vehicle_config.limits.steering_rate_abs_max == pytest.approx(0.3)
    assert vehicle_config.command.nominal_target_speed == pytest.approx(-0.2)
    assert vehicle_config.timing.kinematic_substeps_per_control == 5
    assert vehicle_config.timing.mujoco_substeps_per_control == 50
    assert len(vehicle_config.source_fingerprint) == 64
    assert len(vehicle_config.effective_fingerprint) == 64


def test_nominal_dock_configuration_values(dock_config: DockConfiguration) -> None:
    dock = dock_config.geometry
    assert dock.specification_id == "nominal-dock-v1"
    assert dock.interior_width == pytest.approx(1.5)
    assert dock.interior_depth == pytest.approx(6.0)
    assert dock.target_position.x == pytest.approx(0.25)
    assert dock.target_position.y == pytest.approx(0.0)
    assert dock.x_bounds.minimum == 0.0
    assert dock.y_bounds.maximum == pytest.approx(0.75)


def test_gymnasium_task_configuration_values(
    task_config: DockingTaskConfiguration,
) -> None:
    assert task_config.specification_id == "gymnasium-docking-v1"
    assert task_config.max_episode_steps == 600
    assert task_config.preview_distances == (0.5, 1.0, 2.0)
    assert task_config.reward_success == pytest.approx(100.0)
    assert task_config.source_path is not None
    assert len(task_config.source_fingerprint) == 64


def test_task_configuration_strictly_rejects_unknown_fields(
    tmp_path: Path, task_config: DockingTaskConfiguration
) -> None:
    assert task_config.source_path is not None
    data = _load_yaml(task_config.source_path)
    data["unknown"] = True
    with pytest.raises(ConfigurationError, match="unknown"):
        load_docking_task_configuration(_write_yaml(tmp_path, data))


def test_manifest_summaries_include_provenance(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> None:
    vehicle_manifest = vehicle_config.manifest_summary()
    dock_manifest = dock_config.manifest_summary()
    assert vehicle_manifest["specification_id"] == "compact-semitrailer-v1"
    assert vehicle_manifest["trailer_variant"] == "nominal"
    assert Path(vehicle_manifest["source_path"]).is_file()
    assert dock_manifest["specification_id"] == "nominal-dock-v1"
    assert Path(dock_manifest["source_path"]).is_file()


def test_new_specification_id_uses_same_schema_without_code_change(
    tmp_path: Path, vehicle_config: VehicleConfiguration
) -> None:
    data = _load_yaml(vehicle_config.source_path)
    data["specification_id"] = "highway-semitrailer-v1"
    changed = load_vehicle_configuration(_write_yaml(tmp_path, data))
    assert changed.specification_id == "highway-semitrailer-v1"
    assert changed.source_fingerprint != vehicle_config.source_fingerprint


def test_consistent_dimension_changes_propagate_to_derived_geometry(
    tmp_path: Path, vehicle_config: VehicleConfiguration
) -> None:
    data = _load_yaml(vehicle_config.source_path)
    data["specification_id"] = "custom-semitrailer-v1"
    data["tractor"]["wheelbase"] = 2.40  # type: ignore[index]
    data["tractor"]["collision_footprint_from_rear_axle"]["x"] = [  # type: ignore[index]
        -0.40,
        3.00,
    ]
    data["trailer"]["hitch_to_axle"] = 5.00  # type: ignore[index]
    data["trailer"]["collision_footprint_from_axle"]["x"] = [  # type: ignore[index]
        -1.00,
        5.50,
    ]
    data["trailer_variants"]["nominal"]["hitch_to_axle"] = 5.00  # type: ignore[index]
    data["neutral_pose_sites"]["tractor_front_axle"] = [2.40, 0.00, 0.30]  # type: ignore[index]
    data["neutral_pose_sites"]["trailer_axle"] = [-5.00, 0.00, 0.30]  # type: ignore[index]
    data["neutral_pose_sites"]["trailer_rear_face"] = [-6.00, 0.00, 1.65]  # type: ignore[index]
    data["neutral_pose_sites"]["docking_point"] = [-6.00, 0.00, 0.00]  # type: ignore[index]

    changed = load_vehicle_configuration(_write_yaml(tmp_path, data))
    state = changed.geometry.neutral_state()
    assert changed.tractor.body_length == pytest.approx(3.40)
    assert changed.trailer.body_length == pytest.approx(6.50)
    assert state.trailer_pose.x == pytest.approx(-5.00)
    assert changed.trailer.rear_face(state.trailer_pose).x == pytest.approx(-6.00)


def test_configurable_off_axle_hitch_updates_assembled_geometry(
    tmp_path: Path, vehicle_config: VehicleConfiguration
) -> None:
    data = _load_yaml(vehicle_config.source_path)
    data["specification_id"] = "off-axle-semitrailer-v1"
    data["tractor"]["hitch_offset"]["longitudinal"] = 0.25  # type: ignore[index]
    data["neutral_pose_sites"]["hitch"] = [0.25, 0.00, 0.75]  # type: ignore[index]
    data["neutral_pose_sites"]["trailer_axle"] = [-3.75, 0.00, 0.30]  # type: ignore[index]
    data["neutral_pose_sites"]["trailer_rear_face"] = [-4.75, 0.00, 1.65]  # type: ignore[index]
    data["neutral_pose_sites"]["docking_point"] = [-4.75, 0.00, 0.00]  # type: ignore[index]

    changed = load_vehicle_configuration(_write_yaml(tmp_path, data))
    state = changed.geometry.neutral_state()
    assert state.hitch_position.x == pytest.approx(0.25)
    assert state.trailer_pose.x == pytest.approx(-3.75)
    assert max(changed.geometry.constraint_residuals(state).values()) == pytest.approx(
        0.0
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update({"unknown": 1}), "unknown"),
        (lambda data: data.pop("tractor"), "missing"),
        (lambda data: data.update({"schema_version": 99}), "unsupported"),
        (
            lambda data: data["tractor"].update({"wheelbase": -1.0}),
            "greater than zero",
        ),
        (
            lambda data: data["limits"].update({"articulation_soft_abs_max": 1.0}),
            "soft < hard",
        ),
        (
            lambda data: data["timing"].update({"kinematic_integration_step": 0.03}),
            "integer multiple",
        ),
        (
            lambda data: data["command"].update(
                {"fields": ["steering_angle", "target_speed"]}
            ),
            "command fields",
        ),
        (
            lambda data: data["neutral_pose_sites"].update(
                {"trailer_axle": [-99.0, 0.0, 0.3]}
            ),
            "disagrees",
        ),
    ],
)
def test_vehicle_configuration_rejects_invalid_mutations(
    tmp_path: Path,
    vehicle_config: VehicleConfiguration,
    mutation: object,
    message: str,
) -> None:
    data = deepcopy(_load_yaml(vehicle_config.source_path))
    mutation(data)  # type: ignore[operator]
    with pytest.raises(ConfigurationError, match=message):
        load_vehicle_configuration(_write_yaml(tmp_path, data))


def test_configuration_rejects_nan_and_boolean_numbers(
    tmp_path: Path, vehicle_config: VehicleConfiguration
) -> None:
    data = _load_yaml(vehicle_config.source_path)
    data["tractor"]["wheelbase"] = math.nan  # type: ignore[index]
    with pytest.raises(ConfigurationError, match="finite"):
        load_vehicle_configuration(_write_yaml(tmp_path, data, "nan.yaml"))

    data = _load_yaml(vehicle_config.source_path)
    data["tractor"]["wheelbase"] = True  # type: ignore[index]
    with pytest.raises(ConfigurationError, match="real number"):
        load_vehicle_configuration(_write_yaml(tmp_path, data, "bool.yaml"))


def test_fingerprint_is_stable_and_changes_with_configuration(
    tmp_path: Path, vehicle_config: VehicleConfiguration
) -> None:
    reloaded = load_vehicle_configuration(vehicle_config.source_path)
    assert reloaded.source_fingerprint == vehicle_config.source_fingerprint

    data = _load_yaml(vehicle_config.source_path)
    data["mujoco"]["masses"]["trailer_body"] = 1190.0  # type: ignore[index]
    changed = load_vehicle_configuration(_write_yaml(tmp_path, data))
    assert changed.source_fingerprint != vehicle_config.source_fingerprint


def test_unknown_variant_and_missing_file_are_clear(
    tmp_path: Path, vehicle_config: VehicleConfiguration
) -> None:
    with pytest.raises(ConfigurationError, match="unknown trailer variant"):
        vehicle_config.with_trailer_variant("extra-long")
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_vehicle_configuration(tmp_path / "missing.yaml")


def test_dock_rejects_unknown_fields_and_incompatible_frame(
    tmp_path: Path, dock_config: DockConfiguration
) -> None:
    data = _load_yaml(dock_config.source_path)
    data["bay"]["mystery"] = 1  # type: ignore[index]
    with pytest.raises(ConfigurationError, match="unknown"):
        load_dock_configuration(_write_yaml(tmp_path, data, "unknown.yaml"))

    data = _load_yaml(dock_config.source_path)
    data["dock_frame"]["outward_axis"] = "-x"  # type: ignore[index]
    with pytest.raises(ConfigurationError, match="frame"):
        load_dock_configuration(_write_yaml(tmp_path, data, "frame.yaml"))


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("root: [unterminated", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid YAML"):
        load_vehicle_configuration(path)
