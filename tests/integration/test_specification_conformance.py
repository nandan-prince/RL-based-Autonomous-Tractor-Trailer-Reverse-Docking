import ast
from pathlib import Path

from trailer_docking.core.config import DockConfiguration, VehicleConfiguration
from trailer_docking.validation.vehicle_spec import main, run_conformance_checks


def test_complete_specification_conformance(
    vehicle_config: VehicleConfiguration, dock_config: DockConfiguration
) -> None:
    report = run_conformance_checks(vehicle_config, dock_config)
    assert report.passed
    assert all(check.passed for check in report.checks)
    assert report.to_dict()["passed"] is True


def test_conformance_cli_human_and_json_output(
    capsys: object,
    vehicle_config: VehicleConfiguration,
    dock_config: DockConfiguration,
) -> None:
    assert (
        main(
            [
                "--vehicle",
                str(vehicle_config.source_path),
                "--dock",
                str(dock_config.source_path),
            ]
        )
        == 0
    )
    human = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Overall: PASS" in human
    assert "rigid_hitch_constraint" in human

    assert (
        main(
            [
                "--vehicle",
                str(vehicle_config.source_path),
                "--dock",
                str(dock_config.source_path),
                "--json",
            ]
        )
        == 0
    )
    json_output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert '"passed": true' in json_output


def test_core_has_no_framework_or_simulator_imports() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "trailer_docking" / "core"
    banned = {"gymnasium", "mujoco", "stable_baselines3", "casadi", "torch"}
    imports: set[str] = set()
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    assert imports.isdisjoint(banned)
