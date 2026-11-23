"""Versioned, strict configuration for the classical control baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from trailer_docking.controllers.lqr import LQRConfiguration
from trailer_docking.controllers.nmpc import NMPCConfiguration

CURRENT_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTROLLER_CONFIG_PATH = Path(
    "configs/controllers/baseline_controllers_v1.yaml"
)


def _strict_dataclass(data: Any, kind: type[Any], name: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{name} configuration must be a mapping")
    expected = {item.name for item in fields(kind)}
    if set(data) != expected:
        missing = sorted(expected - set(data))
        unknown = sorted(set(data) - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ValueError(f"invalid {name} fields: {', '.join(details)}")
    try:
        return kind(**data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name} configuration: {error}") from error


@dataclass(frozen=True, slots=True)
class ControllerConfigurations:
    """Named LQR/NMPC settings used by one experimental condition."""

    specification_id: str = "baseline-controllers-v1"
    schema_version: int = CURRENT_SCHEMA_VERSION
    lqr: LQRConfiguration = field(default_factory=LQRConfiguration)
    nmpc: NMPCConfiguration = field(default_factory=NMPCConfiguration)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.specification_id, str)
            or not self.specification_id.strip()
        ):
            raise ValueError("controller specification_id cannot be empty")
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError("unsupported controller-configuration schema version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "specification_id": self.specification_id,
            "schema_version": self.schema_version,
            "lqr": asdict(self.lqr),
            "nmpc": asdict(self.nmpc),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ControllerConfigurations:
        if not isinstance(data, dict):
            raise ValueError("controller configuration must be a mapping")
        expected = {"specification_id", "schema_version", "lqr", "nmpc"}
        if set(data) != expected:
            raise ValueError("controller configuration fields do not match the schema")
        return cls(
            specification_id=data["specification_id"],
            schema_version=data["schema_version"],
            lqr=_strict_dataclass(data["lqr"], LQRConfiguration, "LQR"),
            nmpc=_strict_dataclass(data["nmpc"], NMPCConfiguration, "NMPC"),
        )

    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_CONTROLLER_CONFIG_PATH
    ) -> ControllerConfigurations:
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = REPOSITORY_ROOT / source
        try:
            content = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError(
                f"cannot read controller configuration {source}: {error}"
            ) from error
        return cls.from_dict(content)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )
        return target

    @property
    def fingerprint(self) -> str:
        serialized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(serialized.encode("utf-8")).hexdigest()
