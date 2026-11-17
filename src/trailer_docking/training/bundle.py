"""Portable trained-policy bundles: export, load, and replay in a fresh sim.

A bundle is one ``.zip`` file holding everything needed to run a trained
policy somewhere else::

    bundle.yaml    scenario, training settings, provenance, evaluation summary
    policy.zip     the Stable-Baselines3 PPO checkpoint

The scenario travels inside the bundle, so loading it rebuilds the same
vehicle, dock, task, and start distribution the policy was trained on, on
either simulator backend.  The frozen base specifications are referenced by
path and checked by fingerprint rather than copied, so a bundle is portable
between checkouts of this repository as long as its scenario uses
repository-relative specification paths (the default).

Loading never unpickles anything: the metadata is plain YAML read with the
safe loader, and only the network weights are read from the checkpoint (see
:func:`~trailer_docking.training.evaluate.policy_from_bytes`).
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from trailer_docking.core.state import VehicleState
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.training.config import TrainingConfig, validate_backend
from trailer_docking.training.evaluate import (
    EpisodeResult,
    Policy,
    PolicyFormatError,
    policy_from_bytes,
    policy_observation_size,
    run_docking_test,
)

BUNDLE_FORMAT = "trailer-docking-policy-bundle"
BUNDLE_VERSION = 1
BUNDLE_SUFFIX = ".dockpolicy.zip"
METADATA_MEMBER = "bundle.yaml"
POLICY_MEMBER = "policy.zip"
BUNDLE_NOTE = (
    "Trained with untuned reward weights and start ranges. The evaluation "
    "numbers stored here are pipeline output, not research results, unless "
    "the source run is registered as an experiment."
)


class BundleError(ValueError):
    """Raised when a file is not a usable policy bundle."""


@dataclass(frozen=True, slots=True)
class PolicyBundle:
    """A loaded bundle; the policy network is built only when asked for."""

    path: Path
    metadata: Mapping[str, Any]
    scenario: Scenario
    training: TrainingConfig | None
    policy_bytes: bytes

    @property
    def fingerprint_matches(self) -> bool:
        """Whether today's base specifications match those used in training."""

        return bool(
            self.metadata.get("scenario_fingerprint") == self.scenario.fingerprint
        )

    @property
    def observation_size(self) -> int:
        return int(self.metadata["observation_size"])

    def load_policy(self) -> Policy:
        """Build the deterministic policy (needs PyTorch and SB3)."""

        return policy_from_bytes(self.policy_bytes)

    @property
    def max_episode_steps(self) -> int | None:
        """Step limit the policy was trained with, when that is known."""

        return None if self.training is None else self.training.max_episode_steps

    def description(self) -> str:
        source = self.metadata.get("source")
        if not isinstance(source, dict):
            source = {}
        label = (
            source.get("campaign_id") or source.get("experiment_id") or "unknown run"
        )
        where = (
            "training settings unknown"
            if self.training is None
            else f"trained in {self.training.backend}"
        )
        return f"{self.path.name} ({label}, {where})"


def _observation_size(scenario: Scenario) -> int:
    return 15 + 2 * len(scenario.build().task.preview_distances)


def export_bundle(
    policy_path: str | Path,
    scenario: Scenario,
    destination: str | Path,
    *,
    training: TrainingConfig | None = None,
    source: Mapping[str, Any] | None = None,
    evaluation: Sequence[Mapping[str, Any]] = (),
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write a bundle for one saved policy and return its path.

    ``training`` is the configuration the policy was trained with; leave it
    out when that is not known, and the bundle records it as unknown rather
    than inventing defaults.  ``source`` identifies the run (experiment or
    campaign ID, seed), ``evaluation`` is a list of per-backend summaries, and
    ``provenance`` carries package versions and the Git revision.  The
    observation size is read from the policy itself and must match the
    scenario.  An existing file at *destination* is never overwritten.
    """

    checkpoint = Path(policy_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"policy file not found: {checkpoint}")
    target = Path(destination)
    if target.is_dir():
        target = target / f"{scenario.name}{BUNDLE_SUFFIX}"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    content = checkpoint.read_bytes()
    try:
        observation_size = policy_observation_size(content)
    except PolicyFormatError as error:
        raise BundleError(f"{checkpoint}: {error}") from error
    if observation_size != _observation_size(scenario):
        raise BundleError(
            f"the policy expects {observation_size} observation values but the"
            f" scenario's task provides {_observation_size(scenario)}"
        )
    metadata = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_VERSION,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": dict(source or {}),
        "scenario": scenario.to_dict(),
        "scenario_fingerprint": scenario.fingerprint,
        "training": None if training is None else training.to_dict(),
        "observation_size": observation_size,
        "action_size": 1,
        "evaluation": [dict(item) for item in evaluation],
        "provenance": dict(provenance or {}),
        "note": BUNDLE_NOTE,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(METADATA_MEMBER, yaml.safe_dump(metadata, sort_keys=False))
        archive.writestr(POLICY_MEMBER, content)
    return target


def load_bundle(path: str | Path) -> PolicyBundle:
    """Read and validate a bundle without building the policy network."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"bundle not found: {source}")
    try:
        with zipfile.ZipFile(source) as archive:
            metadata = yaml.safe_load(archive.read(METADATA_MEMBER).decode("utf-8"))
            policy_bytes = archive.read(POLICY_MEMBER)
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise BundleError(f"{source} is not a policy bundle: {error}") from error
    if not isinstance(metadata, dict) or metadata.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"{source} is not a policy bundle")
    if metadata.get("format_version") != BUNDLE_VERSION:
        raise BundleError(
            f"{source} has unsupported bundle version"
            f" {metadata.get('format_version')!r}"
        )
    try:
        scenario = Scenario.from_dict(metadata["scenario"])
        scenario.build()
        saved = metadata.get("training")
        training: TrainingConfig | None = None
        if saved is not None:
            if not isinstance(saved, dict):
                raise ValueError("training must be a mapping")
            # A later version may have added settings this one does not know.
            known = {item.name for item in fields(TrainingConfig)}
            training = TrainingConfig.from_dict(
                {key: value for key, value in saved.items() if key in known}
            )
        int(metadata["observation_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise BundleError(f"{source} has invalid contents: {error}") from error
    return PolicyBundle(
        path=source,
        metadata=metadata,
        scenario=scenario,
        training=training,
        policy_bytes=policy_bytes,
    )


def run_bundle(
    bundle: PolicyBundle | str | Path,
    *,
    backend: str = "2d",
    episodes: int = 1,
    seed: int = 0,
    scenario: Scenario | None = None,
    on_state: Callable[[VehicleState], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[EpisodeResult]:
    """Run a bundled policy in a freshly built simulator.

    The bundle's own scenario is used unless *scenario* is given, for example
    to test the policy on a different truck or dock.  The observation layout
    must match the one the policy was trained with.
    """

    loaded = bundle if isinstance(bundle, PolicyBundle) else load_bundle(bundle)
    chosen = scenario if scenario is not None else loaded.scenario
    if _observation_size(chosen) != loaded.observation_size:
        raise BundleError(
            "the scenario's observation has"
            f" {_observation_size(chosen)} values but the policy expects"
            f" {loaded.observation_size}"
        )
    return run_docking_test(
        chosen,
        loaded.load_policy(),
        episodes=episodes,
        seed=seed,
        max_episode_steps=loaded.max_episode_steps,
        backend=validate_backend(backend),
        on_state=on_state,
        should_stop=should_stop,
    )
