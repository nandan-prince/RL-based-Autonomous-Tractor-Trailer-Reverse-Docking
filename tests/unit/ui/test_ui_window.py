"""The control window itself: tooltips and the obstacle editor.

These build the real Tk window, so they are skipped where no display exists.
"""

from __future__ import annotations

from collections.abc import Iterator
from math import radians
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trailer_docking.core.obstacles import Obstacle
from trailer_docking.ui.controller import (
    CONTROL_HELP,
    PARAMETER_HELP,
    PARAMETERS,
    AppController,
    ParameterError,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _repository_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)


def test_every_parameter_and_control_has_help_text() -> None:
    assert set(PARAMETER_HELP) == {spec.key for spec in PARAMETERS}
    for text in (*PARAMETER_HELP.values(), *CONTROL_HELP.values()):
        assert len(text) > 10 and text == text.strip()


def test_controller_sets_and_validates_obstacles(tmp_path: Path) -> None:
    controller = AppController(output_root=tmp_path)
    boxes = (Obstacle(10.0, 3.0, 1.0, 1.0), Obstacle(12.0, -3.0, 2.0, 0.5, yaw=0.4))
    controller.set_obstacles(boxes)
    assert controller.scenario.obstacles == boxes
    assert controller.built.obstacles == boxes
    with pytest.raises(ParameterError, match="docked vehicle would overlap"):
        controller.set_obstacles((*boxes, Obstacle(3.0, 0.0, 1.0, 1.0)))
    assert controller.scenario.obstacles == boxes
    controller.set_obstacles(())
    assert controller.scenario.obstacles == ()


@pytest.fixture
def window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    tk = pytest.importorskip("tkinter")
    pytest.importorskip("matplotlib")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"no display: {error}")
    from tkinter import messagebox

    from trailer_docking.ui.app import DockingApp

    errors: list[str] = []
    monkeypatch.setattr(
        messagebox, "showerror", lambda title, message, **_: errors.append(message)
    )
    app = DockingApp(root, AppController(output_root=tmp_path))
    app.errors = errors  # type: ignore[attr-defined]
    root.update()
    yield app
    app.close()


def _click(app: Any, x: float, y: float) -> None:
    app._on_view_click(SimpleNamespace(xdata=x, ydata=y, inaxes=app._view.axes))


def test_window_gives_every_captioned_control_a_tooltip(window: Any) -> None:
    from trailer_docking.ui.app import CAPTION_HELP, Tooltip, help_for_caption

    assert set(CAPTION_HELP.values()) <= set(CONTROL_HELP)
    # Every scenario parameter with a visible caption, plus the other controls.
    assert window.tooltips >= 2 * (len(PARAMETERS) - 1) + 25
    assert help_for_caption("Tractor wheelbase") == PARAMETER_HELP["tractor_wheelbase"]
    assert help_for_caption("Run test") == CONTROL_HELP["run_test"]
    assert help_for_caption("no such caption") is None

    tip = Tooltip(window._train_button, "Starts the runs")
    tip._show()
    window.root.update()
    assert tip._window is not None and tip._window.winfo_exists()
    tip._hide()
    assert tip._window is None


def test_backend_and_algorithm_controls_match_capabilities(
    window: Any,
) -> None:
    """Only PPO trains; both backends and all test methods stay selectable."""

    ttk = pytest.importorskip("tkinter.ttk")

    def descendants(widget: Any) -> list[Any]:
        children = list(widget.winfo_children())
        return children + [item for child in children for item in descendants(child)]

    radios = [
        widget
        for widget in descendants(window.root)
        if isinstance(widget, ttk.Radiobutton)
    ]
    captions = {widget.cget("text") for widget in radios}
    assert {"2D", "3D"} <= captions
    assert window._target_selector.cget("values") == (
        "Saved PPO model",
        "Loaded portable model",
        "NMPC",
        "LQR",
        "No-steering baseline",
    )
    window._training_backend.set("3d")
    assert window._training_backend.get() == "3d"
    assert window._method.get() == "ppo"


def test_training_controls_build_only_supported_ppo_config(
    window: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, bool]] = []
    monkeypatch.setattr(
        window.controller,
        "start_training",
        lambda config, show_3d=False: calls.append((config, show_3d)),
    )
    window._training_backend.set("3d")
    window._runs.set("2")
    window._timesteps.set("5000")
    window._training_seed.set("4")
    window._evaluation_episodes.set("0")
    window._ppo_vars["learning_rate"].set("0.0007")
    window._ppo_vars["n_steps"].set("128")
    window._ppo_vars["batch_size"].set("64")
    window._parameter_vars["reward_progress"].set("12.5")
    window._render_training_3d.set(True)
    window._on_start_training()

    config, show_3d = calls[0]
    assert config.method == "ppo"
    assert config.backend == "3d"
    assert config.runs == 2 and config.total_timesteps == 5000 and config.seed == 4
    assert config.learning_rate == pytest.approx(0.0007)
    assert config.n_steps == 128 and config.batch_size == 64
    assert window.controller.scenario.overrides.reward_progress == pytest.approx(12.5)
    assert config.evaluation_episodes == 0
    assert show_3d is True


@pytest.mark.parametrize(
    ("choice", "method", "baseline"),
    [
        ("No-steering baseline", "ppo", True),
        ("NMPC", "nmpc", False),
        ("LQR", "lqr", False),
    ],
)
@pytest.mark.parametrize("backend", ["2d", "3d"])
def test_test_controls_route_every_algorithm_backend_cell(
    window: Any,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
    method: str,
    baseline: bool,
    backend: str,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        window.controller,
        "start_test",
        lambda **options: calls.append(options),
    )
    window._evaluation_target.set(choice)
    window._test_backend.set(backend)
    window._episodes.set("2")
    window._test_seed.set("7")
    window._on_start_test()

    assert calls == [
        {
            "policy_path": None,
            "episodes": 2,
            "seed": 7,
            "baseline": baseline,
            "use_bundle": False,
            "method": method,
            "backend": backend,
            "show_3d": True,
        }
    ]


@pytest.mark.parametrize("backend", ["2d", "3d"])
def test_saved_ppo_control_routes_to_selected_backend(
    window: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        window.controller,
        "start_test",
        lambda **options: calls.append(options),
    )
    policy = tmp_path / "policy.zip"
    window._evaluation_target.set("Saved PPO model")
    window._policy_file.set(str(policy))
    window._test_backend.set(backend)
    window._on_start_test()

    assert calls[0]["policy_path"] == str(policy)
    assert calls[0]["method"] == "ppo"
    assert calls[0]["backend"] == backend
    assert calls[0]["baseline"] is False


def test_model_browser_discovers_and_selects_previous_checkpoint(window: Any) -> None:
    checkpoint = (
        window.controller.output_root
        / "artifacts"
        / "past-run"
        / "checkpoints"
        / "policy_final.zip"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    window._refresh_saved_models()

    rows = window._model_tree.get_children()
    assert len(rows) == 1
    assert tuple(window._model_tree.item(rows[0], "values"))[:2] == (
        "past-run",
        "policy_final.zip",
    )
    assert window._policy_file.get() == checkpoint.as_posix()
    assert window._evaluation_target.get() == "Saved PPO model"


def test_clicking_the_map_places_boxes_only_while_placing(window: Any) -> None:
    controller = window.controller
    _click(window, 10.0, 3.0)
    assert controller.scenario.obstacles == ()  # placing is off by default

    window._placing.set(True)
    window._obstacle_fields["Length"].set("2")
    window._obstacle_fields["Width"].set("0.5")
    window._obstacle_fields["Turn (deg)"].set("30")
    _click(window, 10.0, 3.0)
    _click(window, 12.26, -3.004)
    first, second = controller.scenario.obstacles
    assert first == Obstacle(10.0, 3.0, 2.0, 0.5, yaw=radians(30))
    assert (second.x, second.y) == (12.26, -3.0)  # rounded to centimetres
    assert window._obstacle_list.size() == 2
    assert "at (10, 3) m, 2 x 0.5 m, 30 deg" in window._obstacle_list.get(0)
    assert window._parameter_vars["obstacles"].get().startswith("10, 3, 2, 0.5")
    assert len(window._view._obstacle_patches) == 2  # drawn on the rebuilt view

    # A click that would block the dock is refused with a message.
    _click(window, 3.0, 0.0)
    assert len(controller.scenario.obstacles) == 2
    assert "docked vehicle would overlap" in window.errors[-1]
    # Clicks outside the plot are ignored.
    window._on_view_click(SimpleNamespace(xdata=None, ydata=None, inaxes=None))
    assert len(controller.scenario.obstacles) == 2


def test_boxes_can_be_typed_removed_and_cleared(window: Any) -> None:
    controller = window.controller
    window._on_add_obstacle()
    assert "enter a position" in window.errors[-1]
    window._obstacle_fields["x"].set("11")
    window._obstacle_fields["y"].set("-2.5")
    window._on_add_obstacle()
    window._obstacle_fields["y"].set("2.5")
    window._on_add_obstacle()
    assert [(b.x, b.y) for b in controller.scenario.obstacles] == [
        (11.0, -2.5),
        (11.0, 2.5),
    ]
    window._obstacle_fields["Length"].set("zero")
    window._on_add_obstacle()
    assert "Length" in window.errors[-1]
    assert len(controller.scenario.obstacles) == 2

    window._on_remove_obstacle()  # nothing selected: a hint, no change
    assert len(controller.scenario.obstacles) == 2
    window._obstacle_list.selection_set(0)
    window._on_remove_obstacle()
    assert [(b.x, b.y) for b in controller.scenario.obstacles] == [(11.0, 2.5)]
    assert window._obstacle_list.size() == 1
    window._on_clear_obstacles()
    assert controller.scenario.obstacles == ()
    assert window._parameter_vars["obstacles"].get() == ""

    # Loading a scenario with boxes fills the list too.
    controller.set_obstacles((Obstacle(10.0, 3.0, 1.0, 1.0),))
    window._sync_parameter_fields()
    assert window._obstacle_list.size() == 1


def test_top_down_view_maps_clicks_back_to_the_dock_frame() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from trailer_docking.core.state import Point2D
    from trailer_docking.evaluation.scenarios import Scenario
    from trailer_docking.visualization.topdown import TopDownView

    built = Scenario().build()
    view = TopDownView(built.vehicle, built.dock)
    point = Point2D(9.0, -2.0)
    shown = view._display_point(point)
    assert view.dock_point(shown.x, shown.y) == point
    plt.close(view.figure)
