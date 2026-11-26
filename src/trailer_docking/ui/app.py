"""Tkinter control window: edit a scenario, train, and run docking tests.

Run from the repository root::

    python -m trailer_docking.ui.app

The left column holds three tabs (``Scenario``, ``Training``, ``Docking
test``); the right side shows the live top-down view of whichever of training
or testing is running.  This module is only the view: every decision is made
by :class:`~trailer_docking.ui.controller.AppController`, which the window
polls about fifteen times a second.

``--smoke-test SECONDS`` opens the window, closes it again after that many
seconds, and exits with status 0; it exists for automated verification.
"""

from __future__ import annotations

import argparse
import tkinter as tk
from collections.abc import Callable, Sequence
from math import degrees, isfinite, radians
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from trailer_docking.core.config import ConfigurationError
from trailer_docking.core.obstacles import Obstacle
from trailer_docking.core.state import VehicleState
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.ui.controller import (
    BACKENDS,
    CONTROL_HELP,
    GROUPS,
    PARAMETER_HELP,
    PARAMETERS,
    PHYSICS_GROUP,
    PHYSICS_GROUP_NOTE,
    AppController,
    ControllerError,
    ParameterSpec,
    format_obstacles,
    parse_obstacles,
)

REFRESH_MILLISECONDS = 66  # about 15 Hz
POLICY_LATEST, POLICY_BASELINE, POLICY_FILE = "latest", "baseline", "file"
POLICY_BUNDLE, POLICY_NMPC, POLICY_LQR = "bundle", "nmpc", "lqr"
BUNDLE_FILETYPES = (
    ("Policy bundle", "*.dockpolicy.zip"),
    ("All files", "*.*"),
)
SCENARIO_FILETYPES = (("Scenario YAML", "*.yaml *.yml"), ("All files", "*.*"))
POLICY_FILETYPES = (("Saved policy", "*.zip"), ("All files", "*.*"))

PPO_FIELDS = (
    ("learning_rate", "Learning rate", "0.0003"),
    ("n_steps", "Rollout steps", "1024"),
    ("batch_size", "Batch size", "256"),
    ("n_epochs", "Epochs per rollout", "10"),
    ("gamma", "Discount (gamma)", "0.99"),
    ("gae_lambda", "GAE lambda", "0.95"),
    ("clip_range", "Policy clip range", "0.2"),
    ("clip_range_vf", "Value clip range (blank = off)", ""),
    ("entropy_coefficient", "Entropy coefficient", "0.0"),
    ("value_coefficient", "Value-loss coefficient", "0.5"),
    ("max_grad_norm", "Maximum gradient norm", "0.5"),
    ("target_kl", "Target KL (blank = off)", ""),
)

TRAINING_REWARD_KEYS = (
    "path_lateral_weight",
    "path_heading_weight",
    "reward_progress",
    "reward_time",
    "reward_lateral_improvement",
    "reward_lateral_deviation_multiplier",
    "reward_heading_improvement",
    "reward_articulation",
    "reward_steering_rate",
    "reward_success",
    "reward_collision",
    "reward_jackknife",
    "reward_out_of_bounds",
)


TOOLTIP_DELAY_MILLISECONDS = 450
# Help for controls that are not scenario parameters, found by their caption.
CAPTION_HELP = {
    "Algorithm": "method",
    "Simulator": "training_backend",
    "Number of runs": "runs",
    "Timesteps per run": "timesteps",
    "Experiment name": "purpose",
    "Timesteps": "timesteps",
    "Training seed": "training_seed",
    "Independent runs": "runs",
    "Episode step limit": "max_episode_steps",
    "Save every N steps": "checkpoint_interval",
    "Device": "device",
    "Evaluate after training": "evaluation_episodes",
    "Learning rate": "learning_rate",
    "Rollout steps": "n_steps",
    "Batch size": "batch_size",
    "Epochs per rollout": "n_epochs",
    "Discount (gamma)": "gamma",
    "GAE lambda": "gae_lambda",
    "Policy clip range": "clip_range",
    "Value clip range (blank = off)": "clip_range_vf",
    "Entropy coefficient": "entropy_coefficient",
    "Value-loss coefficient": "value_coefficient",
    "Maximum gradient norm": "max_grad_norm",
    "Target KL (blank = off)": "target_kl",
    "Start training": "start",
    "Save portable model\N{HORIZONTAL ELLIPSIS}": "export",
    "Controller": "evaluation_controller",
    "Refresh": "refresh_models",
    "Open checkpoint\N{HORIZONTAL ELLIPSIS}": "policy_file",
    "Open portable model\N{HORIZONTAL ELLIPSIS}": "load_result",
    "Render live MuJoCo frames (slow)": "render_3d",
    "Render live MuJoCo frames": "render_3d",
    "Seed of first run": "training_seed",
    "Test episodes per run": "evaluation_episodes",
    "Test each run in:": "evaluation_episodes",
    "2D": "evaluate_2d",
    "3D": "evaluate_3d",
    "Start": "start",
    "Stop": "stop",
    "Export result\N{HORIZONTAL ELLIPSIS}": "export",
    "Episodes": "episodes",
    "Seed": "test_seed",
    "Latest training checkpoint": "policy_latest",
    "Loaded result bundle": "policy_bundle",
    "Untrained straight-reverse baseline": "policy_baseline",
    "NMPC controller": "policy_nmpc",
    "Saved policy file (.zip):": "policy_file",
    "Browse\N{HORIZONTAL ELLIPSIS}": "policy_file",
    "Run test": "run_test",
    "Load result\N{HORIZONTAL ELLIPSIS}": "load_result",
    "Show planned route": "show_route",
    "Render 3D runs (slower training)": "render_3d",
    "Apply": "apply",
    "Save scenario\N{HORIZONTAL ELLIPSIS}": "save_scenario",
    "Load scenario\N{HORIZONTAL ELLIPSIS}": "load_scenario",
    "Click the map to place": "obstacle_place",
    "Add at x, y": "obstacle_add",
    "Remove selected": "obstacle_remove",
    "Clear all": "obstacle_clear",
    "Length": "obstacle_size",
    "Width": "obstacle_size",
    "Height": "obstacle_size",
    "Turn (deg)": "obstacle_size",
    "x": "obstacle_add",
    "y": "obstacle_add",
}


class Tooltip:
    """A small help window shown while the pointer rests on a widget."""

    def __init__(self, widget: tk.Misc, text: str) -> None:
        self.widget = widget
        self.text = text
        self._after: str | None = None
        self._window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: object = None) -> None:
        self._cancel()
        self._after = self.widget.after(TOOLTIP_DELAY_MILLISECONDS, self._show)

    def _cancel(self) -> None:
        if self._after is not None:
            self.widget.after_cancel(self._after)
            self._after = None

    def _show(self) -> None:
        self._after = None
        if self._window is not None:
            return
        window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(
            f"+{self.widget.winfo_rootx() + 16}"
            f"+{self.widget.winfo_rooty() + self.widget.winfo_height() + 4}"
        )
        tk.Label(
            window,
            text=self.text,
            justify=tk.LEFT,
            wraplength=320,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            padx=6,
            pady=4,
        ).pack()
        self._window = window

    def _hide(self, _event: object = None) -> None:
        self._cancel()
        if self._window is not None:
            self._window.destroy()
            self._window = None


def help_for_caption(caption: str) -> str | None:
    """Help text for a control with this caption, if it has any."""

    for spec in PARAMETERS:
        if spec.label == caption:
            return PARAMETER_HELP.get(spec.key)
    key = CAPTION_HELP.get(caption)
    return None if key is None else CONTROL_HELP.get(key)


def attach_tooltips(widget: tk.Misc) -> int:
    """Give every captioned control under *widget* its help text.

    A caption's help also covers the entry or list on the same grid row, so
    the pointer can rest on either.  Returns how many tooltips were made.
    """

    made = 0
    for child in widget.winfo_children():
        made += attach_tooltips(child)
        try:
            caption = str(child.cget("text"))
        except tk.TclError:
            continue
        text = help_for_caption(caption)
        if text is None:
            continue
        Tooltip(child, text)
        made += 1
        info = child.grid_info()
        if not info:
            continue
        for sibling in widget.grid_slaves(row=int(info["row"])):
            if sibling is not child and not isinstance(sibling, ttk.Label):
                Tooltip(sibling, text)
                made += 1
    return made


def _whole_number(text: str, label: str, *, minimum: int) -> int:
    try:
        value = int(text.strip().replace("_", ""))
    except ValueError:
        raise ControllerError(f"{label}: '{text}' is not a whole number") from None
    if value < minimum:
        raise ControllerError(f"{label} must be at least {minimum}")
    return value


def _number(text: str, label: str, *, minimum: float | None = None) -> float:
    try:
        value = float(text.strip())
    except ValueError:
        raise ControllerError(f"{label}: '{text}' is not a number") from None
    if minimum is not None and value < minimum:
        raise ControllerError(f"{label} must be at least {minimum:g}")
    return value


def _optional_number(
    text: str, label: str, *, minimum: float | None = None
) -> float | None:
    return None if not text.strip() else _number(text, label, minimum=minimum)


class DockingApp:
    """The main window; a thin view over an :class:`AppController`."""

    def __init__(
        self,
        root: tk.Tk,
        controller: AppController,
        *,
        timesteps: int = 200_000,
        learning_rate: float | None = None,
        purpose: str = "ppo-ui",
    ) -> None:
        self.root = root
        self.controller = controller
        self._closed = False
        self._after_id: str | None = None
        self._view: Any = None
        self._canvas: Any = None
        self._shown_state: VehicleState | None = None
        self._shown_route: object = ()
        self._idle_route: tuple[Any, ...] = ()
        self._shown_frame: object = None
        self._photo: Any = None
        self._raw_3d_frame: Any = None
        self._rendered_3d_size: tuple[int, int] | None = None
        self._resize_3d_after_id: str | None = None
        self._shown_results: tuple[str, ...] = ()
        self._parameter_vars: dict[str, tk.StringVar] = {}
        self._saved_policy_paths: dict[str, Path] = {}
        self._last_model_refresh_checkpoint: str | None = None

        root.title("Tractor-trailer docking \N{EM DASH} training control")
        window_width = min(1400, max(root.winfo_screenwidth() - 80, 900))
        window_height = min(850, max(root.winfo_screenheight() - 120, 620))
        root.geometry(f"{window_width}x{window_height}")
        root.minsize(min(1100, window_width), min(620, window_height))
        root.protocol("WM_DELETE_WINDOW", self.close)

        panes = ttk.Frame(root, padding=6)
        panes.pack(fill=tk.BOTH, expand=True)
        notebook = ttk.Notebook(panes)
        notebook.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(panes)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self._build_scenario_tab(notebook)
        self._build_training_tab(
            notebook, timesteps=timesteps, learning_rate=learning_rate, purpose=purpose
        )
        self._build_test_tab(notebook)
        # Scenario construction initializes variables shared by the training
        # reward editor, but the user-facing order follows the three normal
        # tasks: train, inspect/evaluate models, then advanced scene editing.
        notebook.insert(0, self._training_tab)
        notebook.insert(1, self._test_tab)
        notebook.select(self._training_tab)

        toolbar = ttk.Frame(right)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self._show_route = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toolbar,
            text="Show planned route",
            variable=self._show_route,
            command=self._on_route_toggled,
        ).pack(side=tk.LEFT)
        self._status_bar = tk.StringVar(value="Ready")
        ttk.Label(
            right, textvariable=self._status_bar, relief=tk.SUNKEN, anchor=tk.W
        ).pack(side=tk.BOTTOM, fill=tk.X)
        self._displays = ttk.Notebook(right)
        self._displays.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._view_frame = ttk.Frame(self._displays)
        self._displays.add(self._view_frame, text="Top-down (2D and 3D runs)")
        self._frame_3d = ttk.Frame(self._displays)
        self._displays.add(self._frame_3d, text="3D view (MuJoCo)")
        self._image_3d = ttk.Label(
            self._frame_3d,
            anchor=tk.CENTER,
            text="Select this tab to preview the scene in 3D.\n"
            "Runs on the 3D simulator are shown here while they run.",
        )
        self._image_3d.pack(fill=tk.BOTH, expand=True)
        self._frame_3d.bind("<Configure>", self._on_3d_panel_resized)
        self._displays.bind("<<NotebookTabChanged>>", self._on_display_changed)

        self._rebuild_view()
        self.tooltips = attach_tooltips(root)
        self._tick()

    # ---------------------------------------------------------------- tabs

    def _ui_parameter_values(self) -> dict[str, str]:
        """Scenario values with effective reward defaults made explicit."""

        values = self.controller.parameter_values()
        task = self.controller.built.task
        reward_defaults = {
            "path_lateral_weight": task.reward_lateral_error,
            "path_heading_weight": task.reward_heading_error,
            "reward_progress": task.reward_progress,
            "reward_time": task.reward_time,
            "reward_lateral_improvement": task.reward_lateral_improvement,
            "reward_lateral_deviation_multiplier": (
                task.reward_lateral_deviation_multiplier
            ),
            "reward_heading_improvement": task.reward_heading_improvement,
            "reward_articulation": task.reward_articulation,
            "reward_steering_rate": task.reward_steering_rate,
            "reward_success": task.reward_success,
            "reward_collision": task.reward_collision,
            "reward_jackknife": task.reward_jackknife,
            "reward_out_of_bounds": task.reward_out_of_bounds,
        }
        for key, value in reward_defaults.items():
            if not values[key]:
                values[key] = f"{value:g}"
        return values

    def _build_scenario_tab(self, notebook: ttk.Notebook) -> None:
        outer = ttk.Frame(notebook)
        self._scenario_tab = outer
        notebook.add(outer, text="3  Advanced scenario")
        # The parameter groups are taller than the window, so they scroll.
        canvas = tk.Canvas(outer, highlightthickness=0, width=350)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tab = ttk.Frame(canvas, padding=8)
        window = canvas.create_window((0, 0), window=tab, anchor=tk.NW)
        tab.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>", lambda event: canvas.itemconfigure(window, width=event.width)
        )
        values = self._ui_parameter_values()
        for group in GROUPS:
            frame = ttk.LabelFrame(tab, text=group, padding=6)
            frame.pack(fill=tk.X, pady=(0, 6))
            frame.columnconfigure(1, weight=1)
            row = 0
            if group == PHYSICS_GROUP:
                ttk.Label(frame, text=PHYSICS_GROUP_NOTE, foreground="#a04000").grid(
                    row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 4)
                )
                row += 1
            for spec in PARAMETERS:
                if spec.group != group:
                    continue
                variable = tk.StringVar(value=values[spec.key])
                self._parameter_vars[spec.key] = variable
                if spec.key == "obstacles":
                    # Boxes get their own editor; the text stays the value
                    # that Apply, Save, and Load work with.
                    self._build_obstacle_editor(frame, variable)
                    continue
                ttk.Label(frame, text=spec.label).grid(
                    row=row, column=0, sticky=tk.W, padx=(0, 6), pady=1
                )
                self._parameter_widget(frame, spec, variable).grid(
                    row=row, column=1, sticky=tk.EW, pady=1
                )
                ttk.Label(frame, text=spec.unit).grid(
                    row=row, column=2, sticky=tk.W, padx=(4, 0)
                )
                row += 1
        ttk.Label(
            tab,
            text="Blank = value from the base specification\n"
            "(blank randomization = off).\n"
            "Ranges are 'minimum, maximum'.",
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(0, 6))
        buttons = ttk.Frame(tab)
        buttons.pack(fill=tk.X)
        self._apply_button = ttk.Button(buttons, text="Apply", command=self._on_apply)
        self._apply_button.pack(side=tk.LEFT)
        ttk.Button(
            buttons, text="Save scenario\N{HORIZONTAL ELLIPSIS}", command=self._on_save
        ).pack(side=tk.LEFT, padx=4)
        self._load_button = ttk.Button(
            buttons, text="Load scenario\N{HORIZONTAL ELLIPSIS}", command=self._on_load
        )
        self._load_button.pack(side=tk.LEFT)

    def _build_obstacle_editor(
        self, frame: ttk.LabelFrame, variable: tk.StringVar
    ) -> None:
        """List of boxes with click-to-place, typed placement, and removal."""

        self._obstacle_text = variable
        ttk.Label(
            frame,
            text="Tick the box below, then click the top-down view.",
            foreground="#555555",
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W)
        self._placing = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, text="Click the map to place", variable=self._placing
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(2, 4))
        self._obstacle_fields: dict[str, tk.StringVar] = {}
        for index, (name, default) in enumerate(
            (("Length", "1"), ("Width", "1"), ("Height", "1.5"), ("Turn (deg)", "0"))
        ):
            field = tk.StringVar(value=default)
            self._obstacle_fields[name] = field
            ttk.Label(frame, text=name).grid(
                row=2 + index // 2, column=2 * (index % 2), sticky=tk.W, padx=(0, 4)
            )
            ttk.Entry(frame, textvariable=field, width=6).grid(
                row=2 + index // 2, column=2 * (index % 2) + 1, sticky=tk.W, pady=1
            )
        typed = ttk.Frame(frame)
        typed.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))
        for name in ("x", "y"):
            field = tk.StringVar()
            self._obstacle_fields[name] = field
            ttk.Label(typed, text=name).pack(side=tk.LEFT)
            ttk.Entry(typed, textvariable=field, width=6).pack(
                side=tk.LEFT, padx=(2, 6)
            )
        ttk.Button(typed, text="Add at x, y", command=self._on_add_obstacle).pack(
            side=tk.LEFT
        )
        self._obstacle_list = tk.Listbox(frame, height=4, exportselection=False)
        self._obstacle_list.grid(row=5, column=0, columnspan=4, sticky=tk.EW, pady=4)
        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=4, sticky=tk.W)
        ttk.Button(
            buttons, text="Remove selected", command=self._on_remove_obstacle
        ).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Clear all", command=self._on_clear_obstacles).pack(
            side=tk.LEFT, padx=4
        )
        variable.trace_add("write", lambda *_: self._refresh_obstacle_list())
        self._refresh_obstacle_list()

    def _refresh_obstacle_list(self) -> None:
        """Show one readable line per box in the obstacle text."""

        try:
            boxes = parse_obstacles(self._obstacle_text.get())
        except ControllerError:
            return
        self._obstacle_list.delete(0, tk.END)
        for index, box in enumerate(boxes, start=1):
            self._obstacle_list.insert(
                tk.END,
                f"{index}: at ({box.x:g}, {box.y:g}) m, "
                f"{box.length:g} x {box.width:g} m, {degrees(box.yaw):g} deg",
            )

    def _new_obstacle(self, x: float, y: float) -> Obstacle:
        """A box at *x*, *y* with the size typed in the editor."""

        def number(name: str) -> float:
            text = self._obstacle_fields[name].get().strip()
            try:
                return float(text)
            except ValueError:
                raise ControllerError(f"{name}: '{text}' is not a number") from None

        try:
            return Obstacle(
                x,
                y,
                number("Length"),
                number("Width"),
                yaw=radians(number("Turn (deg)")),
                height=number("Height"),
            )
        except ValueError as error:
            raise ControllerError(str(error)) from None

    def _set_obstacles(self, boxes: tuple[Obstacle, ...]) -> None:
        """Apply a new list of boxes; a refused list changes nothing."""

        def apply() -> None:
            self.controller.set_obstacles(boxes)
            self._obstacle_text.set(format_obstacles(boxes))
            self._rebuild_view()

        self._guard(apply)()

    def _current_obstacles(self) -> tuple[Obstacle, ...]:
        return tuple(self.controller.scenario.obstacles)

    def _on_add_obstacle(self) -> None:
        def add() -> None:
            position = []
            for name in ("x", "y"):
                text = self._obstacle_fields[name].get().strip()
                try:
                    position.append(float(text))
                except ValueError:
                    raise ControllerError(
                        f"{name}: enter a position in metres"
                    ) from None
            box = self._new_obstacle(position[0], position[1])
            self._set_obstacles((*self._current_obstacles(), box))

        self._guard(add)()

    def _on_remove_obstacle(self) -> None:
        chosen = self._obstacle_list.curselection()
        if not chosen:
            self._status_bar.set("Select a box in the list first")
            return
        boxes = list(self._current_obstacles())
        if chosen[0] < len(boxes):
            del boxes[chosen[0]]
            self._set_obstacles(tuple(boxes))

    def _on_clear_obstacles(self) -> None:
        self._set_obstacles(())

    def _on_view_click(self, event: Any) -> None:
        """Place a box where the top-down view was clicked."""

        if (
            not self._placing.get()
            or event.xdata is None
            or event.ydata is None
            or event.inaxes is not self._view.axes
        ):
            return
        if self.controller.busy:
            self._status_bar.set("Boxes cannot be placed while a run is active")
            return

        def place() -> None:
            point = self._view.dock_point(float(event.xdata), float(event.ydata))
            box = self._new_obstacle(round(point.x, 2), round(point.y, 2))
            self._set_obstacles((*self._current_obstacles(), box))

        self._guard(place)()

    @staticmethod
    def _parameter_widget(
        parent: tk.Misc, spec: ParameterSpec, variable: tk.StringVar
    ) -> tk.Widget:
        if spec.kind == "choice":
            return ttk.Combobox(
                parent,
                textvariable=variable,
                values=spec.choices,
                state="readonly",
                width=14,
            )
        return ttk.Entry(parent, textvariable=variable, width=16)

    def _build_training_tab(
        self,
        notebook: ttk.Notebook,
        *,
        timesteps: int,
        learning_rate: float | None,
        purpose: str,
    ) -> None:
        tab = ttk.Frame(notebook, padding=8)
        self._training_tab = tab
        notebook.add(tab, text="1  Train PPO")
        ttk.Label(
            tab,
            text="Configure one reproducible PPO run. Models are saved automatically.",
            foreground="#444444",
            wraplength=330,
        ).pack(anchor=tk.W, pady=(0, 6))

        if self.controller.initial_policy is not None:
            ttk.Label(
                tab,
                text=f"Warm start: {self.controller.initial_policy.name}",
                wraplength=330,
            ).pack(anchor=tk.W, pady=(0, 6))

        settings = ttk.Notebook(tab)
        settings.pack(fill=tk.X)
        setup = ttk.Frame(settings, padding=6)
        ppo = ttk.Frame(settings, padding=6)
        rewards = ttk.Frame(settings, padding=6)
        settings.add(setup, text="Run")
        settings.add(ppo, text="PPO")
        settings.add(rewards, text="Rewards")

        form = setup
        form.columnconfigure(1, weight=1)
        self._method = tk.StringVar(value="ppo")
        self._training_backend = tk.StringVar(value=BACKENDS[0])
        self._runs = tk.StringVar(value="1")
        self._timesteps = tk.StringVar(value=str(timesteps))
        self._training_seed = tk.StringVar(value="0")
        self._purpose = tk.StringVar(value=purpose)
        self._max_episode_steps = tk.StringVar(
            value=str(max(900, self.controller.built.task.max_episode_steps))
        )
        self._checkpoint_interval = tk.StringVar(value="20000")
        self._device = tk.StringVar(value="cpu")
        self._evaluation_episodes = tk.StringVar(value="20")
        self._evaluate_2d = tk.BooleanVar(value=True)
        self._evaluate_3d = tk.BooleanVar(value=False)
        self._render_training_3d = tk.BooleanVar(value=False)
        # Compatibility alias for callers of the former shared render switch.
        self._render_3d = self._render_training_3d
        ttk.Label(form, text="Simulator").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 6)
        )
        selector = ttk.Frame(form)
        selector.grid(row=0, column=1, sticky=tk.EW, pady=1)
        self._training_backend_buttons: dict[str, ttk.Radiobutton] = {}
        for column, backend in enumerate(BACKENDS):
            selector.columnconfigure(column, weight=1)
            button = ttk.Radiobutton(
                selector,
                text=backend.upper(),
                variable=self._training_backend,
                value=backend,
            )
            button.grid(row=0, column=column, sticky=tk.W, padx=(0, 12))
            self._training_backend_buttons[backend] = button
        entries: tuple[tuple[str, tk.StringVar, str], ...] = (
            ("Experiment name", self._purpose, "entry"),
            ("Timesteps", self._timesteps, "entry"),
            ("Training seed", self._training_seed, "entry"),
            ("Independent runs", self._runs, "entry"),
            ("Episode step limit", self._max_episode_steps, "entry"),
            ("Save every N steps", self._checkpoint_interval, "entry"),
            ("Device", self._device, "device"),
            ("Evaluate after training", self._evaluation_episodes, "entry"),
        )
        self._training_entries: list[tk.Widget] = []
        for row, (label, variable, kind) in enumerate(entries, start=1):
            ttk.Label(form, text=label).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 6)
            )
            if kind == "device":
                entry: tk.Widget = ttk.Combobox(
                    form,
                    textvariable=variable,
                    values=("cpu", "auto", "mps", "cuda"),
                    state="readonly",
                    width=12,
                )
            else:
                entry = ttk.Entry(form, textvariable=variable, width=14)
            entry.grid(row=row, column=1, sticky=tk.EW, pady=1)
            self._training_entries.append(entry)
        tests = ttk.Frame(form)
        tests.grid(
            row=len(entries) + 1,
            column=0,
            columnspan=2,
            sticky=tk.W,
            pady=(4, 0),
        )
        ttk.Label(tests, text="Evaluation simulator:").pack(side=tk.LEFT)
        self._evaluation_buttons: list[ttk.Checkbutton] = []
        for text, test_variable in (
            ("2D", self._evaluate_2d),
            ("3D", self._evaluate_3d),
        ):
            evaluation_button = ttk.Checkbutton(
                tests, text=text, variable=test_variable
            )
            evaluation_button.pack(side=tk.LEFT, padx=(6, 0))
            self._evaluation_buttons.append(evaluation_button)
        self._training_render_button = ttk.Checkbutton(
            form,
            text="Render live MuJoCo frames (slow)",
            variable=self._render_training_3d,
        )
        self._training_render_button.grid(
            row=len(entries) + 2, column=0, columnspan=2, sticky=tk.W, pady=(3, 0)
        )

        ppo.columnconfigure(1, weight=1)
        self._ppo_vars: dict[str, tk.StringVar] = {}
        self._normalize_advantage = tk.BooleanVar(value=True)
        for row, (key, label, default) in enumerate(PPO_FIELDS):
            value = (
                str(learning_rate)
                if key == "learning_rate" and learning_rate is not None
                else default
            )
            variable = tk.StringVar(value=value)
            self._ppo_vars[key] = variable
            ttk.Label(ppo, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 6))
            entry = ttk.Entry(ppo, textvariable=variable, width=14)
            entry.grid(row=row, column=1, sticky=tk.EW, pady=1)
            self._training_entries.append(entry)
        normalize = ttk.Checkbutton(
            ppo,
            text="Normalize advantages",
            variable=self._normalize_advantage,
        )
        normalize.grid(row=len(PPO_FIELDS), column=0, columnspan=2, sticky=tk.W)
        self._training_entries.append(normalize)

        rewards.columnconfigure(1, weight=1)
        self._reward_entries: list[ttk.Entry] = []
        specs = {spec.key: spec for spec in PARAMETERS}
        for row, key in enumerate(TRAINING_REWARD_KEYS):
            spec = specs[key]
            ttk.Label(rewards, text=spec.label).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 6)
            )
            entry = ttk.Entry(rewards, textvariable=self._parameter_vars[key], width=14)
            entry.grid(row=row, column=1, sticky=tk.EW, pady=1)
            self._reward_entries.append(entry)
        ttk.Label(
            rewards,
            text="These are the effective coefficients saved with the run.",
            foreground="#555555",
        ).grid(row=len(TRAINING_REWARD_KEYS), column=0, columnspan=2, sticky=tk.W)

        buttons = ttk.Frame(tab)
        buttons.pack(fill=tk.X, pady=6)
        self._train_button = ttk.Button(
            buttons, text="Start training", command=self._on_start_training
        )
        self._train_button.pack(side=tk.LEFT)
        self._train_stop_button = ttk.Button(
            buttons, text="Stop", command=self._guard(self.controller.stop_training)
        )
        self._train_stop_button.pack(side=tk.LEFT, padx=4)
        self._export_button = ttk.Button(
            buttons,
            text="Save portable model\N{HORIZONTAL ELLIPSIS}",
            command=self._on_export,
        )
        self._export_button.pack(side=tk.LEFT)
        self._progress = ttk.Progressbar(tab, mode="determinate", maximum=1.0)
        self._progress.pack(fill=tk.X)
        self._training_text = tk.StringVar()
        ttk.Label(
            tab,
            textvariable=self._training_text,
            justify=tk.LEFT,
            anchor=tk.NW,
            wraplength=320,
        ).pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    def _build_test_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=8)
        self._test_tab = tab
        notebook.add(tab, text="2  Models & Evaluate")

        models = ttk.LabelFrame(tab, text="Saved PPO models", padding=6)
        models.pack(fill=tk.BOTH, expand=True)
        self._model_tree = ttk.Treeview(
            models,
            columns=("run", "checkpoint", "sim", "status"),
            show="headings",
            height=7,
            selectmode="browse",
        )
        self._model_tree.heading("run", text="Experiment")
        self._model_tree.heading("checkpoint", text="Checkpoint")
        self._model_tree.heading("sim", text="Sim")
        self._model_tree.heading("status", text="Status")
        self._model_tree.column("run", width=145, stretch=True)
        self._model_tree.column("checkpoint", width=95, stretch=False)
        self._model_tree.column("sim", width=38, stretch=False)
        self._model_tree.column("status", width=60, stretch=False)
        self._model_tree.pack(fill=tk.BOTH, expand=True)
        self._model_tree.bind("<<TreeviewSelect>>", self._on_model_selected)
        model_buttons = ttk.Frame(models)
        model_buttons.pack(fill=tk.X, pady=(4, 0))
        self._refresh_models_button = ttk.Button(
            model_buttons, text="Refresh", command=self._refresh_saved_models
        )
        self._refresh_models_button.pack(side=tk.LEFT)
        self._browse_button = ttk.Button(
            model_buttons,
            text="Open checkpoint\N{HORIZONTAL ELLIPSIS}",
            command=self._on_browse,
        )
        self._browse_button.pack(side=tk.LEFT, padx=4)
        self._load_result_button = ttk.Button(
            model_buttons,
            text="Open portable model\N{HORIZONTAL ELLIPSIS}",
            command=self._on_load_result,
        )
        self._load_result_button.pack(side=tk.LEFT)

        self._policy_file = tk.StringVar()
        self._bundle_text = tk.StringVar(value="No model selected")
        ttk.Label(
            models,
            textvariable=self._bundle_text,
            foreground="#555555",
            wraplength=330,
        ).pack(anchor=tk.W, pady=(4, 0))

        evaluation = ttk.LabelFrame(tab, text="Evaluate", padding=6)
        evaluation.pack(fill=tk.X, pady=(6, 0))
        evaluation.columnconfigure(1, weight=1)
        form = evaluation
        form.columnconfigure(1, weight=1)
        self._episodes = tk.StringVar(value="1")
        self._test_seed = tk.StringVar(value="0")
        self._test_backend = tk.StringVar(value=BACKENDS[0])
        self._evaluation_target = tk.StringVar(value="Saved PPO model")
        self._policy_choice = tk.StringVar(value=POLICY_FILE)
        ttk.Label(form, text="Controller").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 6)
        )
        self._target_selector = ttk.Combobox(
            form,
            textvariable=self._evaluation_target,
            values=(
                "Saved PPO model",
                "Loaded portable model",
                "NMPC",
                "LQR",
                "No-steering baseline",
            ),
            state="readonly",
        )
        self._target_selector.grid(row=0, column=1, sticky=tk.EW, pady=1)
        ttk.Label(form, text="Simulator").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 6)
        )
        backend_selector = ttk.Frame(form)
        backend_selector.grid(row=1, column=1, sticky=tk.EW, pady=1)
        self._test_backend_buttons: dict[str, ttk.Radiobutton] = {}
        for column, backend in enumerate(BACKENDS):
            backend_selector.columnconfigure(column, weight=1)
            button = ttk.Radiobutton(
                backend_selector,
                text=backend.upper(),
                variable=self._test_backend,
                value=backend,
            )
            button.grid(row=0, column=column, sticky=tk.W, padx=(0, 12))
            self._test_backend_buttons[backend] = button
        self._test_entries: list[ttk.Entry] = []
        for row, (label, variable) in enumerate(
            (("Episodes", self._episodes), ("Seed", self._test_seed)), start=2
        ):
            ttk.Label(form, text=label).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 6)
            )
            entry = ttk.Entry(form, textvariable=variable, width=14)
            entry.grid(row=row, column=1, sticky=tk.EW, pady=1)
            self._test_entries.append(entry)
        self._render_test_3d = tk.BooleanVar(value=True)
        self._test_render_button = ttk.Checkbutton(
            form, text="Render live MuJoCo frames", variable=self._render_test_3d
        )
        self._test_render_button.grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=(3, 0)
        )
        # Kept for the refresh-state code; selection is now one clear combobox.
        self._policy_buttons: dict[str, ttk.Radiobutton] = {}
        self._policy_file_entry = ttk.Entry(form, textvariable=self._policy_file)

        buttons = ttk.Frame(tab)
        buttons.pack(fill=tk.X)
        self._test_button = ttk.Button(
            buttons, text="Run test", command=self._on_start_test
        )
        self._test_button.pack(side=tk.LEFT)
        self._test_stop_button = ttk.Button(
            buttons, text="Stop", command=self._guard(self.controller.stop_test)
        )
        self._test_stop_button.pack(side=tk.LEFT, padx=4)
        self._test_text = tk.StringVar()
        ttk.Label(
            tab,
            textvariable=self._test_text,
            justify=tk.LEFT,
            anchor=tk.NW,
            wraplength=320,
        ).pack(fill=tk.X, pady=(6, 0))
        ttk.Label(tab, text="Results").pack(anchor=tk.W, pady=(6, 0))
        results = ttk.Frame(tab)
        results.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(results, orient=tk.HORIZONTAL)
        self._results = tk.Listbox(
            results, height=8, width=44, xscrollcommand=scrollbar.set
        )
        scrollbar.configure(command=self._results.xview)
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self._results.pack(fill=tk.BOTH, expand=True)
        self._refresh_saved_models()

    # ---------------------------------------------------------------- view

    def _rebuild_view(self) -> None:
        """Create a fresh top-down view for the controller's built scenario."""

        import matplotlib

        # TopDownView makes its figure through pyplot.  The non-interactive
        # backend keeps pyplot from opening a window of its own; the figure
        # is then re-homed in a Tk canvas inside this window.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        from trailer_docking.visualization.topdown import TopDownView

        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
        if self._view is not None:
            plt.close(self._view.figure)
        built = self.controller.built
        self._view = TopDownView(built.vehicle, built.dock)
        self._view.set_obstacles(built.obstacles)
        self._view.figure.set_layout_engine("tight")
        self._canvas = FigureCanvasTkAgg(self._view.figure, master=self._view_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas.mpl_connect("button_press_event", self._on_view_click)
        self._shown_state = None
        self._shown_route = ()
        preview = self.controller.preview_state()
        if self.controller.scenario.randomization.mirror_starts:
            from trailer_docking.environments.task import workspace_bounds

            self._view.set_view_bounds(
                *workspace_bounds(
                    preview.tractor_pose, built.dock.geometry, built.task, built.vehicle
                )
            )
        self._view.update(preview, note="start-pose preview")
        self._view.clear_trail()
        self._idle_route = self.controller.preview_route()
        self._canvas.draw_idle()
        self._shown_frame = None
        if self._showing_3d():
            self._preview_3d()

    def _showing_3d(self) -> bool:
        return bool(self._displays.index(self._displays.select()) == 1)

    def _show_frame(self, frame: Any) -> None:
        """Put one rendered RGB image into the 3D tab."""

        from PIL import Image, ImageTk

        self._raw_3d_frame = frame
        image = Image.fromarray(frame)
        width = self._frame_3d.winfo_width() - 16
        height = self._frame_3d.winfo_height() - 16
        if width > 10 and height > 10:
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            self._rendered_3d_size = (width, height)
        self._photo = ImageTk.PhotoImage(image, master=self.root)
        self._image_3d.configure(image=self._photo, text="")

    def _on_3d_panel_resized(self, event: tk.Event) -> None:
        """Debounce panel resizes and refit the latest frame without cropping."""

        if self._raw_3d_frame is None or event.width <= 10 or event.height <= 10:
            return
        requested = (event.width - 16, event.height - 16)
        if requested == self._rendered_3d_size:
            return
        if self._resize_3d_after_id is not None:
            self.root.after_cancel(self._resize_3d_after_id)
        self._resize_3d_after_id = self.root.after(75, self._refit_3d_frame)

    def _refit_3d_frame(self) -> None:
        self._resize_3d_after_id = None
        if not self._closed and self._raw_3d_frame is not None:
            self._show_frame(self._raw_3d_frame)

    def _preview_3d(self) -> None:
        """Render the idle scene; a failure becomes a message, not a dialog."""

        try:
            self._show_frame(
                self.controller.preview_frame(show_route=self._show_route.get())
            )
        except ControllerError as error:
            self._photo = None
            self._raw_3d_frame = None
            self._image_3d.configure(image="", text=str(error))

    def _on_route_toggled(self) -> None:
        """Show or hide the route in both displays, including live 3D runs."""

        from trailer_docking.visualization.render3d import display

        display.show_route = bool(self._show_route.get())
        if self._showing_3d() and not self.controller.busy:
            self._preview_3d()

    def _on_display_changed(self, _event: object = None) -> None:
        if self._showing_3d() and not self.controller.busy:
            self._preview_3d()

    def _refresh_view(self) -> None:
        changed = False
        state = self.controller.spectator_state
        if state is not None and state is not self._shown_state:
            previous = self._shown_state
            if previous is None or state.simulation_time < previous.simulation_time:
                self._view.clear_trail()
            self._view.update(state)
            self._shown_state = state
            changed = True
        route = None
        if self._show_route.get():
            # Before any run, show the route the chosen planner would draw.
            route = self.controller.spectator_route or self._idle_route
        if route is not self._shown_route:
            self._view.set_route(route)
            self._shown_route = route
            changed = True
        if changed:
            self._canvas.draw_idle()
        frame = self.controller.spectator_frame
        if frame is not None and frame is not self._shown_frame:
            self._shown_frame = frame
            if self._showing_3d():
                self._show_frame(frame)
        elif frame is None and self.controller.busy and self._photo is not None:
            # A run without 3D pictures started: a picture left over from an
            # earlier run or preview would look like the current one.
            self._shown_frame = None
            self._photo = None
            self._raw_3d_frame = None
            self._image_3d.configure(
                image="",
                text="This run has no 3D picture.\n"
                "It is on the 2D simulator, or 3D rendering is switched off.\n"
                "See the top-down tab.",
            )

    # ------------------------------------------------------------- actions

    def _guard(self, action: Callable[[], None]) -> Callable[[], None]:
        """Wrap a button command so controller errors become a dialog."""

        def run() -> None:
            try:
                action()
            except (ControllerError, ValueError, OSError) as error:
                messagebox.showerror("Docking control", str(error), parent=self.root)

        return run

    def _sync_parameter_fields(self) -> None:
        for key, value in self._ui_parameter_values().items():
            self._parameter_vars[key].set(value)

    def _on_apply(self) -> None:
        def apply() -> None:
            self.controller.apply_parameters(
                {key: variable.get() for key, variable in self._parameter_vars.items()}
            )
            self._sync_parameter_fields()
            self._rebuild_view()

        self._guard(apply)()

    def _on_save(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save scenario",
            defaultextension=".yaml",
            filetypes=SCENARIO_FILETYPES,
        )
        if path:
            self._guard(lambda: self._saved(self.controller.save_scenario(path)))()

    def _saved(self, path: Path) -> None:
        self._status_bar.set(f"Saved scenario to {path}")

    def _on_load(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root, title="Load scenario", filetypes=SCENARIO_FILETYPES
        )
        if not path:
            return

        def load() -> None:
            self.controller.load_scenario(path)
            self._sync_parameter_fields()
            self._rebuild_view()

        self._guard(load)()

    def _on_browse(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root, title="Choose a saved policy", filetypes=POLICY_FILETYPES
        )
        if path:
            self._policy_file.set(path)
            self._policy_choice.set(POLICY_FILE)
            self._evaluation_target.set("Saved PPO model")
            self._bundle_text.set(f"Selected checkpoint: {path}")

    def _refresh_saved_models(self) -> None:
        """Rebuild the model browser from checkpoints already on disk."""

        selected_path = (
            self._policy_file.get().strip() if hasattr(self, "_policy_file") else ""
        )
        self._model_tree.delete(*self._model_tree.get_children())
        self._saved_policy_paths.clear()
        selected_item = ""
        for index, policy in enumerate(self.controller.saved_policies()):
            item = f"policy-{index}"
            self._saved_policy_paths[item] = policy.path
            self._model_tree.insert(
                "",
                tk.END,
                iid=item,
                values=(
                    policy.experiment_id,
                    policy.name,
                    policy.backend or "?",
                    policy.status or "?",
                ),
            )
            if policy.path.as_posix() == selected_path:
                selected_item = item
        if selected_item:
            self._model_tree.selection_set(selected_item)
            self._model_tree.see(selected_item)
        elif self._saved_policy_paths:
            first = next(iter(self._saved_policy_paths))
            self._model_tree.selection_set(first)
            self._on_model_selected()
        else:
            self._bundle_text.set(
                "No checkpoints found. Complete a training run or open a checkpoint."
            )

    def _on_model_selected(self, _event: object = None) -> None:
        selected = self._model_tree.selection()
        if not selected:
            return
        path = self._saved_policy_paths.get(selected[0])
        if path is None:
            return
        self._policy_file.set(path.as_posix())
        self._policy_choice.set(POLICY_FILE)
        self._evaluation_target.set("Saved PPO model")
        self._bundle_text.set(f"Selected checkpoint: {path}")

    def _on_start_training(self) -> None:
        def start() -> None:
            # Reward fields are part of the scenario contract consumed by the
            # environment. Apply exactly the fields shown on the training tab
            # before creating the immutable training configuration.
            self.controller.apply_parameters(
                {key: self._parameter_vars[key].get() for key in TRAINING_REWARD_KEYS}
            )
            backends = tuple(
                name
                for name, chosen in (
                    ("2d", self._evaluate_2d.get()),
                    ("3d", self._evaluate_3d.get()),
                )
                if chosen
            )
            episodes = _whole_number(
                self._evaluation_episodes.get(), "Test episodes per run", minimum=0
            )
            ppo = self._ppo_vars
            config = self.controller.make_training_config(
                method="ppo",
                backend=self._training_backend.get(),
                runs=_whole_number(self._runs.get(), "Number of runs", minimum=1),
                total_timesteps=_whole_number(
                    self._timesteps.get(), "Timesteps per run", minimum=1
                ),
                seed=_whole_number(self._training_seed.get(), "Seed", minimum=0),
                learning_rate=_number(
                    ppo["learning_rate"].get(), "Learning rate", minimum=0.0
                ),
                n_steps=_whole_number(ppo["n_steps"].get(), "Rollout steps", minimum=1),
                batch_size=_whole_number(
                    ppo["batch_size"].get(), "Batch size", minimum=1
                ),
                n_epochs=_whole_number(
                    ppo["n_epochs"].get(), "Epochs per rollout", minimum=1
                ),
                gamma=_number(ppo["gamma"].get(), "Discount", minimum=0.0),
                gae_lambda=_number(ppo["gae_lambda"].get(), "GAE lambda", minimum=0.0),
                clip_range=_number(
                    ppo["clip_range"].get(), "Policy clip range", minimum=0.0
                ),
                clip_range_vf=_optional_number(
                    ppo["clip_range_vf"].get(),
                    "Value clip range",
                    minimum=0.0,
                ),
                normalize_advantage=self._normalize_advantage.get(),
                entropy_coefficient=_number(
                    ppo["entropy_coefficient"].get(),
                    "Entropy coefficient",
                    minimum=0.0,
                ),
                value_coefficient=_number(
                    ppo["value_coefficient"].get(),
                    "Value-loss coefficient",
                    minimum=0.0,
                ),
                max_grad_norm=_number(
                    ppo["max_grad_norm"].get(),
                    "Maximum gradient norm",
                    minimum=0.0,
                ),
                target_kl=_optional_number(
                    ppo["target_kl"].get(), "Target KL", minimum=0.0
                ),
                device=self._device.get(),
                max_episode_steps=_whole_number(
                    self._max_episode_steps.get(), "Episode step limit", minimum=1
                ),
                checkpoint_interval=_whole_number(
                    self._checkpoint_interval.get(),
                    "Checkpoint interval",
                    minimum=1,
                ),
                purpose=self._purpose.get().strip(),
                evaluation_episodes=episodes if backends else 0,
                evaluation_backends=backends or ("2d",),
            )
            self.controller.start_training(
                config, show_3d=self._render_training_3d.get()
            )

        self._guard(start)()

    def _on_export(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export trained result",
            defaultextension=".dockpolicy.zip",
            filetypes=BUNDLE_FILETYPES,
        )
        if path:
            self._guard(
                lambda: self._status_bar.set(
                    f"Exported result to {self.controller.export_result(path)}"
                )
            )()

    def _on_load_result(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root, title="Load trained result", filetypes=BUNDLE_FILETYPES
        )
        if not path:
            return

        def load() -> None:
            bundle = self.controller.load_result(path)
            self._bundle_text.set(f"Loaded: {bundle.description()}")
            self._policy_choice.set(POLICY_BUNDLE)
            self._evaluation_target.set("Loaded portable model")
            self._sync_parameter_fields()
            self._rebuild_view()
            self._status_bar.set(
                "Loaded result and its scenario"
                + (
                    ""
                    if bundle.fingerprint_matches
                    else " (base specifications changed since training)"
                )
            )

        self._guard(load)()

    def _on_start_test(self) -> None:
        def start() -> None:
            episodes = _whole_number(self._episodes.get(), "Episodes", minimum=1)
            seed = _whole_number(self._test_seed.get(), "Seed", minimum=0)
            target = self._evaluation_target.get()
            choice = {
                "Saved PPO model": POLICY_FILE,
                "Loaded portable model": POLICY_BUNDLE,
                "NMPC": POLICY_NMPC,
                "LQR": POLICY_LQR,
                "No-steering baseline": POLICY_BASELINE,
            }.get(target)
            if choice is None:
                raise ControllerError(f"unknown evaluation controller: {target}")
            policy_path: str | None = None
            if choice == POLICY_BUNDLE and self.controller.loaded_bundle is None:
                raise ControllerError("open a portable model first")
            if choice == POLICY_FILE:
                policy_path = self._policy_file.get().strip()
                if not policy_path:
                    raise ControllerError("choose a saved policy file first")
            self.controller.start_test(
                policy_path=policy_path,
                episodes=episodes,
                seed=seed,
                baseline=choice == POLICY_BASELINE,
                use_bundle=choice == POLICY_BUNDLE,
                method=(
                    "nmpc"
                    if choice == POLICY_NMPC
                    else "lqr"
                    if choice == POLICY_LQR
                    else "ppo"
                ),
                backend=self._test_backend.get(),
                show_3d=self._render_test_3d.get(),
            )

        self._guard(start)()

    # ---------------------------------------------------------------- loop

    def _tick(self) -> None:
        if self._closed:
            return
        try:
            self._refresh()
        finally:
            if not self._closed:
                self._after_id = self.root.after(REFRESH_MILLISECONDS, self._tick)

    def _refresh(self) -> None:
        controller = self.controller
        self._refresh_view()

        training = controller.training_status_lines()
        self._training_text.set("\n".join(training))
        progress = controller.training_progress
        fraction = 0.0
        if progress is not None and progress.total_timesteps > 0:
            fraction = min(progress.timesteps / progress.total_timesteps, 1.0)
        self._progress.configure(value=fraction)
        checkpoint = None if progress is None else progress.checkpoint_path
        checkpoint_key = None if checkpoint is None else str(checkpoint)
        if (
            not controller.busy
            and checkpoint_key != self._last_model_refresh_checkpoint
        ):
            self._last_model_refresh_checkpoint = checkpoint_key
            self._refresh_saved_models()

        testing = controller.test_status_lines()
        # The per-episode lines go in the list below; keep the summary short.
        self._test_text.set(
            "\n".join(line for line in testing if not line.startswith("Episode "))
        )
        results = controller.test_result_lines()
        if results != self._shown_results:
            self._results.delete(0, tk.END)
            for line in results:
                self._results.insert(tk.END, line)
            self._shown_results = results

        training_active = controller.training_active
        test_active = controller.test_status == "running"
        busy = controller.busy
        idle_state = tk.DISABLED if busy else tk.NORMAL
        for action_button in (
            self._apply_button,
            self._load_button,
            self._train_button,
            self._test_button,
            self._load_result_button,
        ):
            action_button.configure(state=idle_state)
        controlled_widgets: tuple[Any, ...] = (
            *self._training_backend_buttons.values(),
            *self._test_backend_buttons.values(),
            *self._policy_buttons.values(),
            *self._training_entries,
            *self._reward_entries,
            *self._test_entries,
            self._policy_file_entry,
            self._browse_button,
            self._refresh_models_button,
            self._target_selector,
            self._training_render_button,
            self._test_render_button,
        )
        for controlled_widget in controlled_widgets:
            controlled_widget.configure(state=idle_state)
        evaluations_enabled = False
        try:
            evaluations_enabled = int(self._evaluation_episodes.get()) > 0
        except ValueError:
            evaluations_enabled = True
        evaluation_state = (
            tk.NORMAL if not busy and evaluations_enabled else tk.DISABLED
        )
        for evaluation_button in self._evaluation_buttons:
            evaluation_button.configure(state=evaluation_state)
        self._export_button.configure(
            state=tk.NORMAL
            if not busy and controller.latest_bundle is not None
            else tk.DISABLED
        )
        self._train_stop_button.configure(
            state=tk.NORMAL if training_active else tk.DISABLED
        )
        self._test_stop_button.configure(
            state=tk.NORMAL if test_active else tk.DISABLED
        )
        if busy:
            self._status_bar.set(f"{training[0]}  |  {testing[0]}")
        elif self._status_bar.get().startswith(("Training:", "Test:")):
            self._status_bar.set(f"Ready  ({training[0]}; {testing[0]})")

    def close(self) -> None:
        """Stop background work and destroy the window."""

        if self._closed:
            return
        self._closed = True
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        if self._resize_3d_after_id is not None:
            self.root.after_cancel(self._resize_3d_after_id)
        self.controller.shutdown()
        if self._view is not None:
            import matplotlib.pyplot as plt

            plt.close(self._view.figure)
        self.root.destroy()


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m trailer_docking.ui.app",
        description="Control window for docking scenarios, training, and tests.",
    )
    parser.add_argument(
        "--scenario", type=Path, default=None, help="scenario YAML to open"
    )
    parser.add_argument(
        "--initial-policy",
        type=Path,
        help="PPO checkpoint for a weights-only warm start",
    )
    parser.add_argument(
        "--timesteps", type=int, default=200_000, help="prefill the training budget"
    )
    parser.add_argument(
        "--learning-rate", type=float, help="prefill the PPO learning rate"
    )
    parser.add_argument(
        "--purpose", default="ppo-ui", help="prefill the experiment name"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("."),
        help="directory under which training writes results and checkpoints",
    )
    parser.add_argument(
        "--smoke-test",
        type=float,
        default=None,
        metavar="SECONDS",
        help="open the window, close it after SECONDS, and exit with status 0",
    )
    arguments = parser.parse_args(argv)
    if arguments.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if arguments.learning_rate is not None and (
        not isfinite(arguments.learning_rate) or arguments.learning_rate <= 0
    ):
        parser.error("--learning-rate must be finite and positive")
    if not arguments.purpose.strip():
        parser.error("--purpose cannot be empty")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    """Run the control window; returns a process exit status."""

    arguments = _arguments(argv)
    try:
        scenario = (
            Scenario.load(arguments.scenario)
            if arguments.scenario is not None
            else Scenario()
        )
        controller = AppController(
            scenario,
            output_root=arguments.output_root,
            initial_policy=arguments.initial_policy,
        )
    except (ConfigurationError, ValueError) as error:
        # Geometry validation errors are ValueErrors but not ConfigurationErrors.
        print(f"error: {error}")
        return 2

    root = tk.Tk()
    app = DockingApp(
        root,
        controller,
        timesteps=arguments.timesteps,
        learning_rate=arguments.learning_rate,
        purpose=arguments.purpose,
    )
    if arguments.smoke_test is not None:
        root.after(int(max(arguments.smoke_test, 0.0) * 1000), app.close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
