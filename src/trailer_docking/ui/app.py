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
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from trailer_docking.core.config import ConfigurationError
from trailer_docking.core.state import VehicleState
from trailer_docking.evaluation.scenarios import Scenario
from trailer_docking.ui.controller import (
    BACKENDS,
    GROUPS,
    OBSTACLE_NOTE,
    PARAMETERS,
    PHYSICS_GROUP,
    PHYSICS_GROUP_NOTE,
    AppController,
    ControllerError,
    ParameterSpec,
)

REFRESH_MILLISECONDS = 66  # about 15 Hz
POLICY_LATEST, POLICY_BASELINE, POLICY_FILE = "latest", "baseline", "file"
POLICY_BUNDLE, POLICY_NMPC = "bundle", "nmpc"
METHODS = ("ppo", "nmpc")
BUNDLE_FILETYPES = (
    ("Policy bundle", "*.dockpolicy.zip"),
    ("All files", "*.*"),
)
SCENARIO_FILETYPES = (("Scenario YAML", "*.yaml *.yml"), ("All files", "*.*"))
POLICY_FILETYPES = (("Saved policy", "*.zip"), ("All files", "*.*"))


def _whole_number(text: str, label: str, *, minimum: int) -> int:
    try:
        value = int(text.strip().replace("_", ""))
    except ValueError:
        raise ControllerError(f"{label}: '{text}' is not a whole number") from None
    if value < minimum:
        raise ControllerError(f"{label} must be at least {minimum}")
    return value


class DockingApp:
    """The main window; a thin view over an :class:`AppController`."""

    def __init__(self, root: tk.Tk, controller: AppController) -> None:
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
        self._shown_results: tuple[str, ...] = ()
        self._parameter_vars: dict[str, tk.StringVar] = {}

        root.title("Tractor-trailer docking \N{EM DASH} training control")
        root.minsize(1100, 620)
        root.protocol("WM_DELETE_WINDOW", self.close)

        panes = ttk.Frame(root, padding=6)
        panes.pack(fill=tk.BOTH, expand=True)
        notebook = ttk.Notebook(panes)
        notebook.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(panes)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self._build_scenario_tab(notebook)
        self._build_training_tab(notebook)
        self._build_test_tab(notebook)

        toolbar = ttk.Frame(right)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self._show_route = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toolbar,
            text="Show planned route",
            variable=self._show_route,
            command=self._on_route_toggled,
        ).pack(side=tk.LEFT)
        self._render_3d = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toolbar,
            text="Render 3D runs (slower training)",
            variable=self._render_3d,
        ).pack(side=tk.LEFT, padx=(12, 0))
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
        self._displays.bind("<<NotebookTabChanged>>", self._on_display_changed)

        self._rebuild_view()
        self._tick()

    # ---------------------------------------------------------------- tabs

    def _build_scenario_tab(self, notebook: ttk.Notebook) -> None:
        outer = ttk.Frame(notebook)
        notebook.add(outer, text="Scenario")
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
        values = self.controller.parameter_values()
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
            if group == "Obstacles":
                ttk.Label(frame, text=OBSTACLE_NOTE, foreground="#555555").grid(
                    row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 4)
                )
                row += 1
            for spec in PARAMETERS:
                if spec.group != group:
                    continue
                variable = tk.StringVar(value=values[spec.key])
                self._parameter_vars[spec.key] = variable
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

    def _build_training_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text="Training")
        form = ttk.Frame(tab)
        form.pack(fill=tk.X)
        form.columnconfigure(1, weight=1)
        self._method = tk.StringVar(value=METHODS[0])
        self._training_backend = tk.StringVar(value=BACKENDS[0])
        self._runs = tk.StringVar(value="1")
        self._timesteps = tk.StringVar(value="200000")
        self._training_seed = tk.StringVar(value="0")
        self._evaluation_episodes = tk.StringVar(value="20")
        self._evaluate_2d = tk.BooleanVar(value=True)
        self._evaluate_3d = tk.BooleanVar(value=False)
        choices: tuple[tuple[str, tk.StringVar, tuple[str, ...]], ...] = (
            ("Method", self._method, METHODS),
            ("Simulator", self._training_backend, BACKENDS),
        )
        for row, (label, variable, options) in enumerate(choices):
            ttk.Label(form, text=label).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 6)
            )
            ttk.Combobox(
                form, textvariable=variable, values=options, state="readonly", width=12
            ).grid(row=row, column=1, sticky=tk.EW, pady=1)
        entries = (
            ("Number of runs", self._runs),
            ("Timesteps per run", self._timesteps),
            ("Seed of first run", self._training_seed),
            ("Test episodes per run", self._evaluation_episodes),
        )
        for row, (label, variable) in enumerate(entries, start=len(choices)):
            ttk.Label(form, text=label).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 6)
            )
            ttk.Entry(form, textvariable=variable, width=14).grid(
                row=row, column=1, sticky=tk.EW, pady=1
            )
        tests = ttk.Frame(tab)
        tests.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(tests, text="Test each run in:").pack(side=tk.LEFT)
        ttk.Checkbutton(tests, text="2D", variable=self._evaluate_2d).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Checkbutton(tests, text="3D", variable=self._evaluate_3d).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Label(
            tab,
            text="NMPC is not trained: it is only tested.",
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(4, 0))
        buttons = ttk.Frame(tab)
        buttons.pack(fill=tk.X, pady=6)
        self._train_button = ttk.Button(
            buttons, text="Start", command=self._on_start_training
        )
        self._train_button.pack(side=tk.LEFT)
        self._train_stop_button = ttk.Button(
            buttons, text="Stop", command=self._guard(self.controller.stop_training)
        )
        self._train_stop_button.pack(side=tk.LEFT, padx=4)
        self._export_button = ttk.Button(
            buttons,
            text="Export result\N{HORIZONTAL ELLIPSIS}",
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
        notebook.add(tab, text="Docking test")
        form = ttk.Frame(tab)
        form.pack(fill=tk.X)
        form.columnconfigure(1, weight=1)
        self._episodes = tk.StringVar(value="1")
        self._test_seed = tk.StringVar(value="0")
        self._test_backend = tk.StringVar(value=BACKENDS[0])
        ttk.Label(form, text="Simulator").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 6)
        )
        ttk.Combobox(
            form,
            textvariable=self._test_backend,
            values=BACKENDS,
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky=tk.EW, pady=1)
        for row, (label, variable) in enumerate(
            (("Episodes", self._episodes), ("Seed", self._test_seed)), start=1
        ):
            ttk.Label(form, text=label).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 6)
            )
            ttk.Entry(form, textvariable=variable, width=14).grid(
                row=row, column=1, sticky=tk.EW, pady=1
            )

        policy = ttk.LabelFrame(tab, text="Policy", padding=6)
        policy.pack(fill=tk.X, pady=6)
        self._policy_choice = tk.StringVar(value=POLICY_LATEST)
        self._policy_file = tk.StringVar()
        for value, text in (
            (POLICY_LATEST, "Latest training checkpoint"),
            (POLICY_BUNDLE, "Loaded result bundle"),
            (POLICY_BASELINE, "Untrained straight-reverse baseline"),
            (POLICY_NMPC, "NMPC controller"),
            (POLICY_FILE, "Saved policy file (.zip):"),
        ):
            ttk.Radiobutton(
                policy, text=text, value=value, variable=self._policy_choice
            ).pack(anchor=tk.W)
        self._bundle_text = tk.StringVar(value="No result bundle loaded")
        ttk.Label(
            policy, textvariable=self._bundle_text, foreground="#555555", wraplength=300
        ).pack(anchor=tk.W)
        chooser = ttk.Frame(policy)
        chooser.pack(fill=tk.X)
        ttk.Entry(chooser, textvariable=self._policy_file).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(
            chooser, text="Browse\N{HORIZONTAL ELLIPSIS}", command=self._on_browse
        ).pack(side=tk.LEFT, padx=(4, 0))

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
        self._load_result_button = ttk.Button(
            buttons,
            text="Load result\N{HORIZONTAL ELLIPSIS}",
            command=self._on_load_result,
        )
        self._load_result_button.pack(side=tk.LEFT)
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
        self._shown_state = None
        self._shown_route = ()
        self._view.update(self.controller.preview_state(), note="start-pose preview")
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

        self._photo = ImageTk.PhotoImage(Image.fromarray(frame), master=self.root)
        self._image_3d.configure(image=self._photo, text="")

    def _preview_3d(self) -> None:
        """Render the idle scene; a failure becomes a message, not a dialog."""

        try:
            self._show_frame(
                self.controller.preview_frame(show_route=self._show_route.get())
            )
        except ControllerError as error:
            self._photo = None
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
        for key, value in self.controller.parameter_values().items():
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

    def _on_start_training(self) -> None:
        def start() -> None:
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
            config = self.controller.make_training_config(
                method=self._method.get(),
                backend=self._training_backend.get(),
                runs=_whole_number(self._runs.get(), "Number of runs", minimum=1),
                total_timesteps=_whole_number(
                    self._timesteps.get(), "Timesteps per run", minimum=1
                ),
                seed=_whole_number(self._training_seed.get(), "Seed", minimum=0),
                evaluation_episodes=episodes if backends else 0,
                evaluation_backends=backends or ("2d",),
            )
            self.controller.start_training(config, show_3d=self._render_3d.get())

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
            choice = self._policy_choice.get()
            policy_path: str | None = None
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
                method="nmpc" if choice == POLICY_NMPC else "ppo",
                backend=self._test_backend.get(),
                show_3d=self._render_3d.get(),
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
        for button in (
            self._apply_button,
            self._load_button,
            self._train_button,
            self._test_button,
            self._load_result_button,
        ):
            button.configure(state=idle_state)
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the control window; returns a process exit status."""

    arguments = _arguments(argv)
    try:
        scenario = (
            Scenario.load(arguments.scenario)
            if arguments.scenario is not None
            else Scenario()
        )
        controller = AppController(scenario, output_root=arguments.output_root)
    except (ConfigurationError, ValueError) as error:
        # Geometry validation errors are ValueErrors but not ConfigurationErrors.
        print(f"error: {error}")
        return 2

    root = tk.Tk()
    app = DockingApp(root, controller)
    if arguments.smoke_test is not None:
        root.after(int(max(arguments.smoke_test, 0.0) * 1000), app.close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
