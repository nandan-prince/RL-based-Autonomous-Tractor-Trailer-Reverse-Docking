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
    GROUPS,
    PARAMETERS,
    PHYSICS_GROUP,
    PHYSICS_GROUP_NOTE,
    AppController,
    ControllerError,
    ParameterSpec,
)

REFRESH_MILLISECONDS = 66  # about 15 Hz
POLICY_LATEST, POLICY_BASELINE, POLICY_FILE = "latest", "baseline", "file"
SCENARIO_FILETYPES = (("Scenario YAML", "*.yaml *.yml"), ("All files", "*.*"))
POLICY_FILETYPES = (("Saved policy", "*.zip"), ("All files", "*.*"))


def _whole_number(text: str, label: str, *, minimum: int) -> int:
    try:
        value = int(text.strip().replace("_", "").replace(",", ""))
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
            toolbar, text="Show planned route", variable=self._show_route
        ).pack(side=tk.LEFT)
        self._status_bar = tk.StringVar(value="Ready")
        ttk.Label(
            right, textvariable=self._status_bar, relief=tk.SUNKEN, anchor=tk.W
        ).pack(side=tk.BOTTOM, fill=tk.X)
        self._view_frame = ttk.Frame(right)
        self._view_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._rebuild_view()
        self._tick()

    # ---------------------------------------------------------------- tabs

    def _build_scenario_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text="Scenario")
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
            text="Blank = value from the base specification.\n"
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
        self._timesteps = tk.StringVar(value="200000")
        self._training_seed = tk.StringVar(value="0")
        for row, (label, variable) in enumerate(
            (("Total timesteps", self._timesteps), ("Seed", self._training_seed))
        ):
            ttk.Label(form, text=label).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 6)
            )
            ttk.Entry(form, textvariable=variable, width=14).grid(
                row=row, column=1, sticky=tk.EW, pady=1
            )
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
        for row, (label, variable) in enumerate(
            (("Episodes", self._episodes), ("Seed", self._test_seed))
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
            (POLICY_BASELINE, "Untrained straight-reverse baseline"),
            (POLICY_FILE, "Saved policy file (.zip):"),
        ):
            ttk.Radiobutton(
                policy, text=text, value=value, variable=self._policy_choice
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
        self._view.figure.set_layout_engine("tight")
        self._canvas = FigureCanvasTkAgg(self._view.figure, master=self._view_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._shown_state = None
        self._shown_route = ()
        self._view.update(self.controller.preview_state(), note="start-pose preview")
        self._view.clear_trail()
        self._canvas.draw_idle()

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
        route = self.controller.spectator_route if self._show_route.get() else None
        if route is not self._shown_route:
            self._view.set_route(route)
            self._shown_route = route
            changed = True
        if changed:
            self._canvas.draw_idle()

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
            config = self.controller.make_training_config(
                total_timesteps=_whole_number(
                    self._timesteps.get(), "Total timesteps", minimum=1
                ),
                seed=_whole_number(self._training_seed.get(), "Seed", minimum=0),
            )
            self.controller.start_training(config)

        self._guard(start)()

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
        ):
            button.configure(state=idle_state)
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
