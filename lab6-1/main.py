import numpy as np
from dataclasses import dataclass

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


@dataclass
class HeatEquationConfig:
    length: float = 1.0      # Physical length of the rod [metres]
    nx: int = 200            # Number of spatial grid points
    alpha: float = 0.01      # Thermal diffusivity [m²/s]
    dt: float = 0.001        # Time step size [seconds]
    total_time: float = 10.0 # Total simulation duration [seconds]


class HeatSolver1D:
    def __init__(self, config: HeatEquationConfig) -> None:
        self.config = config
        self.elapsed_time = 0.0

        self.x = np.linspace(0, config.length, config.nx)
        self.dx = self.x[1] - self.x[0]


        self.cfl_number = config.alpha * config.dt / (self.dx ** 2)
        if self.cfl_number > 0.5:
            raise ValueError(
                f"CFL condition violated: r={self.cfl_number:.4f} > 0.5. "
                "Reduce dt or increase nx to restore stability."
            )

        self.u = self._initial_condition()
        self.u_new = np.copy(self.u)

    def _initial_condition(self) -> np.ndarray:
        centre = self.config.length / 2
        peak_temperature_celsius = 100.0  # °C
        return peak_temperature_celsius * np.exp(-200 * (self.x - centre) ** 2)

    def step(self) -> None:
        u = self.u

        self.u_new[1:-1] = u[1:-1] + self.cfl_number * (u[2:] - 2 * u[1:-1] + u[:-2])
        self.u_new[0] = self.u_new[-1] = 0.0  # Dirichlet BCs: fixed zero temperature at walls
        self.u, self.u_new = self.u_new, self.u  
        self.elapsed_time += self.config.dt


class HeatVisualizer(QtWidgets.QMainWindow):
    def __init__(self, solver: HeatSolver1D) -> None:
        super().__init__()
        self.solver = solver

        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        self._configure_plot()
        self._start_animation()

    def _configure_plot(self) -> None:
        self.plot_widget.setLabel("left", "Temperature (normalised)")
        self.plot_widget.setLabel("bottom", "Position (m)")
        self.plot_widget.setLabel("left", "Temperature (°C)")
        self.plot_widget.setYRange(0.0, 100.0, padding=0.05)  
        self.plot_widget.showGrid(x=False, y=True, alpha=0.3)  
        self.curve = self.plot_widget.plot(pen="b")

        self.time_label = pg.TextItem(anchor=(1, 0), color="k")
        self.plot_widget.addItem(self.time_label)
        self.time_label.setPos(self.solver.config.length, 1.0)

        self._update_title()

    def _update_title(self) -> None:
        t = self.solver.elapsed_time
        total = self.solver.config.total_time
        self.plot_widget.setTitle(f"1D Heat Diffusion — t = {t:.3f}s / {total:.1f}s")

    def _start_animation(self) -> None:
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start(10)  # Fire every 10 ms — targets ~100 FPS

    def refresh_plot(self) -> None:
        self.solver.step()
        self.curve.setData(self.solver.x, self.solver.u)
        self._update_title()


def main() -> None:
    config = HeatEquationConfig()
    solver = HeatSolver1D(config)

    app = QtWidgets.QApplication([])
    pg.setConfigOptions(
        useOpenGL=True,
        antialias=False,
        background="w",
        foreground="k",
    )
    window = HeatVisualizer(solver)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()