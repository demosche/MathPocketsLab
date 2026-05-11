import numpy as np
from dataclasses import dataclass


import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


@dataclass
class HeatEquationConfig:
    length: float = 1.0
    nx: int = 200
    alpha: float = 0.01
    dt: float = 0.001
    total_time: float = 10.0


class HeatSolver1D:
    def __init__(self, config: HeatEquationConfig) -> None:
        self.config = config

        self.x = np.linspace(0, config.length, config.nx)
        self.dx = self.x[1] - self.x[0]

        self.u = self._initial_condition()
        self.u_new = np.copy(self.u)

        self.r = config.alpha * config.dt / (self.dx ** 2)

    def _initial_condition(self) -> np.ndarray:
        return np.exp(-200 * (self.x - self.config.length / 2) ** 2)

    def step(self) -> None:
        u = self.u
        un = self.u_new


        un[0] = 0.0
        un[-1] = 0.0

        # update interior points
        for i in range(1, len(u) - 1):
            un[i] = u[i] + self.r * (u[i + 1] - 2 * u[i] + u[i - 1])

        self.u, self.u_new = self.u_new, self.u


class HeatVisualizer(QtWidgets.QMainWindow):
    def __init__(self, solver: HeatSolver1D) -> None:
        super().__init__()

        self.solver = solver

        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        self.plot_widget.setTitle("1D Heat Diffusion")
        self.plot_widget.setLabel("left", "Temperature")
        self.plot_widget.setLabel("bottom", "Position x")

        self.curve = self.plot_widget.plot(pen="b")

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start(10)

    def refresh_plot(self) -> None:
        self.solver.step()
        self.curve.setData(self.solver.x, self.solver.u)


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