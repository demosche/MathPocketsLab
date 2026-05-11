import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from enum import Enum


class BoundaryType(Enum):
    FIXED = 1
    FREE = 2


class Wave1D:
    def __init__(self, Nx: int, L: float, c: float, boundary: BoundaryType) -> None:
        self.Nx = Nx
        self.boundary = boundary
        self.x: np.ndarray = np.linspace(0, L, Nx)

        dx: float = L / Nx
        dt: float = 0.4 * dx / c
        self.r: float = (c * dt / dx) ** 2

        self.u:      np.ndarray = np.zeros(Nx)
        self.u_prev: np.ndarray = np.zeros(Nx)
        self.u_next: np.ndarray = np.zeros(Nx)

        self.u[int(Nx * 0.2)] = 1.0
        self.u_prev[:] = self.u

    def step(self, n: int = 1) -> None:
        r, u, u_prev, u_next = self.r, self.u, self.u_prev, self.u_next
        fixed: bool = self.boundary == BoundaryType.FIXED

        for _ in range(n):
            u_next[1:-1] = (
                2 * u[1:-1]
                - u_prev[1:-1]
                + r * (u[2:] - 2 * u[1:-1] + u[:-2])
            )
            if fixed:
                u_next[0] = 0.0
                u_next[-1] = 0.0
            else:
                u_next[0]  = u_next[1]
                u_next[-1] = u_next[-2]

            u_prev, u = u, u_next
            u_next = u_prev

        self.u, self.u_prev, self.u_next = u, u_prev, u_next


def run() -> None:
    pg.setConfigOptions(
        useOpenGL=True,
        antialias=False,
        background="w",
        foreground="k",
    )

    app = pg.mkQApp()
    win = pg.GraphicsLayoutWidget(show=True)
    win.setWindowTitle("1D Wave Equation")

    Nx: int = 400
    fixed = Wave1D(Nx, 1.0, 1.0, BoundaryType.FIXED)
    free  = Wave1D(Nx, 1.0, 1.0, BoundaryType.FREE)

    p1 = win.addPlot(title="Fixed boundary")
    win.nextRow()
    p2 = win.addPlot(title="Free boundary")

    for p in (p1, p2):
        p.setYRange(-1.5, 1.5)
        p.showGrid(x=True, y=True, alpha=0.3)
        p.setDownsampling(auto=True, mode="peak")
        p.setClipToView(True)

    c1 = p1.plot(fixed.x, fixed.u, pen=pg.mkPen("b", width=2))
    c2 = p2.plot(free.x,  free.u,  pen=pg.mkPen("g", width=2))

    STEPS_PER_FRAME: int = 4

    def update() -> None:
        fixed.step(STEPS_PER_FRAME)
        free.step(STEPS_PER_FRAME)
        c1.setData(y=fixed.u)
        c2.setData(y=free.u)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0)

    pg.exec()


if __name__ == "__main__":
    run()