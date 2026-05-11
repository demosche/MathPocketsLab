import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from enum import Enum
from numpy.typing import NDArray


class BoundaryType(Enum):
    FIXED = 1
    FREE = 2


class Wave1D:
    def __init__(self, Nx: int, L: float, c: float, boundary: BoundaryType):
        self.Nx = Nx
        self.L = L
        self.c = c
        self.boundary = boundary

        self.x = np.linspace(0, L, Nx)

        dx = L / Nx
        dt = 0.4 * dx / c
        self.r = (c * dt / dx) ** 2


        self.u = np.zeros(Nx)
        self.u_prev = np.zeros(Nx)

        self.u[int(Nx * 0.2)] = 1.0
        self.u_prev[:] = self.u

    def apply_boundary(self, u: NDArray[np.float64]) -> None:
        if self.boundary == BoundaryType.FIXED:
            u[0] = 0
            u[-1] = 0
        else:
            u[0] = u[1]
            u[-1] = u[-2]

    def step(self) -> None:
        u_next = np.zeros_like(self.u)

        for i in range(1, self.Nx - 1):
            u_next[i] = (
                2 * self.u[i]
                - self.u_prev[i]
                + self.r * (self.u[i + 1] - 2 * self.u[i] + self.u[i - 1])
            )

        self.apply_boundary(u_next)

        self.u_prev = self.u.copy()
        self.u = u_next


def simulate():
    Nx = 400
    L = 1.0
    c = 1.0
    Nt = 800

    fixed = Wave1D(Nx, L, c, BoundaryType.FIXED)
    free = Wave1D(Nx, L, c, BoundaryType.FREE)

    fig, ax = plt.subplots(3, 1, figsize=(8, 10))

    line_f, = ax[0].plot(fixed.x, fixed.u)
    ax[0].set_title("Fixed boundary")

    line_fr, = ax[1].plot(free.x, free.u)
    ax[1].set_title("Free boundary")

    line_o1, = ax[2].plot(fixed.x, fixed.u, label="Fixed")
    line_o2, = ax[2].plot(free.x, free.u, label="Free")
    ax[2].legend()
    ax[2].set_title("Overlay")

    for a in ax:
        a.set_ylim(-1.5, 1.5)

    def update(_):
        fixed.step()
        free.step()

        line_f.set_ydata(fixed.u)
        line_fr.set_ydata(free.u)

        line_o1.set_ydata(fixed.u)
        line_o2.set_ydata(free.u)

        return line_f, line_fr, line_o1, line_o2

    animation_object= animation.FuncAnimation(
        fig,
        update,
        frames=Nt,
        interval=20,
        blit=False
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate()