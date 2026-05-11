import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_grid(L, Nx, c):
    dx = L / Nx
    dt = 0.4 * dx / c
    x = np.linspace(0, L, Nx)
    r = (c * dt / dx) ** 2
    return x, r


def init_state(Nx):
    u = np.zeros(Nx)
    u[int(Nx * 0.2)] = 1.0
    return u, u.copy()


def step(u, u_prev, r, Nx):
    u_next = np.zeros_like(u)

    for i in range(1, Nx - 1):
        u_next[i] = (2 * u[i] - u_prev[i] +
                     r * (u[i + 1] - 2 * u[i] + u[i - 1]))

    return u_next


def apply_boundary(u, kind):
    if kind == "fixed":
        u[0] = 0
        u[-1] = 0
    elif kind == "free":
        u[0] = u[1]
        u[-1] = u[-2]


def simulate():
    L = 1.0
    Nx = 400
    c = 1.0
    Nt = 800

    x, r = create_grid(L, Nx, c)

    u_f, u_f_prev = init_state(Nx)
    u_fr, u_fr_prev = init_state(Nx)

    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    line_f, = axs[0].plot(x, u_f, color="blue")
    axs[0].set_title("Fixed boundary")

    line_fr, = axs[1].plot(x, u_fr, color="green")
    axs[1].set_title("Free boundary")

    line_overlay_f, = axs[2].plot(x, u_f, label="Fixed", color="blue")
    line_overlay_fr, = axs[2].plot(x, u_fr, label="Free", color="green")
    axs[2].set_title("Overlay comparison")
    axs[2].legend()

    for ax in axs:
        ax.set_ylim(-1.5, 1.5)

    def update(frame):
        nonlocal u_f, u_f_prev, u_fr, u_fr_prev

        # FIXED
        u_f_next = step(u_f, u_f_prev, r, Nx)
        apply_boundary(u_f_next, "fixed")
        u_f_prev = u_f.copy()
        u_f = u_f_next

        # FREE
        u_fr_next = step(u_fr, u_fr_prev, r, Nx)
        apply_boundary(u_fr_next, "free")
        u_fr_prev = u_fr.copy()
        u_fr = u_fr_next

        # update plots
        line_f.set_ydata(u_f)
        line_fr.set_ydata(u_fr)

        line_overlay_f.set_ydata(u_f)
        line_overlay_fr.set_ydata(u_fr)

        return line_f, line_fr, line_overlay_f, line_overlay_fr

    animation_loop = animation.FuncAnimation(
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