import numpy as np
import matplotlib.pyplot as plt


def vykresli_drift(t, r, v=None, a=None, title=None):
    """
    Plot drift trajectory in the plane and time series.
    When v is provided, the time series are shown as a 2x2 grid: x, y, v_x, v_y.
    When a is provided, an extra row a_x, a_y is added (or shown with x, y if v is None).

    Parameters:
        t     : 1D array of time [s]
        r     : array (n, 2) with position [m]
        v     : array (n, 2) with velocity [m/s] (optional)
        a     : array (n, 2) with acceleration [m/s^2] (optional)
        title : title for the trajectory plot (optional)
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(r[:, 0], r[:, 1], label="trajektorie")
    ax.scatter(r[0, 0], r[0, 1], color="green", label="start")
    ax.scatter(r[-1, 0], r[-1, 1], color="red", label="konec")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_aspect("equal", "box")
    if title:
        ax.set_title(title)
    ax.grid(True)
    fig.subplots_adjust(right=0.78)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

    if v is None and a is None:
        fig2, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 5))
        axes = np.atleast_1d(axes)

        axes[0].plot(t, r[:, 0])
        axes[0].set_ylabel(r"$x$ [m]")
        axes[0].grid(True)

        axes[1].plot(t, r[:, 1])
        axes[1].set_ylabel(r"$y$ [m]")
        axes[1].set_xlabel(r"$t$ [s]")
        axes[1].grid(True)
    else:
        rows = 1 + (1 if v is not None else 0) + (1 if a is not None else 0)
        fig2, axes = plt.subplots(nrows=rows, ncols=2, sharex=True, figsize=(8, 2.5 * rows))
        axes = np.asarray(axes)

        row = 0
        axes[row, 0].plot(t, r[:, 0])
        axes[row, 0].set_ylabel(r"$x$ [m]")
        axes[row, 0].grid(True)

        axes[row, 1].plot(t, r[:, 1])
        axes[row, 1].set_ylabel(r"$y$ [m]")
        axes[row, 1].grid(True)
        row += 1

        if v is not None:
            axes[row, 0].plot(t, v[:, 0])
            axes[row, 0].set_ylabel(r"$v_x$ [m/s]")
            axes[row, 0].grid(True)

            axes[row, 1].plot(t, v[:, 1])
            axes[row, 1].set_ylabel(r"$v_y$ [m/s]")
            axes[row, 1].grid(True)
            row += 1

        if a is not None:
            axes[row, 0].plot(t, a[:, 0])
            axes[row, 0].set_ylabel(r"$a_x$ [m/s$^2$]")
            axes[row, 0].grid(True)

            axes[row, 1].plot(t, a[:, 1])
            axes[row, 1].set_ylabel(r"$a_y$ [m/s$^2$]")
            axes[row, 1].grid(True)
            row += 1

        axes[row - 1, 0].set_xlabel(r"$t$ [s]")
        axes[row - 1, 1].set_xlabel(r"$t$ [s]")

    plt.tight_layout()
    plt.show()


def generate_synthetic_w_grid(L=2_000_000.0, nx=21, ny=21, n_days=30, day=24 * 3600.0):
    """
    Generate a synthetic time-varying velocity field on a grid.

    Returns:
        x, y, t_grid, W_grid
    """
    x = np.linspace(-0.5 * L, 0.5 * L, nx)
    y = np.linspace(-0.5 * L, 0.5 * L, ny)
    xx, yy = np.meshgrid(x, y)

    X = xx / (0.5 * L)
    Y = yy / (0.5 * L)

    t_grid = np.arange(0, n_days + 1) * day
    W_grid = np.zeros((t_grid.size, ny, nx, 2), dtype=float)

    for k, tk in enumerate(t_grid):
        theta = 2.0 * np.pi * tk / (5.0 * day)
        base_amp = 2.2 + 0.6 * np.sin(2.0 * np.pi * tk / (3.0 * day))
        base = base_amp * np.array([np.cos(theta), np.sin(theta)])

        radial_amp = 1.2 * np.sin(2.0 * np.pi * tk / (2.2 * day))
        radial = radial_amp * np.stack([X, Y], axis=-1) * np.exp(-0.8 * (X**2 + Y**2))[..., None]

        cx = 0.45 * np.cos(2.0 * np.pi * tk / (6.0 * day))
        cy = 0.45 * np.sin(2.0 * np.pi * tk / (6.0 * day))
        Xc = X - cx
        Yc = Y - cy
        vortex1 = 5.0 * np.stack([-Yc, Xc], axis=-1) * np.exp(-4.0 * (Xc**2 + Yc**2))[..., None]

        Xc2 = X + cx
        Yc2 = Y + cy
        vortex2 = -5.0 * np.stack([-Yc2, Xc2], axis=-1) * np.exp(-4.0 * (Xc2**2 + Yc2**2))[..., None]

        phase1 = 2.0 * np.pi * (0.7 * X + 0.4 * Y + tk / (4.5 * day))
        phase2 = 2.0 * np.pi * (0.3 * X - 0.8 * Y - tk / (3.2 * day))
        wave1 = 1.8 * np.stack([np.cos(phase1), np.sin(phase1)], axis=-1)
        wave2 = 1.2 * np.stack([-np.sin(phase2), np.cos(phase2)], axis=-1)

        W_grid[k] = base + radial + vortex1 + vortex2 + wave1 + 0.7 * wave2

    return x, y, t_grid, W_grid


def w_synthetic(x_pos, y_pos, t, L=2_000_000.0, day=24 * 3600.0):
    """
    Evaluate the synthetic velocity field at a given position and time.

    Parameters:
        x_pos, y_pos : position in meters (scalars or arrays)
        t            : time in seconds
        L            : domain size used for scaling (meters)
        day          : seconds per day

    Returns:
        Velocity vector(s) with shape (..., 2) in m/s.
    """
    x_pos = np.asarray(x_pos, dtype=float)
    y_pos = np.asarray(y_pos, dtype=float)
    t = float(t)

    if not np.isfinite(t) or not np.all(np.isfinite(x_pos)) or not np.all(np.isfinite(y_pos)):
        shape = np.broadcast(x_pos, y_pos).shape
        return np.zeros(shape + (2,), dtype=float)

    scale = 0.5 * L
    X = x_pos / scale
    Y = y_pos / scale

    theta = 2.0 * np.pi * t / (5.0 * day)
    base_amp = 2.2 + 0.6 * np.sin(2.0 * np.pi * t / (3.0 * day))
    base = np.empty(X.shape + (2,), dtype=float)
    base[..., 0] = base_amp * np.cos(theta)
    base[..., 1] = base_amp * np.sin(theta)

    radial_amp = 1.2 * np.sin(2.0 * np.pi * t / (2.2 * day))
    radial = radial_amp * np.stack([X, Y], axis=-1) * np.exp(-0.8 * (X**2 + Y**2))[..., None]

    cx = 0.45 * np.cos(2.0 * np.pi * t / (6.0 * day))
    cy = 0.45 * np.sin(2.0 * np.pi * t / (6.0 * day))
    Xc = X - cx
    Yc = Y - cy
    vortex1 = 5.0 * np.stack([-Yc, Xc], axis=-1) * np.exp(-4.0 * (Xc**2 + Yc**2))[..., None]

    Xc2 = X + cx
    Yc2 = Y + cy
    vortex2 = -5.0 * np.stack([-Yc2, Xc2], axis=-1) * np.exp(-4.0 * (Xc2**2 + Yc2**2))[..., None]

    phase1 = 2.0 * np.pi * (0.7 * X + 0.4 * Y + t / (4.5 * day))
    phase2 = 2.0 * np.pi * (0.3 * X - 0.8 * Y - t / (3.2 * day))
    wave1 = 1.8 * np.stack([np.cos(phase1), np.sin(phase1)], axis=-1)
    wave2 = 1.2 * np.stack([-np.sin(phase2), np.cos(phase2)], axis=-1)

    return base + radial + vortex1 + vortex2 + wave1 + 0.7 * wave2


def _sample_time_slice(W_grid, t_grid, t, interpolate_time=True):
    t_grid = np.asarray(t_grid, dtype=float)
    t = float(t)
    nt = t_grid.size
    if nt == 0:
        raise ValueError("t_grid is empty")
    if nt == 1 or not interpolate_time:
        it = int(np.clip(np.searchsorted(t_grid, t), 0, nt - 1))
        return W_grid[it]

    k0 = int(np.clip(np.searchsorted(t_grid, t) - 1, 0, nt - 2))
    alpha = (t - t_grid[k0]) / (t_grid[k0 + 1] - t_grid[k0])
    return (1.0 - alpha) * W_grid[k0] + alpha * W_grid[k0 + 1]


def w_from_grid(x_pos, y_pos, t, W_grid, x_grid, y_grid, t_grid):
    """
    Sample the velocity field at a single point (piecewise constant in space/time).
    """
    if not np.isfinite(x_pos) or not np.isfinite(y_pos) or not np.isfinite(t):
        return np.zeros(2, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    nx = x_grid.size
    ny = y_grid.size
    nt = t_grid.size

    x0 = x_grid[0]
    y0 = y_grid[0]
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    t0 = t_grid[0]
    dt = t_grid[1] - t_grid[0]

    ix = int(np.clip(np.floor((x_pos - x0) / dx), 0, nx - 1))
    iy = int(np.clip(np.floor((y_pos - y0) / dy), 0, ny - 1))
    it = int(np.clip(np.floor((t - t0) / dt), 0, nt - 1))
    return W_grid[it, iy, ix]


def w_from_grid_mesh(X, Y, t, W_grid, x_grid, y_grid, t_grid, interpolate_time=True):
    """
    Vectorized sampling of the grid on a mesh of positions.
    """
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)) or not np.isfinite(t):
        return np.zeros((*np.asarray(X).shape, 2), dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)

    nx = x_grid.size
    ny = y_grid.size

    x0 = x_grid[0]
    y0 = y_grid[0]
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    ix = np.clip(np.floor((X - x0) / dx), 0, nx - 1).astype(int)
    iy = np.clip(np.floor((Y - y0) / dy), 0, ny - 1).astype(int)

    W_t = _sample_time_slice(W_grid, t_grid, t, interpolate_time=interpolate_time)
    return W_t[iy, ix]


def save_vector_field_animation(
    W_grid,
    x,
    y,
    t_grid,
    out_path="vector_field_hourly.mp4",
    hour=3600.0,
    day=24 * 3600.0,
    stride=2,
    fps=8,
    dpi=90,
    scale=40.0,
    width=0.0025,
    interval=120,
    bitrate=1800,
):
    """
    Save a quiver animation of the time-varying field and return the output path.
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    ny = y.size
    nx = x.size

    XX, YY = np.meshgrid(x / 1000.0, y / 1000.0)
    t_anim = np.arange(t_grid[0], t_grid[-1] + hour, hour)

    W_anim = np.empty((t_anim.size, ny, nx, 2), dtype=float)
    for i, t in enumerate(t_anim):
        W_anim[i] = _sample_time_slice(W_grid, t_grid, t, interpolate_time=True)

    XXs = XX[::stride, ::stride]
    YYs = YY[::stride, ::stride]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    U0 = W_anim[0, ::stride, ::stride, 0]
    V0 = W_anim[0, ::stride, ::stride, 1]
    quiv = ax.quiver(XXs, YYs, U0, V0, scale=scale, width=width, pivot="mid")
    title = ax.set_title(f"t = {t_anim[0] / hour:.0f} h")
    ax.set_xlabel(r"$x$ [km]")
    ax.set_ylabel(r"$y$ [km]")
    ax.set_aspect("equal")
    ax.set_xlim(x.min() / 1000.0, x.max() / 1000.0)
    ax.set_ylim(y.min() / 1000.0, y.max() / 1000.0)
    ax.grid(True, alpha=0.3)

    def update(frame):
        U = W_anim[frame, ::stride, ::stride, 0]
        V = W_anim[frame, ::stride, ::stride, 1]
        quiv.set_UVC(U, V)
        elapsed = t_anim[frame]
        days = int(elapsed // day)
        hours = int((elapsed % day) // hour)
        title.set_text(f"t = {days} d {hours:02d} h")
        return quiv, title

    anim = FuncAnimation(fig, update, frames=t_anim.size, interval=interval, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)

    return out_path
