"""Utilities used by cv2_reseni.ipynb for drift simulations."""

from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt


# Turbulence mode constants for w_synthetic (kept global for Numba performance).
_K = np.array(
    [
        [2.2, -5.7],
        [4.9, 3.1],
        [-6.4, 2.4],
        [7.8, -2.6],
        [3.0, 7.2],
        [-8.5, -3.4],
        [9.7, 1.1],
        [-1.6, 10.3],
        [5.6, -9.4],
        [-10.8, 4.7],
    ],
    dtype=np.float64,
)

_A = np.array([0.16, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04], dtype=np.float64)
_P = np.array([3.7, 4.9, 6.1, 5.3, 7.4, 8.2, 3.3, 9.1, 6.7, 4.1], dtype=np.float64)
_Q = np.array([5.9, 7.1, 4.3, 8.4, 3.8, 6.6, 9.7, 5.2, 7.7, 3.5], dtype=np.float64)


def vykresli_drift(t, r, v=None, a=None, title=None):
    """Plot drift trajectory and time series.

    Parameters
    ----------
    t : ndarray
        1D array of time in seconds.
    r : ndarray, shape (n, 2)
        Positions in meters.
    v : ndarray, shape (n, 2), optional
        Velocities in m/s.
    a : ndarray, shape (n, 2), optional
        Accelerations in m/s^2.
    title : str, optional
        Title for the trajectory plot.
    """
    # Trajectory plot in the x-y plane.
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

    # Time-series grid: position always, optional rows for velocity and acceleration.
    if v is None and a is None:
        fig_ts, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 5))
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
        fig_ts, axes = plt.subplots(nrows=rows, ncols=2, sharex=True, figsize=(8, 2.5 * rows))
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

        # Label time only on the last row to reduce clutter.
        axes[row - 1, 0].set_xlabel(r"$t$ [s]")
        axes[row - 1, 1].set_xlabel(r"$t$ [s]")

    plt.tight_layout()
    plt.show()


@njit(parallel=True, fastmath=True, cache=True)
def w_synthetic(r, t):
    """Synthetic 2D velocity field (Numba-accelerated).

    Parameters
    ----------
    r : ndarray, shape (..., 2)
        Positions in meters.
    t : float
        Time in seconds.

    Returns
    -------
    ndarray
        Velocities in m/s, same shape as r.
    """
    # Tuning knobs (1.0 is the baseline configuration).
    SPATIAL_FREQ = 1.7  # >1 gives finer spatial structure.
    TEMPORAL_FREQ = 1.0  # >1 changes faster in time.

    # Domain scale in meters: larger -> smoother field, smaller -> more structure.
    L_M = 1.2e6

    # Prepare output and flatten for the parallel loop.
    out = np.empty(r.shape, dtype=np.float64)
    r2 = r.reshape((-1, 2))
    o2 = out.reshape((-1, 2))

    two_pi = 2.0 * np.pi
    u = TEMPORAL_FREQ * (two_pi * t / 86400.0)  # daily scale

    # Time-dependent scalars used by multiple components.
    theta = (u / 8.0) + 0.25 * np.sin(u / 5.0) + 0.18 * np.cos(u / 7.3)
    ct = np.cos(theta)
    st = np.sin(theta)

    base_amp = 2.2 + 0.55 * np.sin(u / 3.1) + 0.25 * np.sin(u / 5.7 + 0.8 * np.sin(u / 7.9))

    sm_t1 = 0.8 * np.sin(u / 6.2)
    sm_t2 = 0.6 * np.cos(u / 4.1)
    sm_t3 = 0.5 * np.cos(u / 8.4)

    xd_t1 = 0.07 * np.sin(u / 5.3)
    xd_t2 = 0.04 * np.sin(u / 2.1 + 0.6 * np.cos(u / 7.7))
    yd_t1 = 0.06 * np.cos(u / 6.1)
    yd_t2 = -0.03 * np.sin(u / 2.7 + 0.5 * np.sin(u / 4.3))

    xw_t = 0.3 * np.sin(u / 3.0)
    yw_t = 0.4 * np.cos(u / 4.4)

    rad_t = 1.05 * np.sin(u / 2.2 + 0.3 * np.sin(u / 7.1))
    rad_phase = 0.5 * np.cos(u / 5.3)

    c1x = 0.55 * np.cos(u / 5.7 + 0.30 * np.sin(u / 2.3))
    c1y = 0.55 * np.sin(u / 6.2 + 0.25 * np.cos(u / 3.1))
    c2x = -0.50 * np.cos(u / 6.8 + 0.22 * np.sin(u / 3.7))
    c2y = 0.50 * np.sin(u / 5.4 + 0.28 * np.cos(u / 4.9))
    c3x = 0.20 * np.cos(u / 3.9 + 0.35 * np.sin(u / 6.1))
    c3y = -0.25 * np.sin(u / 4.6 + 0.31 * np.cos(u / 7.2))

    gust_amp = 0.85 + 0.25 * np.sin(u / 1.9 + 0.4 * np.sin(u / 6.7))

    # Turbulence phase offsets.
    n_modes = _K.shape[0]
    phi = np.empty(n_modes, dtype=np.float64)
    for j in range(n_modes):
        pj = _P[j]
        qj = _Q[j]
        phi[j] = 0.9 * np.sin(u / pj) + 0.35 * np.cos(u / qj + 0.6 * np.sin(u / (pj + qj)))

    # Spatial frequency scaling.
    sf = SPATIAL_FREQ
    w1a = sf * 1.5
    w1b = sf * 1.2
    w2a = sf * 1.1
    w2b = sf * 1.4
    b1a = sf * 1.1
    b1b = sf * 1.7
    b2a = sf * 2.3
    b2b = sf * 1.2
    b3a = sf * 1.9
    rxy = sf * 1.2
    rlin_a = sf * 2.1
    rlin_b = sf * 1.3

    for i in prange(r2.shape[0]):
        x_m = r2[i, 0]
        y_m = r2[i, 1]

        # Dimensionless coordinates.
        X = x_m / L_M
        Y = y_m / L_M

        # Drift and warp.
        Xd = X + xd_t1 + xd_t2
        Yd = Y + yd_t1 + yd_t2
        Xw = Xd + 0.05 * np.sin(w1a * Xd + w1b * Yd + xw_t)
        Yw = Yd + 0.05 * np.cos(w2a * Xd - w2b * Yd + yw_t)

        R2 = Xw * Xw + Yw * Yw

        # Base flow with spatial modulation.
        spatial_mod = (
            1.0
            + 0.22 * np.sin(b1a * Xw + b1b * Yw + sm_t1)
            + 0.17 * np.cos(b2a * Xw - b2b * Yw + sm_t2)
            + 0.10 * np.sin(b3a * (Xw * Xw - 0.7 * Yw * Yw) + sm_t3)
        )
        base_x = base_amp * spatial_mod * ct
        base_y = base_amp * spatial_mod * st

        # Radial component with decay.
        radial_amp = rad_t * (
            1.0
            + 0.25 * np.cos(rxy * (Xw * Yw))
            + 0.12 * np.sin(rlin_a * Xw - rlin_b * Yw + rad_phase)
        )
        radial_decay = np.exp(-0.55 * R2)
        radial_x = radial_amp * Xw * radial_decay
        radial_y = radial_amp * Yw * radial_decay

        # Vortices (three rotating cores).
        dx = Xw - c1x
        dy = Yw - c1y
        e1 = np.exp(-3.6 * (dx * dx + dy * dy))
        vortex1_x = 4.8 * (-dy) * e1
        vortex1_y = 4.8 * (dx) * e1

        dx = Xw - c2x
        dy = Yw - c2y
        e2 = np.exp(-3.8 * (dx * dx + dy * dy))
        vortex2_x = -4.4 * (-dy) * e2
        vortex2_y = -4.4 * (dx) * e2

        dx = Xw - c3x
        dy = Yw - c3y
        e3 = np.exp(-5.5 * (dx * dx + dy * dy))
        vortex3_x = 2.9 * (-dy) * e3
        vortex3_y = 2.9 * (dx) * e3

        # Curl-noise turbulence from fixed modes.
        turb_x = 0.0
        turb_y = 0.0
        for j in range(n_modes):
            kx = sf * _K[j, 0]
            ky = sf * _K[j, 1]
            ph = Xw * kx + Yw * ky + phi[j]
            cph = np.cos(ph)
            a = _A[j] * cph
            turb_x += a * ky
            turb_y += a * (-kx)

        env = 0.70 + 0.30 * np.exp(-0.35 * R2)
        turb_scale = gust_amp * env
        turb_x *= turb_scale
        turb_y *= turb_scale

        # Mild shear to bias the flow.
        sh_e = np.exp(-0.35 * R2)
        shear_x = 0.32 * Yw * sh_e
        shear_y = 0.32 * (0.55 * Xw) * sh_e

        # Sum all components.
        o2[i, 0] = base_x + radial_x + vortex1_x + vortex2_x + vortex3_x + turb_x + shear_x
        o2[i, 1] = base_y + radial_y + vortex1_y + vortex2_y + vortex3_y + turb_y + shear_y

    return out


def visualize_vector_field(
    L=2_000_000.0,
    nx=41,
    ny=41,
    n_days=30,
    day=24 * 3600.0,
    out_path="vector_field_hourly.mp4",
    hour=3600.0,
    stride=1,
    fps=8,
    dpi=90,
    scale=47.0,
    width=0.00175,
    interval=120,
    bitrate=1800,
):
    """Generate and save an MP4 animation of the synthetic vector field.

    Parameters are in SI units unless noted. The output file is written to
    out_path using ffmpeg via Matplotlib.
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    x = np.linspace(-0.5 * L, 0.5 * L, nx)
    y = np.linspace(-0.5 * L, 0.5 * L, ny)
    xx, yy = np.meshgrid(x, y)
    positions = np.stack([xx, yy], axis=-1)

    ny = y.size
    nx = x.size

    XX, YY = np.meshgrid(x / 1000.0, y / 1000.0)
    t_anim = np.arange(0.0, n_days * day + hour, hour)

    # Precompute the full space-time field to keep animation simple.
    W_anim = np.empty((t_anim.size, ny, nx, 2), dtype=float)
    for i, t in enumerate(t_anim):
        W_anim[i] = w_synthetic(positions, t)

    XXs = XX[::stride, ::stride]
    YYs = YY[::stride, ::stride]

    fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)
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
