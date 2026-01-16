"""Utility functions used by the CV3 notebook.

The notebook relies on a synthetic, time-varying 2D velocity field and a
quadratic-drag drift model. Plotting helpers import matplotlib lazily.
"""

import numpy as np
from numba import njit

SECONDS_PER_DAY = 86400.0
METERS_PER_KM = 1000.0

# Wave vectors and weights for curl-noise-like turbulence.
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


@njit(fastmath=True, cache=True)
def w_synthetic_point(
    x_m: float,
    y_m: float,
    t: float,
    spatial_freq: float = 1.7,
    temporal_freq: float = 1.0,
    L_M: float = 1.2e6,
):
    """Return synthetic velocity at a single point.

    Parameters
    ----------
    x_m, y_m : float
        Position in meters.
    t : float
        Time in seconds.
    spatial_freq, temporal_freq : float
        Multipliers for spatial/temporal variability.
    L_M : float
        Spatial scaling length in meters.

    Returns
    -------
    wx, wy : float
        Velocity components in m/s.
    """
    u = temporal_freq * (2.0 * np.pi * t / SECONDS_PER_DAY)

    # Time modulation.
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

    # Spatial frequency scalars.
    sf = spatial_freq

    # Warp.
    w1a = sf * 1.5
    w1b = sf * 1.2
    w2a = sf * 1.1
    w2b = sf * 1.4

    # Base spatial modulation.
    b1a = sf * 1.1
    b1b = sf * 1.7
    b2a = sf * 2.3
    b2b = sf * 1.2
    b3a = sf * 1.9

    # Radial spatial terms.
    rxy = sf * 1.2
    rlin_a = sf * 2.1
    rlin_b = sf * 1.3

    # Nondimensional coordinates.
    X = x_m / L_M
    Y = y_m / L_M

    # Drift.
    Xd = X + xd_t1 + xd_t2
    Yd = Y + yd_t1 + yd_t2

    # Warp.
    Xw = Xd + 0.05 * np.sin(w1a * Xd + w1b * Yd + xw_t)
    Yw = Yd + 0.05 * np.cos(w2a * Xd - w2b * Yd + yw_t)

    R2 = Xw * Xw + Yw * Yw

    # Base flow.
    spatial_mod = (
        1.0
        + 0.22 * np.sin(b1a * Xw + b1b * Yw + sm_t1)
        + 0.17 * np.cos(b2a * Xw - b2b * Yw + sm_t2)
        + 0.10 * np.sin(b3a * (Xw * Xw - 0.7 * Yw * Yw) + sm_t3)
    )
    base_x = base_amp * spatial_mod * ct
    base_y = base_amp * spatial_mod * st

    # Radial components.
    radial_amp = rad_t * (
        1.0
        + 0.25 * np.cos(rxy * (Xw * Yw))
        + 0.12 * np.sin(rlin_a * Xw - rlin_b * Yw + rad_phase)
    )
    radial_decay = np.exp(-0.55 * R2)
    radial_x = radial_amp * Xw * radial_decay
    radial_y = radial_amp * Yw * radial_decay

    # Vortices.
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

    # Turbulence (curl-noise style).
    turb_x = 0.0
    turb_y = 0.0
    n_modes = _K.shape[0]
    for j in range(n_modes):
        pj = _P[j]
        qj = _Q[j]
        phi_j = 0.9 * np.sin(u / pj) + 0.35 * np.cos(u / qj + 0.6 * np.sin(u / (pj + qj)))

        kx = sf * _K[j, 0]
        ky = sf * _K[j, 1]
        ph = Xw * kx + Yw * ky + phi_j
        cph = np.cos(ph)
        aa = _A[j] * cph
        turb_x += aa * ky
        turb_y += aa * (-kx)

    env = 0.70 + 0.30 * np.exp(-0.35 * R2)
    turb_scale = gust_amp * env
    turb_x *= turb_scale
    turb_y *= turb_scale

    # Shear.
    sh_e = np.exp(-0.35 * R2)
    shear_x = 0.32 * Yw * sh_e
    shear_y = 0.32 * (0.55 * Xw) * sh_e

    wx = base_x + radial_x + vortex1_x + vortex2_x + vortex3_x + turb_x + shear_x
    wy = base_y + radial_y + vortex1_y + vortex2_y + vortex3_y + turb_y + shear_y
    return wx, wy


@njit(fastmath=True, cache=True)
def _drag_accel(vx: float, vy: float, wx: float, wy: float, kappa: float):
    """Return quadratic-drag acceleration components."""
    vrel_x = vx - wx
    vrel_y = vy - wy
    vrel_n = np.sqrt(vrel_x * vrel_x + vrel_y * vrel_y)
    ax = -kappa * vrel_n * vrel_x
    ay = -kappa * vrel_n * vrel_y
    return ax, ay


@njit(fastmath=True, cache=True)
def simulate_drag_w_synthetic(
    r0: np.ndarray,
    t0: float,
    kappa: float,
    T: float,
    dt: float = 100.0,
    spatial_freq: float = 1.7,
    temporal_freq: float = 1.0,
    L_M: float = 1.2e6,
):
    """Simulate quadratic-drag drift in the synthetic velocity field.

    Model
    -----
    v_rel = v - w(r,t)
    a     = -kappa * |v_rel| * v_rel

    Integration
    -----------
    - forward Euler for velocity
    - trapezoid rule for position

    Assumptions
    -----------
    - units: meters / seconds
    - initial velocity equals the field: v0 = w(r0, t0)
    - dt is fixed by the caller

    Returns
    -------
    t : (n,) float64 [s]
    r : (n,2) float64 [m]
    v : (n,2) float64 [m/s]
    a : (n,2) float64 [m/s^2]
    """
    n = int(np.floor(T / dt)) + 1

    # Pre-allocate arrays for numba performance.
    t = np.empty(n, dtype=np.float64)
    r = np.empty((n, 2), dtype=np.float64)
    v = np.empty((n, 2), dtype=np.float64)
    a = np.empty((n, 2), dtype=np.float64)

    r[0, 0] = r0[0]
    r[0, 1] = r0[1]
    t[0] = t0

    # Initial velocity equals the field.
    wx, wy = w_synthetic_point(r[0, 0], r[0, 1], t0, spatial_freq, temporal_freq, L_M)
    v[0, 0] = wx
    v[0, 1] = wy
    a[0, 0], a[0, 1] = _drag_accel(v[0, 0], v[0, 1], wx, wy, kappa)

    for i in range(1, n):
        ti = t[i - 1]

        # Wind at current state.
        wx, wy = w_synthetic_point(r[i - 1, 0], r[i - 1, 1], ti, spatial_freq, temporal_freq, L_M)

        # Acceleration at time ti (stored at i-1).
        ax, ay = _drag_accel(v[i - 1, 0], v[i - 1, 1], wx, wy, kappa)
        a[i - 1, 0] = ax
        a[i - 1, 1] = ay

        # Euler update.
        vx_new = v[i - 1, 0] + ax * dt
        vy_new = v[i - 1, 1] + ay * dt
        v[i, 0] = vx_new
        v[i, 1] = vy_new

        # Position update (trapezoid).
        r[i, 0] = r[i - 1, 0] + 0.5 * (v[i - 1, 0] + vx_new) * dt
        r[i, 1] = r[i - 1, 1] + 0.5 * (v[i - 1, 1] + vy_new) * dt

        t[i] = ti + dt

    # Final acceleration at (r_n, t_n).
    wx, wy = w_synthetic_point(r[n - 1, 0], r[n - 1, 1], t[n - 1], spatial_freq, temporal_freq, L_M)
    a[n - 1, 0], a[n - 1, 1] = _drag_accel(v[n - 1, 0], v[n - 1, 1], wx, wy, kappa)

    return t, r, v, a


def _extent_km(xg, yg):
    """Return imshow extent in km for 1D grids in meters."""
    return [
        xg[0] / METERS_PER_KM,
        xg[-1] / METERS_PER_KM,
        yg[0] / METERS_PER_KM,
        yg[-1] / METERS_PER_KM,
    ]


def show_map(data, xg, yg, title, cbar_label, vmin=None, vmax=None):
    """Plot a 2D map on a regular grid with km axes."""
    import matplotlib.pyplot as plt

    extent = _extent_km(xg, yg)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set(xlabel="x [km]", ylabel="y [km]", title=title)
    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.grid(False)
    plt.show()


def show_like_post(like_data, post_data, xg, yg, title_like, title_post, u_true, u_map):
    """Plot likelihood and posterior side by side (no colorbars)."""
    import matplotlib.pyplot as plt

    extent = _extent_km(xg, yg)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(like_data, origin="lower", extent=extent, aspect="equal")
    axes[0].set(xlabel="x [km]", ylabel="y [km]", title=title_like)

    axes[1].imshow(post_data, origin="lower", extent=extent, aspect="equal")
    axes[1].set(xlabel="x [km]", ylabel="y [km]", title=title_post)
    axes[1].scatter(
        u_true[0] / METERS_PER_KM,
        u_true[1] / METERS_PER_KM,
        marker="*",
        s=220,
        c="white",
        edgecolors="black",
        label="u_true",
    )
    axes[1].scatter(
        u_map[0] / METERS_PER_KM,
        u_map[1] / METERS_PER_KM,
        marker="x",
        s=150,
        c="white",
        label="MAP",
    )
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def make_grid(L, n):
    """Return a regular grid on the square [-L, L]^2."""
    xg = np.linspace(-L, L, n)
    yg = np.linspace(-L, L, n)
    XX, YY = np.meshgrid(xg, yg, indexing="xy")
    return xg, yg, XX, YY
