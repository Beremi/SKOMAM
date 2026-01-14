import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# -----------------------------------------------------------------------------
# Synthetic field constants (copied from cv2_utils for self-contained cv3_utils)
# -----------------------------------------------------------------------------
_K = np.array([
    [ 2.2,  -5.7],
    [ 4.9,   3.1],
    [-6.4,   2.4],
    [ 7.8,  -2.6],
    [ 3.0,   7.2],
    [-8.5,  -3.4],
    [ 9.7,   1.1],
    [-1.6,  10.3],
    [ 5.6,  -9.4],
    [-10.8,  4.7],
], dtype=np.float64)

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
    """
    Fast single-point evaluator of the synthetic 2D velocity field.

    Inputs
    ------
    x_m, y_m : float
        Position [m]
    t : float
        Time [s]
    spatial_freq : float
        >1 increases spatial frequency (finer spatial structures).
    temporal_freq : float
        >1 increases temporal frequency (faster temporal changes).
    L_M : float
        Spatial scaling [m]. Larger => smoother field.

    Returns
    -------
    wx, wy : float
        Velocity components [m/s]
    """

    two_pi = 2.0 * np.pi
    u = temporal_freq * (two_pi * t / 86400.0)

    # time scalars
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
    c2y =  0.50 * np.sin(u / 5.4 + 0.28 * np.cos(u / 4.9))
    c3x = 0.20 * np.cos(u / 3.9 + 0.35 * np.sin(u / 6.1))
    c3y = -0.25 * np.sin(u / 4.6 + 0.31 * np.cos(u / 7.2))

    gust_amp = 0.85 + 0.25 * np.sin(u / 1.9 + 0.4 * np.sin(u / 6.7))

    # spatial frequency scalars
    sf = spatial_freq

    # warp
    w1a = sf * 1.5
    w1b = sf * 1.2
    w2a = sf * 1.1
    w2b = sf * 1.4

    # base spatial_mod
    b1a = sf * 1.1
    b1b = sf * 1.7
    b2a = sf * 2.3
    b2b = sf * 1.2
    b3a = sf * 1.9

    # radial spatial pieces
    rxy = sf * 1.2
    rlin_a = sf * 2.1
    rlin_b = sf * 1.3

    # dimensionless coordinates
    X = x_m / L_M
    Y = y_m / L_M

    # drift
    Xd = X + xd_t1 + xd_t2
    Yd = Y + yd_t1 + yd_t2

    # warp
    Xw = Xd + 0.05 * np.sin(w1a * Xd + w1b * Yd + xw_t)
    Yw = Yd + 0.05 * np.cos(w2a * Xd - w2b * Yd + yw_t)

    R2 = Xw * Xw + Yw * Yw

    # base
    spatial_mod = (
        1.0
        + 0.22 * np.sin(b1a * Xw + b1b * Yw + sm_t1)
        + 0.17 * np.cos(b2a * Xw - b2b * Yw + sm_t2)
        + 0.10 * np.sin(b3a * (Xw * Xw - 0.7 * Yw * Yw) + sm_t3)
    )
    base_x = base_amp * spatial_mod * ct
    base_y = base_amp * spatial_mod * st

    # radial
    radial_amp = rad_t * (
        1.0
        + 0.25 * np.cos(rxy * (Xw * Yw))
        + 0.12 * np.sin(rlin_a * Xw - rlin_b * Yw + rad_phase)
    )
    radial_decay = np.exp(-0.55 * R2)
    radial_x = radial_amp * Xw * radial_decay
    radial_y = radial_amp * Yw * radial_decay

    # vortices
    dx = Xw - c1x
    dy = Yw - c1y
    e1 = np.exp(-3.6 * (dx * dx + dy * dy))
    vortex1_x = 4.8 * (-dy) * e1
    vortex1_y = 4.8 * ( dx) * e1

    dx = Xw - c2x
    dy = Yw - c2y
    e2 = np.exp(-3.8 * (dx * dx + dy * dy))
    vortex2_x = -4.4 * (-dy) * e2
    vortex2_y = -4.4 * ( dx) * e2

    dx = Xw - c3x
    dy = Yw - c3y
    e3 = np.exp(-5.5 * (dx * dx + dy * dy))
    vortex3_x = 2.9 * (-dy) * e3
    vortex3_y = 2.9 * ( dx) * e3

    # turbulence (curl-noise style)
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

    # shear
    sh_e = np.exp(-0.35 * R2)
    shear_x = 0.32 * Yw * sh_e
    shear_y = 0.32 * (0.55 * Xw) * sh_e

    wx = base_x + radial_x + vortex1_x + vortex2_x + vortex3_x + turb_x + shear_x
    wy = base_y + radial_y + vortex1_y + vortex2_y + vortex3_y + turb_y + shear_y
    return wx, wy


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
    """
    High-performance drift solver with quadratic drag in the synthetic field.

    Model
    -----
    v_rel = v - w(r,t)
    a     = -kappa * |v_rel| * v_rel

    Time integration
    ----------------
    Forward Euler for velocity:
        v_{n+1} = v_n + a_n * dt

    Trapezoid for position (matches your CV2 notebook):
        r_{n+1} = r_n + 0.5*(v_n + v_{n+1})*dt

    Assumptions (as requested)
    --------------------------
    - units: meters / seconds
    - initial velocity equals the field: v0 = w(r0, t0)
    - dt is fixed by the caller (default: 100 s)

    Returns
    -------
    t : (n,) float64 [s]
    r : (n,2) float64 [m]
    v : (n,2) float64 [m/s]
    a : (n,2) float64 [m/s^2]
    """
    n = int(np.floor(T / dt)) + 1

    t = np.empty(n, dtype=np.float64)
    r = np.empty((n, 2), dtype=np.float64)
    v = np.empty((n, 2), dtype=np.float64)
    a = np.empty((n, 2), dtype=np.float64)

    r[0, 0] = r0[0]
    r[0, 1] = r0[1]
    t[0] = t0

    # initial velocity = field
    wx, wy = w_synthetic_point(r[0, 0], r[0, 1], t0, spatial_freq, temporal_freq, L_M)
    v[0, 0] = wx
    v[0, 1] = wy

    # initial acceleration (v0 == w0 => ~0)
    vrel_x = v[0, 0] - wx
    vrel_y = v[0, 1] - wy
    vrel_n = np.sqrt(vrel_x * vrel_x + vrel_y * vrel_y)
    a[0, 0] = -kappa * vrel_n * vrel_x
    a[0, 1] = -kappa * vrel_n * vrel_y

    for i in range(1, n):
        ti = t0 + (i - 1) * dt

        # wind at current state
        wx, wy = w_synthetic_point(r[i - 1, 0], r[i - 1, 1], ti, spatial_freq, temporal_freq, L_M)

        # a at time ti (stored at i-1)
        vrel_x = v[i - 1, 0] - wx
        vrel_y = v[i - 1, 1] - wy
        vrel_n = np.sqrt(vrel_x * vrel_x + vrel_y * vrel_y)
        ax = -kappa * vrel_n * vrel_x
        ay = -kappa * vrel_n * vrel_y
        a[i - 1, 0] = ax
        a[i - 1, 1] = ay

        # Euler update
        vx_new = v[i - 1, 0] + ax * dt
        vy_new = v[i - 1, 1] + ay * dt
        v[i, 0] = vx_new
        v[i, 1] = vy_new

        # position update (trapezoid)
        r[i, 0] = r[i - 1, 0] + 0.5 * (v[i - 1, 0] + vx_new) * dt
        r[i, 1] = r[i - 1, 1] + 0.5 * (v[i - 1, 1] + vy_new) * dt

        t[i] = ti + dt

    # final acceleration at (r_n, t_n)
    wx, wy = w_synthetic_point(r[n - 1, 0], r[n - 1, 1], t[n - 1], spatial_freq, temporal_freq, L_M)
    vrel_x = v[n - 1, 0] - wx
    vrel_y = v[n - 1, 1] - wy
    vrel_n = np.sqrt(vrel_x * vrel_x + vrel_y * vrel_y)
    a[n - 1, 0] = -kappa * vrel_n * vrel_x
    a[n - 1, 1] = -kappa * vrel_n * vrel_y

    return t, r, v, a


def default_params():
    """Default parameters used in the CV3 notebook."""
    return {
        "T_WINDOW_MIN": 1.0 * 60.0,          # s
        "T_WINDOW_MAX": 16.0 * 60.0,         # s
        "V_PLANE_KMH": 800.0,                # km/h
        "V_PLANE": 800.0 * 1000.0 / 3600.0,  # m/s
        "THETA_MU": 0.5 * np.pi,             # rad
        "THETA_SIG": np.pi / 3.0,            # rad
        "DT_DRIFT": 100.0,                   # s
        "SPATIAL_FREQ": 1.7,
        "TEMPORAL_FREQ": 1.0,
        "L_M": 1.2e6,                        # m
        "SIGMA0_KM": 100.0,                  # km
        "SIGMA_PER_DAY_KM": 10.0,            # km/day
        "GRID_N": 160,
        "THETA_COVER_SIGMAS": 2.5,
        "MIN_FIND_DAY": 3.0,
        "MAX_FIND_DAY": 9.0,
        "N_DEBRIS": 3,
        "RNG_SEED": 12345,
    }


def make_semicircle_grid(V, tmax, grid_n=160):
    """Return (xg, yg, XX, YY) for a semicircle with radius R = V * tmax (y >= 0)."""
    radius = V * tmax
    xg = np.linspace(-radius, radius, grid_n)
    yg = np.linspace(0.0, radius, grid_n)
    XX, YY = np.meshgrid(xg, yg, indexing="xy")
    return xg, yg, XX, YY


def semicircle_mask(XX, YY, radius):
    """Mask for points inside the semicircle of radius R (y >= 0 assumed)."""
    return (XX * XX + YY * YY) <= radius * radius


def mask_outside_semicircle(Z, XX, YY, radius, fill_value=np.nan):
    """Replace values outside the semicircle with fill_value."""
    mask = semicircle_mask(XX, YY, radius)
    return np.where(mask, Z, fill_value)


def make_km_axes(title=None, figsize=(6, 6)):
    """Create axes with km units and equal aspect."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_trajectory_km(
    r,
    ax=None,
    label=None,
    show_start=True,
    show_end=True,
    start_label="start",
    end_label="konec",
):
    """Plot a 2D trajectory (meters) on km axes."""
    if ax is None:
        _, ax = make_km_axes()
    ax.plot(r[:, 0] / 1000.0, r[:, 1] / 1000.0, label=label)
    if show_start:
        ax.scatter([r[0, 0] / 1000.0], [r[0, 1] / 1000.0], label=start_label)
    if show_end:
        ax.scatter([r[-1, 0] / 1000.0], [r[-1, 1] / 1000.0], label=end_label)
    return ax


def plot_pdf_pair(
    x1,
    y1,
    x1_label,
    title1,
    x2,
    y2,
    x2_label,
    title2,
    figsize=(10, 3),
):
    """Plot two 1D PDFs side by side."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(x1, y1)
    axes[0].set_xlabel(x1_label)
    axes[0].set_title(title1)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x2, y2)
    axes[1].set_xlabel(x2_label)
    axes[1].set_title(title2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_heatmap_km(
    Z,
    xg,
    yg,
    title,
    cbar_label,
    ax=None,
    cmap="viridis",
    show_colorbar=True,
):
    """Plot a 2D field with km axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[xg[0] / 1000.0, xg[-1] / 1000.0, yg[0] / 1000.0, yg[-1] / 1000.0],
        aspect="auto",
        cmap=cmap,
    )
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    if show_colorbar:
        plt.colorbar(im, ax=ax, label=cbar_label)
    return ax


def normalize_on_grid(density, xg, yg):
    """Normalize a density on a rectangular grid (Riemann sum)."""
    dx = (xg[-1] - xg[0]) / (len(xg) - 1)
    dy = (yg[-1] - yg[0]) / (len(yg) - 1)
    Z = np.sum(density) * dx * dy
    return density / Z, Z
