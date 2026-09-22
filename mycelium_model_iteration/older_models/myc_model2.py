import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Constants / settings
# -----------------------------
# Growth
V0 = 0.4e6        # um^3, sphere volume at t_start
mu = 0.09         # 1/hr
t_start = 9.0     # hr
t_end = 50.0      # hr
dt = 1.0          # hr
N_shells = 200    # number of concentric shells for integration

# Citric acid production + mycelium density
prod_rate_mg_per_g_per_hr = 14.0     # mg / g(dry wt) / hr
rho_mycelium_kg_per_m3 = 150.0       # kg / m^3

# Unit conversions
UM3_TO_M3 = 1e-18
KG_TO_G = 1000.0

rho_mycelium_g_per_m3 = rho_mycelium_kg_per_m3 * KG_TO_G  # g/m^3

# -----------------------------
# Hyphal fraction model (piecewise core + exponential tail)
# -----------------------------
rc = 50.0  # um, core radius used for all curves

t_knots = np.array([12, 18, 24, 32, 48], dtype=float)
f0_knots = np.array([0.21, 0.21, 0.215, 0.22, 0.21], dtype=float)
lam_knots = np.array([40, 60, 85, 110, 140], dtype=float)

def hyphal_fraction(r_um: np.ndarray, t_hr: float) -> np.ndarray:
    """Hyphal fraction at radius r (um) and time t (hr), interpolated across the provided time curves."""
    f0 = np.interp(t_hr, t_knots, f0_knots, left=f0_knots[0], right=f0_knots[-1])
    lam = np.interp(t_hr, t_knots, lam_knots, left=lam_knots[0], right=lam_knots[-1])

    r = np.asarray(r_um, dtype=float)
    return np.where(r <= rc, f0, f0 * np.exp(-(r - rc) / lam))

# -----------------------------
# Growth model
# -----------------------------
def sphere_volume(t_hr: float) -> float:
    """
    Sphere volume at time t (hr).
    Interprets V0 as the volume at t_start, so V(t_start) = V0.
    """
    return V0 * np.exp(mu * (t_hr - t_start))

def sphere_radius_from_volume(V_um3: float) -> float:
    return (3.0 * V_um3 / (4.0 * np.pi)) ** (1.0 / 3.0)

# -----------------------------
# Simulation: volume + citric acid production
# -----------------------------
times = np.arange(t_start, t_end + 1e-9, dt)

V_series = np.zeros_like(times)           # um^3
prod_rate_series = np.zeros_like(times)   # mg/hr
prod_cum_series = np.zeros_like(times)    # mg

cumulative_mg = 0.0

for i, t in enumerate(times):
    V = sphere_volume(t)
    R = sphere_radius_from_volume(V)

    # Concentric shells equally spaced in radius
    r_edges = np.linspace(0.0, R, N_shells + 1)
    r_in = r_edges[:-1]
    r_out = r_edges[1:]
    r_mid = 0.5 * (r_in + r_out)

    # Shell volumes (um^3)
    shell_volumes_um3 = (4.0 / 3.0) * np.pi * (r_out**3 - r_in**3)

    # Hyphal fraction at shell midpoint
    frac = hyphal_fraction(r_mid, t)

    # Mycelium volume in each shell (um^3), then total (um^3)
    myc_vol_um3 = np.sum(shell_volumes_um3 * frac)

    # Convert mycelium volume -> dry mass (g)
    myc_mass_g = myc_vol_um3 * UM3_TO_M3 * rho_mycelium_g_per_m3

    # Citric acid production rate (mg/hr) and timestep production (mg)
    prod_rate_mg_per_hr = prod_rate_mg_per_g_per_hr * myc_mass_g
    cumulative_mg += prod_rate_mg_per_hr * dt

    V_series[i] = V
    prod_rate_series[i] = prod_rate_mg_per_hr
    prod_cum_series[i] = cumulative_mg

# -----------------------------
# Plot 1: cumulative citric acid + volume (different colours)
# -----------------------------
fig, ax1 = plt.subplots(figsize=(8, 4.5))

ax1.plot(times, prod_cum_series, color="tab:blue", linewidth=2)
ax1.set_xlabel("Time (hr)")
ax1.set_ylabel("Cumulative citric acid (mg)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(times, V_series, color="tab:orange", linewidth=2)
ax2.set_ylabel("Sphere volume (µm³)", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

plt.title("Cumulative citric acid production and sphere volume vs time")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 2: Constant-size heatmap pellets + per-panel scale bars
# (All circles same plotted size; radial coordinate is warped per time.)
# -----------------------------
heat_times = [12, 18, 24, 32, 48]

# Real pellet radii at each time (physical radii, um)
R_real = {t: sphere_radius_from_volume(sphere_volume(t)) for t in heat_times}

# Fixed plotting radius so every subplot uses the same circle size on screen
R_plot = 1.0  # arbitrary display radius

# Pixel grid in display coordinates
Npix = 500
x = np.linspace(-R_plot, R_plot, Npix)
y = np.linspace(-R_plot, R_plot, Npix)
X, Y = np.meshgrid(x, y)
R_display = np.sqrt(X**2 + Y**2)

# Normalized radius (0..1 inside displayed circle)
s = np.clip(R_display / R_plot, 0.0, 1.0)

# Colour scale constant across all times
vmin, vmax = 0.0, 0.22
cmap = "viridis"

last_im = None

# Create a separate figure for each time in `heat_times` rather than a single multi-panel figure
for t in heat_times:
    R_t = R_real[t]  # physical radius at time t (um)

    fig, ax = plt.subplots(figsize=(4, 4))

    # Map display radius -> physical radius for this time
    r_phys = s * R_t  # um

    # Evaluate hyphal fraction using physical radius
    frac_map = hyphal_fraction(r_phys, t)

    # Mask only outside the displayed circle (constant visual size)
    frac_map = np.where(R_display <= R_plot, frac_map, np.nan)

    im = ax.imshow(
        frac_map,
        origin="lower",
        extent=[-R_plot, R_plot, -R_plot, R_plot],
        vmin=vmin, vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )

    ax.set_title(f"{t} h")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Per-panel scale bar in PHYSICAL units (µm)
    raw_len_um = 0.25 * (2 * R_t)                      # 25% of diameter
    scale_len_um = max(10.0, np.round(raw_len_um / 10) * 10)  # nearest 10 µm, at least 10

    # Convert physical length -> display length
    scale_len_disp = (scale_len_um / (2 * R_t)) * (2 * R_plot)

    # Draw scale bar bottom-left
    x0 = -0.85 * R_plot
    y0 = -0.85 * R_plot
    ax.plot([x0, x0 + scale_len_disp], [y0, y0], linewidth=4, color="black", solid_capstyle="butt")
    ax.text(x0, y0 + 0.06 * R_plot, f"{int(scale_len_um)} µm", color="black", fontsize=9, va="bottom")

    # Colorbar for this figure
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Hyphal fraction")

    plt.suptitle("Hyphal fraction heatmap (constant visual size; scale bar shows true size)")
    plt.show()
