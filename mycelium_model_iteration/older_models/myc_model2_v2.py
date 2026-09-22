import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Linear-diameter growth -> sphere volume simulation
# - time from t_start to t_end (hrs)
# - diameter grows linearly: D(t) = D0 + dD_dt * (t - t_start)
# - split sphere into N_shells concentric shells (equal radial spacing)
# - hyphal fraction: r <= 50 um -> 0.21; r > 50 -> y = a * exp(-n * (r - 50))
#   where a,n come from the provided table at specific knot times.
#   For intermediate hours use the previous knot's a/n.
# --------------------------------------------------

# Simulation timing
t_start = 9
t_end = 50
dt = 1

# Number of radial shells
N_shells = 200

# Knot table (from attachment)
# hrs:    12       18       24       32       48
a_knots = np.array([0.808454, 0.529138, 0.443178, 0.419051, 0.382963], dtype=float)
n_knots = np.array([0.026254, 0.018894, 0.014035, 0.013273, 0.01297], dtype=float)
knot_hours = np.array([12, 18, 24, 32, 48], dtype=float)

# constant core value for r <= 50 µm
core_radius = 50.0
core_value = 0.21


def radius_from_volume_um3(V_um3: float) -> float:
	return (3.0 * V_um3 / (4.0 * np.pi)) ** (1.0 / 3.0)


def volume_from_radius_um(R_um: float) -> float:
	return (4.0 / 3.0) * np.pi * R_um ** 3

def get_coeffs_for_time(t_hr: float):
	"""Return (a,n) using linear interpolation across knot times.
	For times before the first knot, use the first knot values.
	For times after the last knot, use the last knot values."""
	a = np.interp(t_hr, knot_hours, a_knots, left=a_knots[0], right=a_knots[-1])
	n = np.interp(t_hr, knot_hours, n_knots, left=n_knots[0], right=n_knots[-1])
	return a, n


# Set initial diameter at t_start explicitly (user-specified)
# The user requested D(t_start=9hr) = 199.77 µm
D0 = 199.77

# Default linear diameter growth rate (µm per hour).
# You can change this value directly if you want a different linear slope.
# I set a reasonable default (14.06 µm/hr) — change as needed.
dD_dt = 14.06

# Prepare series storage
times = np.arange(t_start, t_end + 1e-9, dt)
n_steps = times.size
D_series = np.zeros(n_steps)
R_series = np.zeros(n_steps)
V_series = np.zeros(n_steps)
myc_vol_um3_series = np.zeros(n_steps)
frac_profiles = []  # store per-time fraction arrays (optional)

for i, t in enumerate(times):
	# linear diameter
	D_t = D0 + dD_dt * (t - t_start)
	R_t = D_t / 2.0
	V_t = volume_from_radius_um(R_t)

	# radial shell edges and midpoints (µm)
	r_edges = np.linspace(0.0, R_t, N_shells + 1)
	r_in = r_edges[:-1]
	r_out = r_edges[1:]
	r_mid = 0.5 * (r_in + r_out)

	# shell volumes (µm^3)
	shell_volumes = (4.0 / 3.0) * np.pi * (r_out ** 3 - r_in ** 3)

	# hyphal fraction per shell midpoint (regression uses r_mid directly, no offset needed)
	a, n = get_coeffs_for_time(t)
	frac = a * np.exp(-n * r_mid)

	# total mycelium volume (µm^3)
	myc_vol_um3 = np.sum(shell_volumes * frac)

	# store
	D_series[i] = D_t
	R_series[i] = R_t
	V_series[i] = V_t
	myc_vol_um3_series[i] = myc_vol_um3
	frac_profiles.append(frac)

# Diagnostic prints
print("Simulation complete")
print(f"t_start={t_start}, t_end={t_end}, dt={dt}, N_shells={N_shells}")
print(f"D0={D0:.3f} µm, dD_dt={dD_dt:.6f} µm/hr")
print("Sample outputs (first 5 steps):")
for j in range(min(5, n_steps)):
	print(f"t={times[j]:.0f} hr: D={D_series[j]:.3f} µm, R={R_series[j]:.3f} µm, V={V_series[j]:.3e} µm^3, myc_vol={myc_vol_um3_series[j]:.3e} µm^3")

# Optionally plot total mycelium volume vs time for a quick check
plt.figure(figsize=(8, 4))
plt.plot(times, myc_vol_um3_series, '-o', label='Mycelium volume (µm^3)')
plt.plot(times, V_series, '-s', label='Total sphere volume (µm^3)')
plt.xlabel('Time (hr)')
plt.ylabel('Volume (µm^3)')
plt.legend()
plt.title('Linear-diameter growth: sphere vs mycelium volume')
plt.grid()
plt.show()

# --------------------------------------------------
# Heatmap plots: all heat times (12, 18, 24, 32, 48 hr) on one figure
# Shows concentric shells colored by hyphal fraction
# --------------------------------------------------
heat_times = [12, 18, 24, 32, 48]

# Find indices in times array corresponding to heat_times
heat_indices = {}
for ht in heat_times:
	idx = np.where(np.abs(times - ht) < 0.5)[0]
	if idx.size > 0:
		heat_indices[ht] = idx[0]

# Color scale consistent across all times (0 to 0.22)
vmin, vmax = 0.0, 0.22
cmap = "viridis"

# Find maximum radius across all heat times to scale display sizes
max_radius = 0.0
for ht in heat_times:
	idx = np.where(np.abs(times - ht) < 0.5)[0]
	if idx.size > 0:
		heat_indices[ht] = idx[0]
		max_radius = max(max_radius, R_series[idx[0]])

Npix = 500
max_plot_size = 1.5  # Maximum display size for the largest circle

# Create single multi-panel figure (2 rows x 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
axes = axes.flatten()  # flatten to 1D for easy iteration
last_im = None

for i, ht in enumerate(heat_times):
	ax = axes[i]
	if ht not in heat_indices:
		continue

	idx = heat_indices[ht]
	R_t = R_series[idx]  # physical radius at this time (µm)
	frac_t = frac_profiles[idx]  # hyphal fraction array for this time

	# Scale display size proportionally to actual radius
	R_plot = R_t / max_radius * max_plot_size  # Largest sphere will be max_plot_size units

	# Pixel grid for this specific circle size
	x = np.linspace(-R_plot, R_plot, Npix)
	y = np.linspace(-R_plot, R_plot, Npix)
	X, Y = np.meshgrid(x, y)
	R_display = np.sqrt(X**2 + Y**2)
	s = np.clip(R_display / R_plot, 0.0, 1.0)  # normalized radius (0..1)

	# Map display radius -> physical radius
	r_phys = s * R_t  # µm

	# Map physical radius to shell index
	r_edges = np.linspace(0.0, R_t, N_shells + 1)
	shell_idx = np.digitize(r_phys, r_edges) - 1
	shell_idx = np.clip(shell_idx, 0, N_shells - 1)

	# Create fraction map by looking up shell index
	frac_map = frac_t[shell_idx]
	frac_map = np.where(R_display <= R_plot, frac_map, np.nan)

	last_im = ax.imshow(
		frac_map,
		origin="lower",
		extent=[-R_plot, R_plot, -R_plot, R_plot],
		vmin=vmin, vmax=vmax,
		cmap=cmap,
		interpolation="nearest",
	)

	ax.set_title(f"{ht} h")
	ax.set_aspect("equal")
	# Set consistent axis limits for all subplots so circles appear proportionally sized
	ax.set_xlim(-max_plot_size, max_plot_size)
	ax.set_ylim(-max_plot_size, max_plot_size)
	ax.set_xticks([])
	ax.set_yticks([])

	# Scale bar (constant physical length across all plots)
	scale_len_um = 100.0  # Fixed at 100 µm for all plots
	scale_len_disp = (scale_len_um / (2 * R_t)) * (2 * R_plot)

	x0 = -0.85 * max_plot_size  # Position relative to axis limits
	y0 = -0.85 * max_plot_size
	ax.plot([x0, x0 + scale_len_disp], [y0, y0], linewidth=4, color="black", solid_capstyle="butt")
	ax.text(x0, y0 + 0.06 * max_plot_size, f"{int(scale_len_um)} µm", color="black", fontsize=9, va="bottom")

# Hide the unused sixth subplot
axes[5].axis('off')

# Shared colorbar
cbar = fig.colorbar(last_im, ax=axes[:5], shrink=0.9)
cbar.set_label("Hyphal fraction")

plt.suptitle("Hyphal fraction heatmaps (proportional sizes; scale bars show true size)")
plt.show()

# --------------------------------------------------
# Citric acid production section
# --------------------------------------------------
prod_rate_max = 14.0  # mg / g DM / hr (at outermost shell)
rho_mycelium = 150.0  # kg / m^3

# Unit conversions
UM3_TO_M3 = 1e-18
KG_TO_G = 1000.0
rho_g_per_um3 = rho_mycelium * KG_TO_G * UM3_TO_M3  # g / µm^3

# For each timestep, compute citric acid production
ca_prod_rate_series = np.zeros_like(times)  # mg/hr per timestep
ca_cumulative_series = np.zeros_like(times)  # cumulative mg

cumulative_mg = 0.0

for i, t in enumerate(times):
	R_t = R_series[i]
	frac_t = frac_profiles[i]
	
	# Radial edges and midpoints
	r_edges = np.linspace(0.0, R_t, N_shells + 1)
	r_in = r_edges[:-1]
	r_out = r_edges[1:]
	r_mid = 0.5 * (r_in + r_out)
	
	# Shell volumes (µm^3)
	shell_volumes = (4.0 / 3.0) * np.pi * (r_out ** 3 - r_in ** 3)
	
	# Production rate decay (exponential, outermost = 14, decaying toward center)
	# Use normalized radius: 0 at center, 1 at edge
	r_norm = r_mid / R_t
	
	# Exponential decay: prod_rate(r) = prod_rate_max * exp(-k * (1 - r_norm))
	# where (1 - r_norm) = 0 at edge (full rate), increases toward center (lower rate)
	# Choose decay constant k to make the decay noticeable; k=2 is reasonable.
	k = 2.0
	prod_rate_decay = prod_rate_max * np.exp(-k * (1.0 - r_norm))
	
	# Mycelium volume and mass per shell (µm^3 and g)
	myc_vol_per_shell = shell_volumes * frac_t  # µm^3
	myc_mass_per_shell = myc_vol_per_shell * rho_g_per_um3  # g
	
	# Production per shell (mg/hr)
	prod_per_shell = myc_mass_per_shell * prod_rate_decay  # mg/hr
	
	# Total production rate (mg/hr) and cumulative (mg)
	prod_rate_mg_hr = np.sum(prod_per_shell)
	cumulative_mg += prod_rate_mg_hr * dt
	
	ca_prod_rate_series[i] = prod_rate_mg_hr
	ca_cumulative_series[i] = cumulative_mg

# --------------------------------------------------
# Plot 1: Hyphal fraction + Production rate vs radius (2x3 grid for heat times)
# --------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
axes = axes.flatten()

for i, ht in enumerate(heat_times):
	if ht not in heat_indices:
		continue

	idx = heat_indices[ht]
	R_t = R_series[idx]
	
	# Recompute r_mid for this time
	r_edges = np.linspace(0.0, R_t, N_shells + 1)
	r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
	shell_volumes = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
	
	# Hyphal fraction: piecewise (0.21 for r <= 50, else exponential decay from your data)
	a, n = get_coeffs_for_time(ht)
	frac = np.where(r_mid <= core_radius, core_value, a * np.exp(-n * r_mid))
	# Cap exponential at core_value so it doesn't exceed 0.21
	frac = np.minimum(frac, core_value)
	
	# Production rate decay (exponential, 14 at edge, decaying toward center)
	# Normalized radius: 0 at center, 1 at edge
	r_norm = r_mid / R_t
	
	# Exponential decay: full rate 14 at edge (r_norm=1), decays toward center
	k = 2.0
	prod_rate_decay = prod_rate_max * np.exp(-k * (1.0 - r_norm))
	
	ax = axes[i]
	
	# Plot hyphal fraction on left y-axis
	ax2 = ax.twinx()
	
	line1 = ax.plot(r_mid, frac, 'b-', linewidth=2, label='Hyphal fraction')
	ax.set_xlabel('Radius (µm)')
	ax.set_ylabel('Hyphal fraction', color='b')
	ax.tick_params(axis='y', labelcolor='b')
	ax.set_ylim([0, 0.25])  # Set reasonable y-limit for hyphal fraction
	
	# Plot production rate on right y-axis
	line2 = ax2.plot(r_mid, prod_rate_decay, 'r-', linewidth=2, label='CA prod. rate (mg/g/hr)')
	ax2.set_ylabel('CA production rate (mg/g/hr)', color='r')
	ax2.tick_params(axis='y', labelcolor='r')
	ax2.set_ylim([0, 15])  # Production rate should be 0-14 range
	
	ax.set_title(f"{ht} h")
	ax.grid(True, alpha=0.3)
	
	# Combined legend
	lines = line1 + line2
	labels = [l.get_label() for l in lines]
	ax.legend(lines, labels, loc='upper left', fontsize=8)

# Hide the unused sixth subplot
axes[5].axis('off')

plt.suptitle("Hyphal fraction and citric acid production rate vs radius")
plt.show()

# --------------------------------------------------
# Plot 1b: Hyphal fraction + mg/hr production vs radius (2x3 grid for heat times)
# --------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
axes = axes.flatten()

for i, ht in enumerate(heat_times):
	if ht not in heat_indices:
		continue

	idx = heat_indices[ht]
	R_t = R_series[idx]
	
	# Recompute r_mid for this time
	r_edges = np.linspace(0.0, R_t, N_shells + 1)
	r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
	shell_volumes = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
	
	# Hyphal fraction: piecewise
	a, n = get_coeffs_for_time(ht)
	frac = np.where(r_mid <= core_radius, core_value, a * np.exp(-n * r_mid))
	frac = np.minimum(frac, core_value)
	
	# Production rate decay (mg/g/hr)
	r_norm = r_mid / R_t
	k = 2.0
	prod_rate_decay = prod_rate_max * np.exp(-k * (1.0 - r_norm))
	
	# Mycelium mass and total mg/hr per shell
	myc_vol_per_shell = shell_volumes * frac
	myc_mass_per_shell = myc_vol_per_shell * rho_g_per_um3
	prod_per_shell_mg_hr = myc_mass_per_shell * prod_rate_decay
	
	ax = axes[i]
	
	# Plot hyphal fraction on left y-axis
	ax2 = ax.twinx()
	
	line1 = ax.plot(r_mid, frac, 'b-', linewidth=2, label='Hyphal fraction')
	ax.set_xlabel('Radius (µm)')
	ax.set_ylabel('Hyphal fraction', color='b')
	ax.tick_params(axis='y', labelcolor='b')
	ax.set_ylim([0, 0.25])
	
	# Plot mg/hr production on right y-axis
	line2 = ax2.plot(r_mid, prod_per_shell_mg_hr, 'r-', linewidth=2, label='CA prod. rate (mg/hr)')
	ax2.set_ylabel('CA production rate (mg/hr)', color='r')
	ax2.tick_params(axis='y', labelcolor='r')
	
	ax.set_title(f"{ht} h")
	ax.grid(True, alpha=0.3)
	
	# Combined legend
	lines = line1 + line2
	labels = [l.get_label() for l in lines]
	ax.legend(lines, labels, loc='upper left', fontsize=8)

# Hide the unused sixth subplot
axes[5].axis('off')

plt.suptitle("Hyphal fraction and citric acid production (mg/hr) vs radius")
plt.show()

# --------------------------------------------------
# Plot 2: Cumulative citric acid production over time
# --------------------------------------------------
plt.figure(figsize=(10, 5))
ax1 = plt.gca()
ax2 = ax1.twinx()

line1 = ax1.plot(times, ca_cumulative_series, '-o', linewidth=2, markersize=4, color='blue', label='Cumulative CA production')
ax1.set_xlabel('Time (hr)')
ax1.set_ylabel('Cumulative CA production (mg)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

line2 = ax2.plot(times, myc_vol_um3_series, '-s', linewidth=2, markersize=4, color='green', label='Hyphal volume')
ax2.set_ylabel('Hyphal volume (µm³)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Set y-axis limits with padding to prevent lines from converging
ax1.set_ylim([0, ca_cumulative_series[-1] * 1.2])
ax2.set_ylim([0, myc_vol_um3_series[-1] * 1.15])

plt.title('Citric acid production and hyphal volume over time')
ax1.grid(True, alpha=0.3)

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.tight_layout()
plt.show()

print("\nCitric acid production summary:")
print(f"Initial production rate (t={times[0]:.0f}h): {ca_prod_rate_series[0]:.3f} mg/hr")
print(f"Final production rate (t={times[-1]:.0f}h): {ca_prod_rate_series[-1]:.3f} mg/hr")
print(f"Total cumulative CA production: {ca_cumulative_series[-1]:.3f} mg")

# --------------------------------------------------
# Final Results Summary
# --------------------------------------------------
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

# Final total citric acid production
final_ca_production = ca_cumulative_series[-1]
print(f"\nFinal total citric acid production: {final_ca_production:.10f} mg")
print(f"                                     {final_ca_production:.6e} mg (scientific notation)")

# Final hyphal volume
final_hyphal_volume = myc_vol_um3_series[-1]
print(f"\nFinal hyphal volume: {final_hyphal_volume:.10e} µm³")

# Final hyphal mass
final_hyphal_mass = final_hyphal_volume * rho_g_per_um3
print(f"Final hyphal mass: {final_hyphal_mass:.10e} g")
print(f"                   {final_hyphal_mass * 1e6:.10f} µg")

# Fit exponential model to citric acid production over time
# Model: CA(t) = a * (1 - exp(-b * (t - t_start)))
# Or simpler: CA(t) = a * exp(b * t) + c
from scipy.optimize import curve_fit

# Try exponential growth model: CA(t) = a * exp(b * t) + c
def exp_model(t, a, b, c):
    return a * np.exp(b * t) + c

# Fit the model
try:
    popt, pcov = curve_fit(exp_model, times, ca_cumulative_series, p0=[1, 0.05, 0], maxfev=10000)
    a_fit, b_fit, c_fit = popt
    
    print(f"\nExponential model for citric acid production:")
    print(f"CA(t) = {a_fit:.4e} * exp({b_fit:.6f} * t) + {c_fit:.4f}")
    print(f"  where t is time in hours")
    
    # Calculate R-squared
    residuals = ca_cumulative_series - exp_model(times, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ca_cumulative_series - np.mean(ca_cumulative_series))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"  R² = {r_squared:.6f}")
    
except Exception as e:
    print(f"\nCould not fit exponential model: {e}")

print("\n" + "="*60)




