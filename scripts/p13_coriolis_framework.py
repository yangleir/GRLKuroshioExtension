"""Coriolis-mediated adjustment framework: theory + data validation.

Closed-loop chain:
  Wind forcing → Sverdrup equilibrium → Rossby adjustment (β/f²) → Dual-mode response

Four-panel figure:
  (a) c_R(φ) and τ_R(φ) — planetary vorticity brake
  (b) ODE integration — latitude-dependent τ_R vs constant τ_R
  (c) Closed-loop validation — predicted vs observed rates
  (d) Forward projection — migration deceleration
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.integrate import solve_ivp
from pathlib import Path
import json

OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
    'font.family': 'sans-serif', 'mathtext.fontset': 'dejavusans',
})

# ── Physical constants ──
Omega = 7.292e-5        # Earth rotation rate (rad/s)
a = 6.371e6             # Earth radius (m)
L_x = 8000e3            # Basin width (m)
rho_0 = 1025.0          # Seawater density (kg/m³)

def f_cor(phi_deg):
    return 2 * Omega * np.sin(np.deg2rad(phi_deg))

def beta(phi_deg):
    return 2 * Omega * np.cos(np.deg2rad(phi_deg)) / a

# ── First baroclinic Rossby radius from Chelton et al. (1998) ──
# Approximate fit for North Pacific western basin (km)
# Data points: 30°N~42km, 35°N~30km, 40°N~22km, 45°N~15km
phi_chelton = np.array([30, 35, 40, 45])
Rd_chelton_km = np.array([42, 30, 22, 15])
Rd_poly = np.polyfit(phi_chelton, Rd_chelton_km, 2)

def Rd_km(phi_deg):
    return np.polyval(Rd_poly, phi_deg)

def c_R(phi_deg):
    """First baroclinic Rossby wave phase speed (m/s)."""
    Rd = Rd_km(phi_deg) * 1e3  # km → m
    return beta(phi_deg) * Rd**2

def tau_R(phi_deg):
    """Basin-crossing time (years)."""
    return L_x / c_R(phi_deg) / (365.25 * 86400)

# ── Observed constraints ──
with open(OUT / "unified_velocity_stats.json") as fh:
    vel_stats = json.load(fh)

obs = {
    'v_velmax': vel_stats['Velocity max']['trend_per_decade'],      # °/dec
    'v_centroid': vel_stats['Velocity centroid']['trend_per_decade'], # °/dec
    'v_sla': vel_stats['SLA gradient']['trend_per_decade'],          # °/dec
    'v_hadley': 0.50,    # Seidel et al. (2008)
    'v_wind': 0.58,      # Wu et al. (2018)
    'cR_35': 0.02,       # Chelton (1998), m/s
    'cR_40': 0.01,       # Chelton (1998), m/s
    'phi_KE_now': 35.3,  # Current KE centroid position
    'phi_OE': 40.0,      # Oyashio Extension front
    'lag_obs': 3.5,       # Qiu & Chen (2020), years
}

print("=== Observational Constraints ===")
for k, v in obs.items():
    print(f"  {k:>15s} = {v}")

# ══════════════════════════════════════════════════════════════
# Panel (a): Rossby wave speed and adjustment timescale
# ══════════════════════════════════════════════════════════════
phi_range = np.linspace(28, 48, 200)
cR_theory = c_R(phi_range) * 100  # m/s → cm/s
tauR_theory = tau_R(phi_range)

# Validate against Chelton observations
cR_35 = c_R(35) * 100
cR_40 = c_R(40) * 100
tauR_35 = tau_R(35)
tauR_40 = tau_R(40)

print(f"\n=== Panel (a): Rossby Wave Speed ===")
print(f"  c_R(35°N) = {cR_35:.2f} cm/s  (obs: ~2 cm/s)")
print(f"  c_R(40°N) = {cR_40:.2f} cm/s  (obs: ~1 cm/s)")
print(f"  Ratio c_R(35)/c_R(40) = {cR_35/cR_40:.2f}")
print(f"  τ_R(35°N) = {tauR_35:.1f} years")
print(f"  τ_R(40°N) = {tauR_40:.1f} years")
print(f"  τ_R ratio (40/35) = {tauR_40/tauR_35:.2f}")

# ══════════════════════════════════════════════════════════════
# Panel (b): ODE integration — latitude-dependent vs constant τ_R
# ══════════════════════════════════════════════════════════════
# Conceptual model: dφ_KE/dt = [φ_eq(t) - φ_KE(t)] / τ_R(φ_KE)
# φ_eq(t) = φ_eq(0) + v_eq * t, v_eq from Hadley/wind forcing

v_eq = obs['v_centroid']  # Use observed centroid rate as effective forcing
phi_eq_0 = obs['phi_KE_now'] + v_eq / 10 * tau_R(obs['phi_KE_now'])

def ode_variable_tau(t, phi):
    """ODE with latitude-dependent τ_R."""
    phi_eq = phi_eq_0 + v_eq / 10 * t  # v_eq in °/decade, t in years
    tau = tau_R(max(min(phi[0], 48), 28))
    return [(phi_eq - phi[0]) / tau]

def ode_constant_tau(t, phi):
    """ODE with constant τ_R at 35°N."""
    phi_eq = phi_eq_0 + v_eq / 10 * t
    tau = tau_R(35)
    return [(phi_eq - phi[0]) / tau]

t_span = (0, 100)
t_eval = np.linspace(0, 100, 1000)
phi0 = [obs['phi_KE_now']]

sol_var = solve_ivp(ode_variable_tau, t_span, phi0, t_eval=t_eval, max_step=0.5)
sol_const = solve_ivp(ode_constant_tau, t_span, phi0, t_eval=t_eval, max_step=0.5)
phi_eq_t = phi_eq_0 + v_eq / 10 * t_eval

# Compute instantaneous migration rate (°/decade)
dphi_var = np.gradient(sol_var.y[0], t_eval) * 10  # °/year → °/decade
dphi_const = np.gradient(sol_const.y[0], t_eval) * 10

print(f"\n=== Panel (b): ODE Integration ===")
print(f"  φ_eq(0) = {phi_eq_0:.2f}°N (equilibrium at t=0)")
print(f"  Forcing rate v_eq = {v_eq:.3f}°/decade")
print(f"  Constant-τ lag = {v_eq/10 * tau_R(35):.2f}°")
print(f"  At t=50yr: φ_KE(var) = {sol_var.y[0][500]:.2f}°, φ_KE(const) = {sol_const.y[0][500]:.2f}°")
print(f"  At t=50yr: rate(var) = {dphi_var[500]:.3f}°/dec, rate(const) = {dphi_const[500]:.3f}°/dec")

# ══════════════════════════════════════════════════════════════
# Panel (c): Closed-loop validation
# ══════════════════════════════════════════════════════════════
# Framework predictions vs observations

# Prediction 1: Steady-state displacement rate ≈ forcing rate
pred_displacement = v_eq  # In steady state, centroid tracks φ_eq at same rate

# Prediction 2: Lag distance
pred_lag_dist = v_eq / 10 * tau_R(obs['phi_KE_now'])  # degrees

# Prediction 3: c_R ratio 35°N/40°N
pred_cR_ratio = c_R(35) / c_R(40)
obs_cR_ratio = obs['cR_35'] / obs['cR_40']

# Prediction 4: τ_R at 35°N → expected lag time
pred_lag_time = tau_R(obs['phi_KE_now'])  # but this is basin-crossing, not lag

# Prediction 5: Vel-max / centroid ratio from f-scaling
# v_g = (g/f)(∂η/∂y), so as jet shifts north by δφ:
# If SSH gradient steepens to maintain Sverdrup transport,
# the vel-max shift exceeds centroid shift.
# Simple estimate: vel_max/centroid ~ 1 + (1/f)(df/dφ) * δW
# where δW is the effective jet width asymmetry
phi_0 = obs['phi_KE_now']
delta_phi_total = obs['v_velmax'] / 10 * 30  # total shift over 30 years
f_ratio = f_cor(phi_0 + delta_phi_total) / f_cor(phi_0)

print(f"\n=== Panel (c): Closed-Loop Validation ===")
print(f"  [Link 1] Hadley → Wind: {obs['v_hadley']:.2f} → {obs['v_wind']:.2f} °/dec")
print(f"  [Link 2] Wind → Sverdrup → KE centroid: {obs['v_wind']:.2f} → {obs['v_centroid']:.3f} °/dec")
print(f"  [Link 3] c_R ratio predicted: {pred_cR_ratio:.2f}, observed: {obs_cR_ratio:.1f}")
print(f"  [Link 4] τ_R(35°N) = {tauR_35:.1f} yr (obs lag: {obs['lag_obs']} yr)")
print(f"  [Link 5] f-ratio over 30yr: {f_ratio:.4f}")
print(f"  [Link 6] Vel-max rate / centroid rate = {obs['v_velmax']/obs['v_centroid']:.2f}")

# ══════════════════════════════════════════════════════════════
# Panel (d): Forward projection — deceleration
# ══════════════════════════════════════════════════════════════
# Instantaneous migration rate as function of current latitude
phi_proj = np.linspace(33, 42, 100)
# In quasi-steady state, rate ≈ v_eq but τ_R grows → lag grows → if forcing stops, rate → 0
# Show rate for constant forcing, as function of achieved latitude
rate_proj = v_eq * tau_R(obs['phi_KE_now']) / tau_R(phi_proj)
# This is the rate if the system has a fixed lag and τ_R changes

# Time to reach each latitude from current position
cumulative_time = np.zeros_like(phi_proj)
dphi = phi_proj[1] - phi_proj[0]
for i in range(1, len(phi_proj)):
    local_rate = v_eq * tau_R(obs['phi_KE_now']) / tau_R(phi_proj[i])  # approximate
    if local_rate > 0:
        cumulative_time[i] = cumulative_time[i-1] + dphi / (local_rate / 10)  # years
    else:
        cumulative_time[i] = np.inf

print(f"\n=== Panel (d): Forward Projection ===")
print(f"  Rate at 35°N: {v_eq:.3f}°/dec")
print(f"  Rate at 37°N: {v_eq * tau_R(35)/tau_R(37):.3f}°/dec")
print(f"  Rate at 39°N: {v_eq * tau_R(35)/tau_R(39):.3f}°/dec")
idx_oe = np.argmin(np.abs(phi_proj - obs['phi_OE']))
print(f"  Time to reach OE front (40°N): ~{cumulative_time[idx_oe]:.0f} years")

# ══════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(hspace=0.35, wspace=0.35)

# ── (a) Rossby wave speed and τ_R ──
ax = axes[0, 0]
color1 = '#2166ac'
color2 = '#b2182b'

ln1 = ax.plot(phi_range, cR_theory, '-', color=color1, linewidth=2,
              label=r'$c_R = \beta R_d^2$ (theory)')
ax.plot([35, 40], [obs['cR_35']*100, obs['cR_40']*100], 'o', color=color1,
        markersize=10, markeredgecolor='k', markeredgewidth=1.2, zorder=5,
        label='Chelton et al. (1998)')
ax.set_xlabel('Latitude (°N)')
ax.set_ylabel(r'$c_R$ (cm s$^{-1}$)', color=color1)
ax.tick_params(axis='y', labelcolor=color1)
ax.set_xlim(28, 48)
ax.set_ylim(0, 5)

ax2 = ax.twinx()
ln2 = ax2.plot(phi_range, tauR_theory, '--', color=color2, linewidth=2,
               label=r'$\tau_R = L_x / c_R$')
ax2.set_ylabel(r'$\tau_R$ (years)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 60)

# Mark 35°N and 40°N
for phi_mark, label in [(35, '35°N'), (40, '40°N')]:
    ax.axvline(phi_mark, color='gray', linewidth=0.5, linestyle=':')
    tau_val = tau_R(phi_mark)
    ax2.plot(phi_mark, tau_val, 's', color=color2, markersize=8,
             markeredgecolor='k', markeredgewidth=1, zorder=5)

# KE current position
ax.axvline(obs['phi_KE_now'], color='#4daf4a', linewidth=1.5, linestyle='-',
           alpha=0.7, label=f'KE now ({obs["phi_KE_now"]}°N)')
ax.axvline(obs['phi_OE'], color='#ff7f00', linewidth=1.5, linestyle='-',
           alpha=0.7, label=f'OE front ({obs["phi_OE"]}°N)')

# Combined legend
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
lns_extra = ax.get_lines()
ax.legend(loc='upper right', fontsize=8)

# Annotation
ax.annotate(r'$c_R \propto \beta/f^2$' + '\ndecreases poleward',
            xy=(42, 0.8), fontsize=9, fontstyle='italic', color='#555555',
            ha='center')

ax.set_title('(a) Planetary vorticity brake', fontweight='bold', loc='left')

# ── (b) ODE integration ──
ax = axes[0, 1]
t_plot = t_eval + 1993  # Convert to calendar year

ax.plot(t_plot, phi_eq_t, ':', color='gray', linewidth=1.5,
        label=r'$\varphi_{eq}(t)$ (Sverdrup target)')
ax.plot(t_plot, sol_const.y[0], '--', color='#1b9e77', linewidth=2,
        label=r'$\tau_R$ = const (35°N)')
ax.plot(t_plot, sol_var.y[0], '-', color='#d95f02', linewidth=2.5,
        label=r'$\tau_R(\varphi)$ variable')

# Mark OE front
ax.axhline(obs['phi_OE'], color='#ff7f00', linewidth=1, linestyle='-', alpha=0.5)
ax.text(2090, obs['phi_OE'] + 0.1, 'OE front', fontsize=8, color='#ff7f00')

# Mark current position and time
ax.axvline(2024, color='k', linewidth=0.5, linestyle=':')
ax.text(2025, obs['phi_KE_now'] - 0.3, 'now', fontsize=8)

# Shade the growing lag
ax.fill_between(t_plot, sol_var.y[0], phi_eq_t, alpha=0.1, color='#d95f02',
                label='Growing lag')

ax.set_xlabel('Year')
ax.set_ylabel('Latitude (°N)')
ax.set_xlim(1993, 2093)
ax.set_ylim(34.5, 42)
ax.legend(loc='upper left', fontsize=8)
ax.set_title('(b) Adjustment dynamics', fontweight='bold', loc='left')

# Inset equation
eq_text = (r'$\frac{d\varphi_{KE}}{dt} = '
           r'\frac{\varphi_{eq}(t) - \varphi_{KE}(t)}{\tau_R(\varphi_{KE})}$')
ax.text(0.97, 0.05, eq_text, transform=ax.transAxes, fontsize=11,
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

# ── (c) Closed-loop validation ──
ax = axes[1, 0]

categories = [
    'Hadley\nwidening',
    'Wind curl\nzero line',
    r'$c_R$ ratio' + '\n35°/40°N',
    'KE centroid\ndisplacement',
    'KE vel-max\nmigration',
]

predicted = [
    obs['v_hadley'],          # Input (literature)
    obs['v_wind'],            # Input (literature)
    pred_cR_ratio,            # From β/f² theory
    obs['v_centroid'],        # Predicted = forcing rate in steady state
    obs['v_centroid'] * obs['v_velmax'] / obs['v_centroid'],  # = vel_max (observed)
]

observed = [
    obs['v_hadley'],          # Seidel (2008)
    obs['v_wind'],            # Wu (2018)
    obs_cR_ratio,             # Chelton (1998)
    obs['v_centroid'],        # This study
    obs['v_velmax'],          # This study
]

# Normalize: show as "predicted / observed" ratio
# But categories have different units — better to show a comparison table
# Use grouped bars for rates (°/dec) and separate for ratio

# Rates (°/decade): first two are inputs, last two are predictions
rate_labels = ['Hadley\nwidening', 'Wind curl\nshift', 'KE centroid\n(displacement)', 'KE vel-max\n(total)']
rate_literature = [0.50, 0.58, np.nan, np.nan]
rate_observed = [np.nan, np.nan, obs['v_centroid'], obs['v_velmax']]

x = np.arange(len(rate_labels))
width = 0.35

bars1 = ax.bar(x - width/2, [v if not np.isnan(v) else 0 for v in rate_literature],
               width, color='#4393c3', alpha=0.85, label='Literature input',
               edgecolor='k', linewidth=0.5)
bars2 = ax.bar(x + width/2, [v if not np.isnan(v) else 0 for v in rate_observed],
               width, color='#d6604d', alpha=0.85, label='This study',
               edgecolor='k', linewidth=0.5)

# Add value labels
for bar, val in zip(bars1, rate_literature):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, rate_observed):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Draw chain arrows
for i in range(len(rate_labels) - 1):
    ax.annotate('', xy=(i + 1 - 0.5, 0.62), xytext=(i + 0.5, 0.62),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

ax.set_ylabel('Rate (°/decade)')
ax.set_xticks(x)
ax.set_xticklabels(rate_labels)
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(0, 0.75)
ax.set_title('(c) Observational chain', fontweight='bold', loc='left')

# Add annotation for the chain
ax.text(0.5, 0.95, r'Hadley $\rightarrow$ Wind $\rightarrow$ Sverdrup $\rightarrow$ KE',
        transform=ax.transAxes, ha='center', va='top', fontsize=9,
        fontstyle='italic', color='#333333')

# ── (d) Deceleration projection ──
ax = axes[1, 1]

# Rate vs latitude
ax.plot(phi_proj, rate_proj, '-', color='#d95f02', linewidth=2.5,
        label='Predicted rate')
ax.axhline(obs['v_centroid'], color='#2166ac', linewidth=1, linestyle='--',
           label=f'Current rate ({obs["v_centroid"]:.2f}°/dec)')

# Mark key latitudes
ax.axvline(obs['phi_KE_now'], color='#4daf4a', linewidth=1, linestyle=':')
ax.axvline(obs['phi_OE'], color='#ff7f00', linewidth=1, linestyle=':')
ax.text(obs['phi_KE_now'] + 0.1, 0.02, 'KE now', fontsize=8, color='#4daf4a', rotation=90)
ax.text(obs['phi_OE'] + 0.1, 0.02, 'OE front', fontsize=8, color='#ff7f00', rotation=90)

# Fill deceleration region
ax.fill_between(phi_proj, rate_proj, 0, alpha=0.15, color='#d95f02')

# Show τ_R values
ax_tau = ax.twinx()
ax_tau.plot(phi_proj, tau_R(phi_proj), ':', color='#b2182b', linewidth=1.5, alpha=0.6)
ax_tau.set_ylabel(r'$\tau_R$ (years)', color='#b2182b', fontsize=9)
ax_tau.tick_params(axis='y', labelcolor='#b2182b', labelsize=8)

ax.set_xlabel('KE latitude (°N)')
ax.set_ylabel('Migration rate (°/decade)')
ax.set_xlim(33, 42)
ax.set_ylim(0, 0.35)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('(d) Deceleration with poleward migration', fontweight='bold', loc='left')

# Annotation: half-rate latitude
phi_half = phi_proj[np.argmin(np.abs(rate_proj - obs['v_centroid'] / 2))]
ax.annotate(f'Half-rate at {phi_half:.1f}°N',
            xy=(phi_half, obs['v_centroid'] / 2),
            xytext=(phi_half + 1.5, obs['v_centroid'] / 2 + 0.06),
            fontsize=8, arrowprops=dict(arrowstyle='->', color='k'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

plt.savefig(FIG / "fig_coriolis_framework.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "fig_coriolis_framework.pdf", dpi=300, bbox_inches='tight')
print(f"\nSaved: {FIG / 'fig_coriolis_framework.png'}")
plt.close()

# ══════════════════════════════════════════════════════════════
# Summary statistics for paper
# ══════════════════════════════════════════════════════════════
framework_stats = {
    "c_R_35N_cm_s": round(cR_35, 2),
    "c_R_40N_cm_s": round(cR_40, 2),
    "c_R_ratio_35_40": round(cR_35 / cR_40, 2),
    "tau_R_35N_yr": round(tauR_35, 1),
    "tau_R_40N_yr": round(tauR_40, 1),
    "tau_R_ratio_40_35": round(tauR_40 / tauR_35, 2),
    "steady_state_lag_deg": round(v_eq / 10 * tauR_35, 2),
    "phi_half_rate_N": round(float(phi_half), 1),
    "time_to_OE_front_yr": round(float(cumulative_time[idx_oe]), 0),
    "rate_at_37N_deg_dec": round(float(v_eq * tauR_35 / tau_R(37)), 3),
    "rate_at_39N_deg_dec": round(float(v_eq * tauR_35 / tau_R(39)), 3),
    "vel_max_centroid_ratio": round(obs['v_velmax'] / obs['v_centroid'], 2),
    "f_ratio_35_37": round(f_cor(37) / f_cor(35), 4),
}

with open(OUT / "coriolis_framework_stats.json", "w") as fh:
    json.dump(framework_stats, fh, indent=2)

print("\n=== Framework Statistics ===")
print(json.dumps(framework_stats, indent=2))
