"""Sverdrup transport calculation: predict KE position from wind stress curl
Integrate wind stress curl from eastern boundary to 142°E
Track Sverdrup stream function maximum latitude → predicted KE position
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from pathlib import Path
import pandas as pd
import json

OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")

# ── 1. Load ERA5 and compute wind stress curl ──
ds = xr.open_dataset("/Users/zhulin/aitest/黑潮延伸体/data/era5_monthly_wind_npac_1993_2025.nc")
u10, v10 = ds['u10'], ds['v10']
lat = ds.latitude.values
lon = ds.longitude.values

rho_a, Cd = 1.225, 1.3e-3
wspd = np.sqrt(u10**2 + v10**2)
tau_x = rho_a * Cd * wspd * u10
tau_y = rho_a * Cd * wspd * v10

R = 6.371e6
dlat = np.abs(np.diff(lat).mean()) * np.pi / 180
dlon = np.abs(np.diff(lon).mean()) * np.pi / 180
cos_lat = np.cos(np.deg2rad(ds.latitude))

curl = (tau_y.differentiate('longitude') / (R * cos_lat * dlon * (180/np.pi))
        - tau_x.differentiate('latitude') / (R * dlat * (180/np.pi)))

# Annual mean
curl_ann = curl.resample(valid_time='YE').mean()
time_ann = curl_ann.valid_time.values

# ── 2. Sverdrup stream function ──
# ψ_sv(x,y) = (1/β) ∫_{x_E}^{x} curl(τ) dx'
# β = df/dy = 2Ω cos(φ) / R
# Integration from eastern boundary (240°E = 120°W) westward to each longitude

beta = 2 * 7.292e-5 * np.cos(np.deg2rad(lat)) / R  # (nlat,)
rho_0 = 1025.0  # kg/m³

# Grid spacing in meters at each latitude
dx = R * np.cos(np.deg2rad(lat))[:, None] * dlon  # (nlat, 1)

# Eastern boundary index (240°E)
i_east = np.argmin(np.abs(lon - 240))
# Target longitude (142°E)
i_142 = np.argmin(np.abs(lon - 142))

print(f"Eastern boundary: {lon[i_east]:.1f}°E, Target: {lon[i_142]:.1f}°E")
print(f"Integration range: {lon[i_east]:.1f} → {lon[i_142]:.1f}°E (westward)")

# Compute Sverdrup stream function at 142°E for each year
sv_lat_max = np.full(len(time_ann), np.nan)

for t in range(len(time_ann)):
    curl_t = curl_ann.values[t]  # (nlat, nlon)
    # Integrate from east to 142°E (sum curl * dx from i_east to i_142, going west)
    # Note: lon increases eastward, so integrating westward = negative direction
    # ψ = (1/ρβ) ∫ curl dA, integrated from east boundary
    if i_142 < i_east:
        psi = np.nansum(curl_t[:, i_142:i_east+1] * dx, axis=1) / (rho_0 * beta)
    else:
        psi = np.nansum(curl_t[:, i_east:i_142+1] * dx, axis=1) / (rho_0 * beta)

    # Find latitude of maximum (southward) transport = maximum |ψ|
    # In subtropical gyre, ψ < 0 (anticyclonic), max |ψ| = min ψ
    # KE is at the poleward edge where ψ → 0 from negative
    # Find zero crossing (ψ changes sign from negative to positive going poleward)
    mask = (lat >= 25) & (lat <= 50)
    psi_sub = psi.copy()
    psi_sub[~mask] = np.nan

    # Find where psi crosses zero (poleward edge of subtropical gyre)
    for i in range(len(lat)-1):
        if mask[i] and mask[i+1] and not np.isnan(psi_sub[i]) and not np.isnan(psi_sub[i+1]):
            if psi_sub[i] < 0 and psi_sub[i+1] >= 0 and lat[i] > 30:
                frac = -psi_sub[i] / (psi_sub[i+1] - psi_sub[i])
                sv_lat_max[t] = lat[i] + frac * (lat[i+1] - lat[i])
                break

valid_sv = ~np.isnan(sv_lat_max)
print(f"\n有效年数: {valid_sv.sum()}/{len(sv_lat_max)}")
print(f"Sverdrup 零线纬度范围: {np.nanmin(sv_lat_max):.2f} ~ {np.nanmax(sv_lat_max):.2f}°N")

ty = np.arange(len(time_ann), dtype=float)
sl_sv, ic_sv, r_sv, p_sv, _ = linregress(ty[valid_sv], sv_lat_max[valid_sv])
print(f"Sverdrup 边界趋势: {sl_sv*10:.4f}°/decade, p={p_sv:.5f}")

# ── 3. Compare with observed KE axis ──
# Load velocity centroid annual
vel_centroid = json.load(open(OUT / "three_method_comparison.json"))
print(f"\n流速重心趋势: {vel_centroid['Velocity centroid']['trend_per_decade']}°/dec")
print(f"Sverdrup 预测: {sl_sv*10:.4f}°/dec")
print(f"比值: {sl_sv*10 / vel_centroid['Velocity centroid']['trend_per_decade']:.2f}")

# ── 4. Plot ──
fig, ax = plt.subplots(figsize=(12, 5))
dates = pd.to_datetime(time_ann)

# Sverdrup
sv_norm = (sv_lat_max - np.nanmean(sv_lat_max)) / np.nanstd(sv_lat_max)
ax.plot(dates, sv_norm, 'g-o', markersize=4, linewidth=1.5, label=f'Sverdrup gyre boundary ({sl_sv*10:+.2f}°/dec)')

# KE axis (from monthly, annual mean)
ke_ds = xr.open_dataset(OUT / "ke_axis_position.nc")
ke_ann = ke_ds['ke_axis_latitude'].resample(time='YE').mean()
ke_norm = (ke_ann.values - np.nanmean(ke_ann.values)) / np.nanstd(ke_ann.values)
ke_dates = pd.to_datetime(ke_ann.time.values)
ax.plot(ke_dates, ke_norm, 'b-s', markersize=4, linewidth=1.5, label='KE axis SLA gradient (normalized)')

# Trend lines
ax.plot(dates, (sl_sv*ty + ic_sv - np.nanmean(sv_lat_max))/np.nanstd(sv_lat_max),
        'g--', linewidth=1, alpha=0.5)

ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('Standardized Anomaly')
ax.set_xlabel('Year')
ax.set_title('Sverdrup-Predicted Gyre Boundary vs Observed KE Axis')

textstr = (f'Sverdrup boundary trend: {sl_sv*10:+.3f}°/dec (p={p_sv:.3f})\n'
           f'Observed centroid:       +0.142°/dec\n'
           f'Observed vel-max:        +0.508°/dec')
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace')

ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(FIG / "fig_sverdrup.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "fig_sverdrup.pdf", dpi=300, bbox_inches='tight')
print(f"\n保存: {FIG / 'fig_sverdrup.png'}")
plt.close()

stats = {
    "sverdrup_boundary_trend_deg_per_decade": round(sl_sv * 10, 5),
    "sverdrup_boundary_p_value": round(p_sv, 5),
    "sverdrup_mean_lat": round(float(np.nanmean(sv_lat_max)), 2),
}
with open(OUT / "sverdrup_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
