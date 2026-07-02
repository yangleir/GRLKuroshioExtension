"""R11 mechanism figures: causal chain + SLA bias profile + Pacific vs Atlantic wind curl
Fig A: Rate consistency chain (Hadley → zero line → KE)
Fig B: SLA bias mechanism — meridional profiles early vs late
Fig C: Pacific vs Atlantic wind stress curl trends
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import pandas as pd
import json

OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")
ROOT_DAILY = Path("/Volumes/Backup Plus/ssh/dataset-duacs-rep-global-merged-allsat-phy-l4")

# ═══════════════════════════════════════════
# Fig A: Causal chain — zero line + KE velocity + KE SLA gradient
# ═══════════════════════════════════════════
print("=== Fig A: 因果链时间序列 ===")

# Load KE velocity axis (annual)
vel_stats = json.load(open(OUT / "velocity_analysis_stats.json"))

# Load wind curl zero line (annual)
wind_stats = json.load(open(OUT / "wind_curl_stats.json"))

# Load KE axis from monthly gradient
ke_ds = xr.open_dataset(OUT / "ke_axis_position.nc")
ke_monthly = ke_ds['ke_axis_latitude']
ke_annual = ke_monthly.resample(time='YE').mean()

# Load zero line from P4
# Recompute zero line annual from ERA5
era5 = xr.open_dataset("/Users/zhulin/aitest/黑潮延伸体/data/era5_monthly_wind_npac_1993_2025.nc")
u10 = era5['u10']
v10 = era5['v10']
lat_e = era5.latitude.values
lon_e = era5.longitude.values

rho_a, Cd = 1.225, 1.3e-3
wspd = np.sqrt(u10**2 + v10**2)
tau_x = rho_a * Cd * wspd * u10
tau_y = rho_a * Cd * wspd * v10

R = 6.371e6
dlat = np.abs(np.diff(lat_e).mean()) * np.pi / 180
dlon = np.abs(np.diff(lon_e).mean()) * np.pi / 180
cos_lat = np.cos(np.deg2rad(era5.latitude))
dtau_y_dx = tau_y.differentiate('longitude') / (R * cos_lat * dlon * (180/np.pi))
dtau_x_dy = tau_x.differentiate('latitude') / (R * dlat * (180/np.pi))
curl_tau = dtau_y_dx - dtau_x_dy

curl_annual = curl_tau.resample(valid_time='YE').mean()

# Zero line tracking (130-180E)
lon_mask = (lon_e >= 130) & (lon_e <= 180)
curl_region = curl_annual.sel(longitude=lon_e[lon_mask])
time_annual = curl_annual.valid_time.values
zero_lat = np.full(len(time_annual), np.nan)

for t in range(len(time_annual)):
    lats_zero = []
    for j in range(curl_region.shape[2]):
        col = curl_region.values[t, :, j]
        for i in range(len(lat_e) - 1):
            if not np.isnan(col[i]) and not np.isnan(col[i+1]):
                if col[i] > 0 and col[i+1] <= 0 and lat_e[i] > 20 and lat_e[i] < 50:
                    frac = col[i] / (col[i] - col[i+1])
                    lats_zero.append(lat_e[i] + frac * (lat_e[i+1] - lat_e[i]))
                    break
    if len(lats_zero) >= 5:
        zero_lat[t] = np.median(lats_zero)

# Load velocity axis annual
vel_ds = xr.open_dataset(OUT / "ke_axis_position.nc")  # This has SLA gradient
# We need velocity axis — read from velocity stats or recompute
# Use the annual means from fig6 data
# For now, load from the velocity script output if available

# Standardize all series for overlay
years_e = pd.to_datetime(time_annual)
years_k = pd.to_datetime(ke_annual.time.values)

# Zero line
zl_valid = ~np.isnan(zero_lat)
zl_norm = (zero_lat - np.nanmean(zero_lat)) / np.nanstd(zero_lat)

# KE SLA gradient axis
ke_vals = ke_annual.values
ke_valid = ~np.isnan(ke_vals)
ke_norm = (ke_vals - np.nanmean(ke_vals)) / np.nanstd(ke_vals)

# Trends
ty_e = np.arange(len(time_annual), dtype=float)
ty_k = np.arange(len(ke_annual.time), dtype=float)

fig, ax = plt.subplots(figsize=(14, 5))

# Zero line
ax.plot(years_e, zl_norm, 'r-o', markersize=3, linewidth=1.5, label='Wind curl zero line (normalized)', alpha=0.8)
sl_z, ic_z, _, p_z, _ = linregress(ty_e[zl_valid], zero_lat[zl_valid])
ax.plot(years_e, (sl_z * ty_e + ic_z - np.nanmean(zero_lat)) / np.nanstd(zero_lat),
        'r--', linewidth=1, alpha=0.5)

# KE SLA gradient axis
ax.plot(years_k, ke_norm, 'b-o', markersize=3, linewidth=1.5, label='KE axis SLA gradient (normalized)', alpha=0.8)

# Annotations
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('Standardized Anomaly')
ax.set_xlabel('Year')

# Rate box
textstr = ('Rate consistency:\n'
           'Hadley widening    ~0.50°/dec\n'
           f'Wind curl zero line  {sl_z*10:+.2f}°/dec\n'
           'KE velocity max     +0.50°/dec\n'
           'KE SLA gradient     -0.02°/dec')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props, family='monospace')

ax.set_title('Causal Chain: Wind Stress Curl Zero Line vs KE Jet Axis Position')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(FIG / "figA_causal_chain.png", dpi=300, bbox_inches='tight')
print(f"保存: {FIG / 'figA_causal_chain.png'}")
plt.close()

# ═══════════════════════════════════════════
# Fig B: SLA bias mechanism — meridional profiles
# ═══════════════════════════════════════════
print("\n=== Fig B: SLA 偏差机制经向剖面 ===")

# Read daily data: sample early (1995) and late (2019) years
def read_ke_profile(year, month=6):
    day_dir = ROOT_DAILY / str(year) / f"{month:02d}"
    nc_files = sorted(day_dir.glob("*.nc"))
    if not nc_files:
        return None, None, None
    profiles = {'adt': [], 'sla': []}
    for fp in nc_files[::5]:  # sample every 5 days
        ds = xr.open_dataset(fp)
        lon = ds.longitude.values
        if lon.max() > 180:
            ke = ds.sel(latitude=slice(28, 44), longitude=slice(142, 170))
        else:
            ke = ds.sel(latitude=slice(28, 44), longitude=slice(142, 170))
        if len(ke.longitude) == 0:
            ds.close()
            continue
        profiles['adt'].append(ke['adt'].isel(time=0).mean(dim='longitude').values)
        profiles['sla'].append(ke['sla'].isel(time=0).mean(dim='longitude').values)
        lat_p = ke.latitude.values
        ds.close()
    if not profiles['adt']:
        return None, None, None
    adt_mean = np.nanmean(profiles['adt'], axis=0)
    sla_mean = np.nanmean(profiles['sla'], axis=0)
    return lat_p, adt_mean, sla_mean

# Early period (average of 1994-1996)
print("  读取早期数据...")
adt_early_list, sla_early_list = [], []
for yr in [1994, 1995, 1996]:
    for mo in [3, 6, 9, 12]:
        lat_p, adt_m, sla_m = read_ke_profile(yr, mo)
        if adt_m is not None:
            adt_early_list.append(adt_m)
            sla_early_list.append(sla_m)

# Late period (average of 2019-2021)
print("  读取晚期数据...")
adt_late_list, sla_late_list = [], []
for yr in [2019, 2020, 2021]:
    for mo in [3, 6, 9, 12]:
        lat_p, adt_m, sla_m = read_ke_profile(yr, mo)
        if adt_m is not None:
            adt_late_list.append(adt_m)
            sla_late_list.append(sla_m)

adt_early = np.nanmean(adt_early_list, axis=0)
adt_late = np.nanmean(adt_late_list, axis=0)
sla_early = np.nanmean(sla_early_list, axis=0)
sla_late = np.nanmean(sla_late_list, axis=0)
mssh = np.nanmean(adt_early_list + adt_late_list, axis=0) - np.nanmean(sla_early_list + sla_late_list, axis=0)

dlat_p = np.abs(np.diff(lat_p).mean())

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

# SSH (ADT) profiles
ax = axes[0]
ax.plot(adt_early, lat_p, 'b-', linewidth=2, label='1994-1996')
ax.plot(adt_late, lat_p, 'r-', linewidth=2, label='2019-2021')
# Mark gradient max
grad_e = np.gradient(gaussian_filter1d(adt_early, 2), dlat_p)
grad_l = np.gradient(gaussian_filter1d(adt_late, 2), dlat_p)
mask = (lat_p >= 32) & (lat_p <= 40)
idx_e = np.argmax(np.where(mask, grad_e, -np.inf))
idx_l = np.argmax(np.where(mask, grad_l, -np.inf))
ax.axhline(lat_p[idx_e], color='b', linestyle='--', alpha=0.5)
ax.axhline(lat_p[idx_l], color='r', linestyle='--', alpha=0.5)
ax.annotate('', xy=(adt_early[idx_l]*0.95, lat_p[idx_l]), xytext=(adt_early[idx_e]*0.95, lat_p[idx_e]),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.set_xlabel('ADT (m)')
ax.set_ylabel('Latitude (°N)')
ax.set_title('(a) SSH/ADT: jet shifts north')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# MSSH profile (static)
ax = axes[1]
ax.plot(mssh, lat_p, 'k-', linewidth=2, label='MSSH (fixed reference)')
grad_m = np.gradient(gaussian_filter1d(mssh, 2), dlat_p)
idx_m = np.argmax(np.where(mask, grad_m, -np.inf))
ax.axhline(lat_p[idx_m], color='k', linestyle='--', alpha=0.5, label=f'MSSH grad max: {lat_p[idx_m]:.1f}°N')
ax.set_xlabel('MSSH (m)')
ax.set_title('(b) MSSH: anchored to historical mean')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# SLA profiles
ax = axes[2]
ax.plot(sla_early, lat_p, 'b-', linewidth=2, label='1994-1996')
ax.plot(sla_late, lat_p, 'r-', linewidth=2, label='2019-2021')
grad_se = np.gradient(gaussian_filter1d(sla_early, 2), dlat_p)
grad_sl = np.gradient(gaussian_filter1d(sla_late, 2), dlat_p)
idx_se = np.argmax(np.where(mask, grad_se, -np.inf))
idx_sl = np.argmax(np.where(mask, grad_sl, -np.inf))
ax.axhline(lat_p[idx_se], color='b', linestyle='--', alpha=0.5)
ax.axhline(lat_p[idx_sl], color='r', linestyle='--', alpha=0.5)
if lat_p[idx_sl] < lat_p[idx_se]:
    ax.annotate('', xy=(sla_early[idx_sl]*0.95, lat_p[idx_sl]), xytext=(sla_early[idx_se]*0.95, lat_p[idx_se]),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax.text(sla_early.min()*0.5, (lat_p[idx_se]+lat_p[idx_sl])/2, 'Apparent\nsouthward!',
            color='orange', fontsize=10, fontweight='bold', ha='center')
ax.set_xlabel('SLA (m)')
ax.set_title('(c) SLA = SSH - MSSH: bias reverses trend')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('Self-Concealing Mechanism: SSH Shifts North, SLA Gradient Shifts South', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG / "figB_sla_bias_profiles.png", dpi=300, bbox_inches='tight')
print(f"保存: {FIG / 'figB_sla_bias_profiles.png'}")
plt.close()

# ═══════════════════════════════════════════
# Fig C: Pacific vs Atlantic wind stress curl trends
# ═══════════════════════════════════════════
print("\n=== Fig C: 太平洋 vs 大西洋风应力旋度趋势 ===")

import cartopy.crs as ccrs
import cartopy.feature as cfeature

curl_monthly = curl_tau
curl_annual_full = curl_monthly.resample(valid_time='YE').mean()
curl_vals = curl_annual_full.values
years_arr = np.arange(curl_vals.shape[0], dtype=float)

trend_curl_full = np.full((len(lat_e), len(lon_e)), np.nan)
for i in range(len(lat_e)):
    for j in range(len(lon_e)):
        y = curl_vals[:, i, j]
        valid = ~np.isnan(y)
        if valid.sum() > 10:
            sl, _, _, _, _ = linregress(years_arr[valid], y[valid])
            trend_curl_full[i, j] = sl * 10

fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                          subplot_kw={'projection': ccrs.PlateCarree()})

clevels = np.linspace(-3e-9, 3e-9, 25)
curl_clim = curl_monthly.mean(dim='valid_time')

# Pacific
ax = axes[0]
lon2d, lat2d = np.meshgrid(lon_e, lat_e)
cs = ax.contourf(lon2d, lat2d, trend_curl_full, levels=clevels, cmap='RdBu_r', extend='both')
ax.contour(lon2d, lat2d, curl_clim.values, levels=[0], colors='black', linewidths=2)
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.LAND, color='lightgray')
ax.set_extent([120, 240, 10, 60], crs=ccrs.PlateCarree())
ax.set_title('(a) North Pacific Wind Stress Curl Trend')

# Atlantic — use same ERA5 data but different lon range
# ERA5 covers 120-240E, need to check if Atlantic is included
# Atlantic would be ~280-360E = -80 to 0 in -180~180 = 280-360 in 0-360
# Our ERA5 covers 120-240E, so Atlantic is NOT covered.
# Need to note this limitation

ax = axes[1]
ax.text(0.5, 0.5, 'Atlantic wind data\nnot in current ERA5 download\n(120-240°E only)\n\nNeed separate download\nfor 80°W-0°',
        transform=ax.transAxes, ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.LAND, color='lightgray')
ax.set_extent([-80, 0, 10, 60], crs=ccrs.PlateCarree())
ax.set_title('(b) North Atlantic Wind Stress Curl Trend')

plt.colorbar(cs, ax=axes, orientation='horizontal', shrink=0.5, pad=0.08,
             label='Wind stress curl trend (N/m³ per decade)')
plt.tight_layout()
plt.savefig(FIG / "figC_pacific_atlantic_wind.png", dpi=300, bbox_inches='tight')
print(f"保存: {FIG / 'figC_pacific_atlantic_wind.png'}")
print("注意: 大西洋风场需要单独下载 ERA5 数据")
plt.close()

print("\n=== 完成 ===")
