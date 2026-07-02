"""P7: Two analyses to strengthen the paper
1. Velocity centroid method — separate displacement from intensification
2. Simple Sverdrup calculation — predict KE latitude from wind stress curl
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
ROOT = Path("/Volumes/Backup Plus/ssh/dataset-duacs-rep-global-merged-allsat-phy-l4")

# ═══════════════════════════════════════════
# PART 1: Velocity centroid — separate displacement vs intensification
# ═══════════════════════════════════════════

print("=" * 60)
print("PART 1: Velocity centroid analysis")
print("=" * 60)

# Load pre-computed velocity data (reuse P5 file reading)
files = []
for year in range(1993, 2022):
    for month in range(1, 13):
        day_dir = ROOT / str(year) / f"{month:02d}"
        if not day_dir.exists():
            continue
        target = f"dt_global_allsat_phy_l4_{year}{month:02d}15"
        candidates = sorted(day_dir.glob(f"{target}*.nc"))
        if candidates:
            files.append(candidates[0])
        else:
            all_nc = sorted(day_dir.glob("*.nc"))
            if len(all_nc) >= 15:
                files.append(all_nc[14])
            elif all_nc:
                files.append(all_nc[len(all_nc) // 2])

lon_min, lon_max = 142, 170
lat_min, lat_max = 30, 42

ugos_list = []
times = []

for i, fp in enumerate(files):
    ds = xr.open_dataset(fp)
    lon = ds.longitude.values
    if lon.min() < 0:
        ds = ds.assign_coords(longitude=(ds.longitude % 360))
        ds = ds.sortby('longitude')
    ke = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    if len(ke.longitude) == 0:
        ds.close()
        continue
    ugos_list.append(ke['ugos'].isel(time=0).values)
    times.append(ke.time.values[0])
    ds.close()
    if (i + 1) % 100 == 0:
        print(f"  Reading {i+1}/{len(files)}")

ugos_arr = np.array(ugos_list)
time_arr = np.array(times)
lat = ke.latitude.values
lon_ke = ke.longitude.values
dlat = np.abs(np.diff(lat).mean())
time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25
                        for t in time_arr])

print(f"Data: {ugos_arr.shape}, {str(time_arr[0])[:10]} → {str(time_arr[-1])[:10]}")

# Three tracking methods on the same velocity data
ke_vmax = np.full(len(time_arr), np.nan)     # velocity maximum
ke_centroid = np.full(len(time_arr), np.nan)  # velocity-weighted centroid
ke_halfmax = np.full(len(time_arr), np.nan)   # half-maximum midpoint

for t in range(len(time_arr)):
    frame = ugos_arr[t]  # (lat, lon)
    vmax_lats = []
    centroid_lats = []
    halfmax_lats = []

    for j in range(frame.shape[1]):
        col = frame[:, j]
        if np.isnan(col).sum() > len(col) * 0.3:
            continue
        valid_mask = ~np.isnan(col)
        if valid_mask.sum() < 5:
            continue
        col_interp = np.interp(np.arange(len(col)),
                               np.where(valid_mask)[0], col[valid_mask])
        col_s = gaussian_filter1d(col_interp, sigma=2)

        # Restrict to 32-40°N
        mask = (lat >= 32) & (lat <= 40)
        lat_sub = lat[mask]
        col_sub = col_s[mask]

        # Method 1: velocity maximum (parabolic)
        idx = np.argmax(col_sub)
        if col_sub[idx] > 0.05:
            if 0 < idx < len(col_sub) - 1:
                y0, y1, y2 = col_sub[idx-1], col_sub[idx], col_sub[idx+1]
                denom = 2 * (2 * y1 - y0 - y2)
                if abs(denom) > 1e-10:
                    offset = (y0 - y2) / denom
                    vmax_lats.append(lat_sub[idx] + offset * dlat)
                else:
                    vmax_lats.append(lat_sub[idx])
            else:
                vmax_lats.append(lat_sub[idx])

            # Method 2: velocity-weighted centroid (only positive ugos)
            pos = np.maximum(col_sub, 0)
            if pos.sum() > 0:
                centroid = np.sum(pos * lat_sub) / np.sum(pos)
                centroid_lats.append(centroid)

            # Method 3: half-maximum midpoint
            half_max = col_sub[idx] / 2
            above = col_sub >= half_max
            if above.any():
                above_lats = lat_sub[above]
                halfmax_lats.append((above_lats[0] + above_lats[-1]) / 2)

    if len(vmax_lats) >= frame.shape[1] * 0.3:
        ke_vmax[t] = np.median(vmax_lats)
    if len(centroid_lats) >= frame.shape[1] * 0.3:
        ke_centroid[t] = np.median(centroid_lats)
    if len(halfmax_lats) >= frame.shape[1] * 0.3:
        ke_halfmax[t] = np.median(halfmax_lats)

# Trend computation
print("\n=== Velocity tracking method comparison ===")
centroid_results = {}
for name, vals in [('Velocity max', ke_vmax),
                   ('Velocity centroid', ke_centroid),
                   ('Half-max midpoint', ke_halfmax)]:
    valid = ~np.isnan(vals)
    if valid.sum() > 30:
        sl, ic, r, p, se = linregress(time_years[valid], vals[valid])
        print(f"  {name}: {sl*10:+.4f}°/dec, p={p:.5f}, mean={np.nanmean(vals):.2f}°N")
        centroid_results[name] = {
            'trend_deg_per_decade': round(sl * 10, 5),
            'p_value': round(p, 5),
            'mean_lat': round(float(np.nanmean(vals)), 3),
        }

# Peak velocity trend (intensification check)
peak_vel = np.full(len(time_arr), np.nan)
for t in range(len(time_arr)):
    frame = ugos_arr[t]
    mask = (lat >= 32) & (lat <= 40)
    col_mean = np.nanmean(frame[mask, :], axis=1)
    if not np.all(np.isnan(col_mean)):
        col_s = gaussian_filter1d(np.nan_to_num(col_mean, nan=0), sigma=2)
        peak_vel[t] = np.max(col_s)

valid_pv = ~np.isnan(peak_vel)
sl_pv, _, _, p_pv, _ = linregress(time_years[valid_pv], peak_vel[valid_pv])
print(f"  Peak velocity trend: {sl_pv*10:+.4f} m/s/dec, p={p_pv:.5f}")
centroid_results['Peak velocity (m/s)'] = {
    'trend_per_decade': round(sl_pv * 10, 5),
    'p_value': round(p_pv, 5),
}

# ═══════════════════════════════════════════
# PART 2: Sverdrup calculation
# ═══════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: Sverdrup prediction of KE latitude")
print("=" * 60)

era5 = xr.open_dataset("/Users/zhulin/aitest/黑潮延伸体/data/era5_monthly_wind_npac_1993_2025.nc")
u10 = era5['u10']
v10 = era5['v10']
lat_era = era5.latitude.values
lon_era = era5.longitude.values

# Wind stress
rho_a = 1.225
Cd = 1.3e-3
wspd = np.sqrt(u10**2 + v10**2)
tau_x = rho_a * Cd * wspd * u10
tau_y = rho_a * Cd * wspd * v10

# Wind stress curl
R = 6.371e6
dlon_rad = np.abs(np.diff(lon_era).mean()) * np.pi / 180
dlat_rad = np.abs(np.diff(lat_era).mean()) * np.pi / 180
cos_lat = np.cos(np.deg2rad(era5.latitude))

dtau_y_dx = tau_y.differentiate('longitude') / (R * cos_lat * dlon_rad * (180/np.pi))
dtau_x_dy = tau_x.differentiate('latitude') / (R * dlat_rad * (180/np.pi))
curl_tau = dtau_y_dx - dtau_x_dy

# Annual mean curl
curl_annual = curl_tau.resample(valid_time='YE').mean()
time_annual = curl_annual.valid_time.values
years_era = np.array([pd.Timestamp(t).year for t in time_annual])

# Sverdrup transport at 142°E (KE separation longitude)
# ψ_S(y) = (1/ρ₀β) ∫[x_e to x] curl(τ) dx
# KE position ≈ where ψ_S is maximum (gyre boundary)
# Simplified: track latitude of zero Sverdrup streamfunction at 142°E

# Integrate curl from eastern boundary (120°W = 240°E) to 142°E
beta = 2 * 7.292e-5 * np.cos(np.deg2rad(lat_era)) / R
rho_0 = 1025.0

# Select longitude range for integration
if lon_era.min() < 0:
    lon_int = lon_era[(lon_era >= -120) & (lon_era <= 142)]
else:
    lon_int_mask = ((lon_era >= 142) & (lon_era <= 240))
    lon_int = lon_era[lon_int_mask]

print(f"Sverdrup integration: {len(lon_int)} longitudes, {lon_int[0]:.1f} to {lon_int[-1]:.1f}")

# For each year, compute Sverdrup streamfunction at 142°E
sv_lat_max = np.full(len(time_annual), np.nan)

for t in range(len(time_annual)):
    curl_t = curl_annual.isel(valid_time=t)
    # Select integration range
    if lon_era.min() < 0:
        curl_strip = curl_t.sel(longitude=slice(142, -120)).values  # lat x lon
    else:
        curl_strip = curl_t.sel(longitude=slice(142, 240)).values

    if curl_strip.ndim != 2:
        continue

    # Integrate eastward: Sverdrup transport
    dx = R * np.cos(np.deg2rad(lat_era))[:, np.newaxis] * dlon_rad * (180/np.pi)
    # ψ = -(1/β) ∫ curl dx (integrated from east to west, so sum from right)
    psi = np.cumsum(curl_strip[:, ::-1], axis=1)[:, ::-1] * dx[:curl_strip.shape[0], :curl_strip.shape[1]]

    # Sverdrup streamfunction at western boundary (first longitude = 142°E)
    psi_west = psi[:, 0]

    # Find latitude of maximum streamfunction (gyre center) in 25-45°N
    lat_mask = (lat_era >= 25) & (lat_era <= 45)
    psi_sub = psi_west.copy()
    psi_sub[~lat_mask] = np.nan

    if not np.all(np.isnan(psi_sub)):
        # For KE latitude: find where psi changes sign (gyre boundary)
        # or where dpsi/dy = 0 (maximum transport)
        valid_psi = ~np.isnan(psi_sub)
        psi_valid = psi_sub[valid_psi]
        lat_valid = lat_era[valid_psi]

        idx_max = np.argmax(np.abs(psi_valid))
        if 0 < idx_max < len(psi_valid) - 1:
            sv_lat_max[t] = lat_valid[idx_max]

valid_sv = ~np.isnan(sv_lat_max)
if valid_sv.sum() > 5:
    years_sv = years_era - years_era[0]
    sl_sv, ic_sv, _, p_sv, _ = linregress(years_sv[valid_sv], sv_lat_max[valid_sv])
    print(f"Sverdrup max transport latitude trend: {sl_sv*10:+.4f}°/dec, p={p_sv:.5f}")
    print(f"Sverdrup latitude range: {np.nanmin(sv_lat_max):.2f} ~ {np.nanmax(sv_lat_max):.2f}")
    centroid_results['Sverdrup prediction'] = {
        'trend_deg_per_decade': round(sl_sv * 10, 5),
        'p_value': round(p_sv, 5),
        'mean_lat': round(float(np.nanmean(sv_lat_max)), 3),
    }

era5.close()

# ═══════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════

fig, axes = plt.subplots(3, 1, figsize=(14, 14))
dates = pd.to_datetime(time_arr)

# (a) Three velocity methods comparison
ax = axes[0]
for name, vals, color, ls in [
    ('Velocity max', ke_vmax, 'darkred', '-'),
    ('Velocity centroid', ke_centroid, 'darkorange', '--'),
    ('Half-max midpoint', ke_halfmax, 'purple', ':'),
]:
    valid = ~np.isnan(vals)
    if valid.sum() < 30:
        continue
    smooth = pd.Series(vals).rolling(12, center=True, min_periods=6).mean().values
    sl, ic, _, p, _ = linregress(time_years[valid], vals[valid])
    ax.plot(dates, smooth, color=color, linewidth=2, linestyle=ls,
            label=f'{name} ({sl*10:+.3f}°/dec, p={p:.3f})')
    ax.plot(dates, sl * time_years + ic, color=color, linewidth=1, linestyle='--', alpha=0.5)

ax.set_ylabel('KE Axis Latitude (°N)')
ax.set_title('(a) Displacement vs Intensification: Three Velocity Tracking Methods')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) Peak velocity time series
ax = axes[1]
smooth_pv = pd.Series(peak_vel).rolling(12, center=True, min_periods=6).mean().values
ax.plot(dates, smooth_pv, 'k-', linewidth=2,
        label=f'Peak velocity ({sl_pv*10:+.3f} m/s/dec, p={p_pv:.3f})')
ax.plot(dates, sl_pv * time_years + np.mean(peak_vel[valid_pv]), 'r--', linewidth=1.5)
ax.set_ylabel('Peak Eastward Velocity (m/s)')
ax.set_title('(b) KE Jet Intensification: Peak Velocity Trend')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Sverdrup predicted latitude vs observed KE latitude
ax = axes[2]
if valid_sv.sum() > 5:
    dates_sv = pd.to_datetime(time_annual)
    # Normalize both to anomalies for comparison
    sv_norm = sv_lat_max - np.nanmean(sv_lat_max)
    ke_annual = pd.Series(ke_vmax, index=pd.to_datetime(time_arr)).resample('YE').mean()
    ke_annual_vals = ke_annual.values[:len(sv_lat_max)]
    ke_norm = ke_annual_vals - np.nanmean(ke_annual_vals)

    ax.plot(dates_sv, sv_norm, 'r-o', markersize=4, linewidth=1.5,
            label=f'Sverdrup prediction ({sl_sv*10:+.3f}°/dec)')
    ax.plot(dates_sv[:len(ke_norm)], ke_norm, 'b-o', markersize=4, linewidth=1.5,
            label=f'Observed KE velocity max')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('Latitude Anomaly (°)')
    ax.set_title('(c) Sverdrup-Predicted vs Observed KE Latitude (anomalies)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(FIG / "fig8_centroid_sverdrup.png", dpi=300, bbox_inches='tight')
print(f"\n图已保存: {FIG / 'fig8_centroid_sverdrup.png'}")
plt.close()

# Save results
with open(OUT / "centroid_sverdrup_stats.json", "w") as f:
    json.dump(centroid_results, f, indent=2)
print(json.dumps(centroid_results, indent=2))
