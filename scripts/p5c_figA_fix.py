"""Fix Fig A: overlay velocity-max axis + wind curl zero line (standardized)
Read velocity axis from P5's approach (recompute annual means)
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

# ── 1. Recompute velocity-max axis (monthly, 1993-2021) ──
print("计算流速极大法 KE 轴...")
files = []
for year in range(1993, 2022):
    for month in range(1, 13):
        day_dir = ROOT / str(year) / f"{month:02d}"
        if not day_dir.exists():
            continue
        target = f"dt_global_allsat_phy_l4_{year}{month:02d}15"
        cands = sorted(day_dir.glob(f"{target}*.nc"))
        if cands:
            files.append(cands[0])
        else:
            all_nc = sorted(day_dir.glob("*.nc"))
            if len(all_nc) >= 15:
                files.append(all_nc[14])
            elif all_nc:
                files.append(all_nc[len(all_nc)//2])

vel_axis = []
vel_times = []
for i, fp in enumerate(files):
    ds = xr.open_dataset(fp)
    ke = ds.sel(latitude=slice(30, 42), longitude=slice(142, 170))
    if len(ke.longitude) == 0:
        ds.close()
        continue
    ugos = ke['ugos'].isel(time=0).values
    lat = ke.latitude.values
    dlat = np.abs(np.diff(lat).mean())
    jet_lats = []
    for j in range(ugos.shape[1]):
        col = ugos[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 5:
            continue
        col_i = np.interp(np.arange(len(col)), np.where(valid)[0], col[valid])
        col_s = gaussian_filter1d(col_i, sigma=2)
        mask = (lat >= 32) & (lat <= 40)
        col_m = np.where(mask, col_s, -np.inf)
        idx = np.argmax(col_m)
        if col_m[idx] > 0 and 0 < idx < len(col_m) - 1:
            y0, y1, y2 = col_m[idx-1], col_m[idx], col_m[idx+1]
            denom = 2 * (2*y1 - y0 - y2)
            if abs(denom) > 1e-10:
                offset = (y0 - y2) / denom
                jet_lats.append(lat[idx] + offset * dlat)
            else:
                jet_lats.append(lat[idx])
    if len(jet_lats) >= ugos.shape[1] * 0.3:
        vel_axis.append(np.median(jet_lats))
    else:
        vel_axis.append(np.nan)
    vel_times.append(ke.time.values[0])
    ds.close()
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(files)}")

vel_axis = np.array(vel_axis)
vel_times = np.array(vel_times)
vel_annual = pd.Series(vel_axis, index=pd.to_datetime(vel_times)).resample('YE').mean()

# ── 2. Load zero line (annual, from P4) ──
era5 = xr.open_dataset("/Users/zhulin/aitest/黑潮延伸体/data/era5_monthly_wind_npac_1993_2025.nc")
u10, v10 = era5['u10'], era5['v10']
lat_e, lon_e = era5.latitude.values, era5.longitude.values
rho_a, Cd = 1.225, 1.3e-3
wspd = np.sqrt(u10**2 + v10**2)
tau_x = rho_a * Cd * wspd * u10
tau_y = rho_a * Cd * wspd * v10
R = 6.371e6
dlat_e = np.abs(np.diff(lat_e).mean()) * np.pi / 180
dlon_e = np.abs(np.diff(lon_e).mean()) * np.pi / 180
cos_lat = np.cos(np.deg2rad(era5.latitude))
curl = (tau_y.differentiate('longitude') / (R * cos_lat * dlon_e * (180/np.pi))
        - tau_x.differentiate('latitude') / (R * dlat_e * (180/np.pi)))
curl_ann = curl.resample(valid_time='YE').mean()
lon_mask = (lon_e >= 130) & (lon_e <= 180)
curl_reg = curl_ann.sel(longitude=lon_e[lon_mask])
time_ann = curl_ann.valid_time.values

zero_lat = np.full(len(time_ann), np.nan)
for t in range(len(time_ann)):
    zz = []
    for j in range(curl_reg.shape[2]):
        col = curl_reg.values[t, :, j]
        for i in range(len(lat_e)-1):
            if not np.isnan(col[i]) and not np.isnan(col[i+1]):
                if col[i] > 0 and col[i+1] <= 0 and 20 < lat_e[i] < 50:
                    frac = col[i] / (col[i] - col[i+1])
                    zz.append(lat_e[i] + frac * (lat_e[i+1] - lat_e[i]))
                    break
    if len(zz) >= 5:
        zero_lat[t] = np.median(zz)

zero_annual = pd.Series(zero_lat, index=pd.to_datetime(time_ann))

# ── 3. Plot Fig A (fixed) ──
fig, ax = plt.subplots(figsize=(14, 5))

# Standardize
v_vals = vel_annual.values
v_norm = (v_vals - np.nanmean(v_vals)) / np.nanstd(v_vals)
z_vals = zero_annual.values
z_norm = (z_vals - np.nanmean(z_vals)) / np.nanstd(z_vals)

ax.plot(vel_annual.index, v_norm, 'r-o', markersize=4, linewidth=1.5, label='KE axis: velocity max (normalized)')
ax.plot(zero_annual.index, z_norm, 'b-s', markersize=4, linewidth=1.5, label='Wind curl zero line (normalized)')

# Trend lines
ty_v = np.arange(len(v_vals), dtype=float)
ty_z = np.arange(len(z_vals), dtype=float)
valid_v = ~np.isnan(v_vals)
valid_z = ~np.isnan(z_vals)
sl_v, ic_v, _, _, _ = linregress(ty_v[valid_v], v_vals[valid_v])
sl_z, ic_z, _, _, _ = linregress(ty_z[valid_z], z_vals[valid_z])
ax.plot(vel_annual.index, (sl_v*ty_v + ic_v - np.nanmean(v_vals))/np.nanstd(v_vals),
        'r--', linewidth=1, alpha=0.5)
ax.plot(zero_annual.index, (sl_z*ty_z + ic_z - np.nanmean(z_vals))/np.nanstd(z_vals),
        'b--', linewidth=1, alpha=0.5)

ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('Standardized Anomaly')
ax.set_xlabel('Year')

textstr = ('Rate consistency:\n'
           'Hadley widening    ~0.50°/dec (Seidel 2008)\n'
           f'Wind curl zero line  {sl_z*10:+.2f}°/dec (this study)\n'
           f'Zero line shift      ~0.58°/dec (Wu 2018)\n'
           f'KE velocity max      {sl_v*10:+.2f}°/dec (this study)')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props, family='monospace')

ax.set_title('Causal Chain: Wind Stress Curl Zero Line vs KE Velocity-Maximum Axis')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
out = FIG / "figA_causal_chain_v2.png"
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"保存: {out}")
plt.close()

import shutil
shutil.copy(out, "/Users/zhulin/aitest/黑潮延伸体/manuscript/figA.png")
print("已复制到 manuscript/figA.png")
