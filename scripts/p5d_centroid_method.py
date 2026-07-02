"""R11 Task 1: Velocity centroid method — weighted latitude using velocity field
More robust than single-point maximum. Compares centroid vs max vs SLA gradient.
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

print("收集日数据文件...")
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

print(f"共 {len(files)} 个文件")

vel_max_lat = []
vel_centroid_lat = []
sla_grad_lat = []
times = []

for i, fp in enumerate(files):
    ds = xr.open_dataset(fp)
    ke = ds.sel(latitude=slice(30, 42), longitude=slice(142, 170))
    if len(ke.longitude) == 0:
        ds.close()
        continue

    ugos = ke['ugos'].isel(time=0).values
    sla = ke['sla'].isel(time=0).values
    lat = ke.latitude.values
    dlat = np.abs(np.diff(lat).mean())

    # --- Velocity maximum ---
    max_lats = []
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
        if col_m[idx] > 0:
            max_lats.append(lat[idx])
    vel_max_lat.append(np.median(max_lats) if len(max_lats) >= ugos.shape[1]*0.3 else np.nan)

    # --- Velocity centroid (weighted latitude) ---
    centroid_lats = []
    for j in range(ugos.shape[1]):
        col = ugos[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 5:
            continue
        mask = (lat >= 32) & (lat <= 40)
        u_pos = np.where(mask & valid & (col > 0), col, 0)
        total = np.sum(u_pos)
        if total > 0:
            centroid_lats.append(np.sum(u_pos * lat) / total)
    vel_centroid_lat.append(np.median(centroid_lats) if len(centroid_lats) >= ugos.shape[1]*0.3 else np.nan)

    # --- SLA gradient maximum ---
    grad_lats = []
    for j in range(sla.shape[1]):
        col = sla[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 5:
            continue
        col_i = np.interp(np.arange(len(col)), np.where(valid)[0], col[valid])
        col_s = gaussian_filter1d(col_i, sigma=2)
        grad = np.gradient(col_s, dlat)
        mask = (lat >= 32) & (lat <= 40)
        grad_m = np.where(mask, grad, -np.inf)
        idx = np.argmax(grad_m)
        if grad_m[idx] > 0:
            grad_lats.append(lat[idx])
    sla_grad_lat.append(np.median(grad_lats) if len(grad_lats) >= sla.shape[1]*0.3 else np.nan)

    times.append(ke.time.values[0])
    ds.close()
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(files)}")

vel_max_lat = np.array(vel_max_lat)
vel_centroid_lat = np.array(vel_centroid_lat)
sla_grad_lat = np.array(sla_grad_lat)
times = np.array(times)
time_years = np.arange(len(times), dtype=float) / 12.0

print("\n=== 三方法趋势对比 ===")
results = {}
for name, vals in [('Velocity max', vel_max_lat), ('Velocity centroid', vel_centroid_lat), ('SLA gradient', sla_grad_lat)]:
    valid = ~np.isnan(vals)
    sl, ic, r, p, se = linregress(time_years[valid], vals[valid])
    print(f"  {name}: {sl*10:+.4f}°/dec, p={p:.5f}")
    results[name] = {'trend_per_decade': round(sl*10, 5), 'p_value': round(p, 5)}

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
dates = pd.to_datetime(times)

for name, vals, color, marker in [
    ('Velocity max', vel_max_lat, 'red', '-'),
    ('Velocity centroid', vel_centroid_lat, 'darkgreen', '-'),
    ('SLA gradient', sla_grad_lat, 'navy', '-'),
]:
    smooth = pd.Series(vals).rolling(12, center=True, min_periods=6).mean().values
    valid = ~np.isnan(vals)
    sl, ic, _, p, _ = linregress(time_years[valid], vals[valid])
    ax.plot(dates, smooth, color=color, linewidth=2,
            label=f'{name}: {sl*10:+.3f}°/dec (p={p:.3f})')
    ax.plot(dates, sl*time_years + ic, '--', color=color, linewidth=1, alpha=0.5)

ax.set_ylabel('KE Axis Latitude (°N)')
ax.set_xlabel('Year')
ax.set_title('Three Methods for KE Jet Axis Tracking (1993-2021)')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(FIG / "fig_three_methods.png", dpi=300, bbox_inches='tight')
print(f"\n图已保存: {FIG / 'fig_three_methods.png'}")
plt.close()

with open(OUT / "three_method_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
