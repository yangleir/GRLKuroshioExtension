"""P1b: Validate KE axis tracking using ADT (absolute dynamic topography)
ADT = SLA + MSSH. Use daily data sampled monthly (15th of each month).
Compare ADT gradient method vs SLA gradient method.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import pandas as pd

OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")
ROOT = Path("/Volumes/Backup Plus/ssh/dataset-duacs-rep-global-merged-allsat-phy-l4")

# ── 1. 收集每月 15 日文件 ──
print("收集日数据文件（每月15日采样）...")
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
                files.append(all_nc[len(all_nc)//2])

print(f"共 {len(files)} 个月度采样文件")

# ── 2. 提取 KE 区域 ADT 和 SLA ──
lon_min, lon_max = 142, 170
lat_min, lat_max = 30, 42

adt_list = []
sla_list = []
times = []

for i, fp in enumerate(files):
    ds = xr.open_dataset(fp)
    lon = ds.longitude.values
    if lon.min() >= 0 and lon.max() > 180:
        ke = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    else:
        ke = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

    if len(ke.longitude) == 0:
        ds.close()
        continue

    adt_list.append(ke['adt'].isel(time=0).values)
    sla_list.append(ke['sla'].isel(time=0).values)
    times.append(ke.time.values[0])
    ds.close()

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(files)}")

adt_arr = np.array(adt_list)
sla_arr = np.array(sla_list)
time_arr = np.array(times)
lat = ke.latitude.values
lon_ke = ke.longitude.values
dlat = np.abs(np.diff(lat).mean())

print(f"数据形状: {adt_arr.shape}, 时间: {str(time_arr[0])[:10]} → {str(time_arr[-1])[:10]}")

# ── 3. 梯度极大法追踪（ADT vs SLA） ──
def track_axis(data, lat_arr, dlat_val, lat_search=(32, 40)):
    nt = data.shape[0]
    axis_lat = np.full(nt, np.nan)
    for t in range(nt):
        frame = data[t]
        jet_lats = []
        for j in range(frame.shape[1]):
            col = frame[:, j]
            if np.isnan(col).sum() > len(col) * 0.3:
                continue
            valid_mask = ~np.isnan(col)
            col_interp = np.interp(np.arange(len(col)), np.where(valid_mask)[0], col[valid_mask])
            col_s = gaussian_filter1d(col_interp, sigma=2)
            grad = np.gradient(col_s, dlat_val)
            mask = (lat_arr >= lat_search[0]) & (lat_arr <= lat_search[1])
            grad_m = grad.copy()
            grad_m[~mask] = -np.inf
            idx = np.argmax(grad_m)
            if grad_m[idx] > 0:
                # Parabolic interpolation for sub-grid accuracy
                if 0 < idx < len(grad) - 1:
                    y0, y1, y2 = grad[idx-1], grad[idx], grad[idx+1]
                    denom = 2 * (2 * y1 - y0 - y2)
                    if abs(denom) > 1e-10:
                        offset = (y0 - y2) / denom
                        jet_lats.append(lat_arr[idx] + offset * dlat_val)
                    else:
                        jet_lats.append(lat_arr[idx])
                else:
                    jet_lats.append(lat_arr[idx])
        if len(jet_lats) >= frame.shape[1] * 0.3:
            axis_lat[t] = np.median(jet_lats)
    return axis_lat

print("\n追踪 KE 轴...")
ke_adt = track_axis(adt_arr, lat, dlat)
ke_sla = track_axis(sla_arr, lat, dlat)

# ── 4. 趋势对比 ──
time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_arr])

print("\n=== ADT vs SLA 梯度法趋势对比 ===")
results = {}
for name, vals in [('ADT gradient', ke_adt), ('SLA gradient', ke_sla)]:
    valid = ~np.isnan(vals)
    sl, ic, r, p, se = linregress(time_years[valid], vals[valid])
    print(f"  {name}: {sl*10:+.4f}°/dec, p={p:.4f} {'★' if p < 0.05 else ''}")
    results[name] = {'trend_per_decade': round(sl*10, 5), 'p_value': round(p, 5)}

# ── 5. 绘图 ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
dates = pd.to_datetime(time_arr)

for ax_idx, (name, vals, color) in enumerate([
    ('ADT gradient method', ke_adt, 'darkred'),
    ('SLA gradient method', ke_sla, 'navy'),
]):
    ax = axes[ax_idx]
    valid = ~np.isnan(vals)
    smooth = pd.Series(vals).rolling(12, center=True, min_periods=6).mean().values

    ax.plot(dates, vals, color=color, linewidth=0.5, alpha=0.3)
    ax.plot(dates, smooth, color=color, linewidth=2, label=f'{name} (12-mo mean)')

    sl, ic, _, p, _ = linregress(time_years[valid], vals[valid])
    ax.plot(dates, sl * time_years + ic, '--', color=color, linewidth=1.5,
            label=f'Trend: {sl*10:+.4f}°/dec (p={p:.3f})')

    ax.axvline(pd.Timestamp("2017-08-01"), color='green', linewidth=1, linestyle=':', alpha=0.5)
    ax.set_ylabel('KE Axis Latitude (°N)')
    ax.set_title(f'({"a" if ax_idx==0 else "b"}) {name}')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(FIG / "fig1_adt_vs_sla_validation.png", dpi=300, bbox_inches='tight')
print(f"\n图已保存: {FIG / 'fig1_adt_vs_sla_validation.png'}")
plt.close()

# ── 6. 保存 ──
import json
with open(OUT / "adt_vs_sla_stats.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
