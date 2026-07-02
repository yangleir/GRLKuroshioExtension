"""P6: Gulf Stream velocity-based jet tracking — replicate P5 for GS
- Same methodology as KE analysis (velocity max vs SLA gradient)
- GS region: 75-45°W, 33-45°N
- Test whether anomaly-vs-absolute bias is universal
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
                files.append(all_nc[len(all_nc) // 2])

print(f"共 {len(files)} 个月度采样文件")

# ── 2. Gulf Stream 区域参数 ──
# GS 分离点约 35°N, 75°W → 向东延伸
# 经度用 0-360: 75W=285, 45W=315
lon_min_gs, lon_max_gs = 285, 315
lat_min_gs, lat_max_gs = 33, 47

print(f"Gulf Stream 区域: {lon_min_gs}-{lon_max_gs}°E, {lat_min_gs}-{lat_max_gs}°N")

# ── 3. 提取 ugos 和 sla ──
ugos_list = []
sla_list = []
speed_list = []
times = []

for i, fp in enumerate(files):
    ds = xr.open_dataset(fp)
    lon = ds.longitude.values

    if lon.min() < 0:
        ds = ds.assign_coords(longitude=(ds.longitude % 360))
        ds = ds.sortby('longitude')

    gs = ds.sel(latitude=slice(lat_min_gs, lat_max_gs),
                longitude=slice(lon_min_gs, lon_max_gs))

    if len(gs.longitude) == 0:
        ds.close()
        continue

    u = gs['ugos'].isel(time=0).values
    v = gs['vgos'].isel(time=0).values
    sla = gs['sla'].isel(time=0).values
    spd = np.sqrt(u**2 + v**2)

    ugos_list.append(u)
    sla_list.append(sla)
    speed_list.append(spd)
    times.append(gs.time.values[0])
    ds.close()

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(files)}")

ugos_arr = np.array(ugos_list)
sla_arr = np.array(sla_list)
speed_arr = np.array(speed_list)
time_arr = np.array(times)
lat = gs.latitude.values
lon_gs = gs.longitude.values
dlat = np.abs(np.diff(lat).mean())

print(f"数据形状: {ugos_arr.shape}, 时间: {str(time_arr[0])[:10]} → {str(time_arr[-1])[:10]}")

time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25
                        for t in time_arr])

# ── 4. 流速趋势空间分布 ──
print("\n计算 GS 区域流速线性趋势...")
ny, nx = len(lat), len(lon_gs)
trend_speed = np.full((ny, nx), np.nan)

for i in range(ny):
    for j in range(nx):
        y = speed_arr[:, i, j]
        valid = ~np.isnan(y)
        if valid.sum() > 30:
            sl, _, _, _, _ = linregress(time_years[valid], y[valid])
            trend_speed[i, j] = sl * 10

# ── 5. 流速极大法 vs SLA 梯度法 ──
def track_axis(data, lat_arr, dlat_val, lat_search, find_max_positive=True, threshold=0.05):
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
            if valid_mask.sum() < 5:
                continue
            col_interp = np.interp(np.arange(len(col)),
                                   np.where(valid_mask)[0], col[valid_mask])
            col_s = gaussian_filter1d(col_interp, sigma=2)

            if find_max_positive:
                target = col_s.copy()
            else:
                target = np.gradient(col_s, dlat_val)

            mask = (lat_arr >= lat_search[0]) & (lat_arr <= lat_search[1])
            target_m = target.copy()
            target_m[~mask] = -np.inf
            idx = np.argmax(target_m)

            if target_m[idx] > threshold:
                if 0 < idx < len(target) - 1:
                    y0, y1, y2 = target[idx - 1], target[idx], target[idx + 1]
                    denom = 2 * (2 * y1 - y0 - y2)
                    if abs(denom) > 1e-10:
                        offset = (y0 - y2) / denom
                        jet_lats.append(lat_arr[idx] + offset * dlat_val)
                    else:
                        jet_lats.append(lat_arr[idx])
                else:
                    jet_lats.append(lat_arr[idx])

        if len(jet_lats) >= frame.shape[1] * 0.2:
            axis_lat[t] = np.median(jet_lats)
    return axis_lat

print("追踪 GS 轴（流速极大法）...")
gs_vel = track_axis(ugos_arr, lat, dlat, (35, 44), find_max_positive=True, threshold=0.05)

print("追踪 GS 轴（SLA 梯度法）...")
gs_sla = track_axis(sla_arr, lat, dlat, (35, 44), find_max_positive=False, threshold=0)

# ── 6. 趋势计算 ──
results = {}
for name, vals in [('GS_velocity_max', gs_vel), ('GS_SLA_gradient', gs_sla)]:
    valid = ~np.isnan(vals)
    if valid.sum() > 30:
        sl, ic, r, p, se = linregress(time_years[valid], vals[valid])
        print(f"  {name}: {sl * 10:+.4f}°/dec, p={p:.5f}, mean={np.nanmean(vals):.2f}°N, valid={valid.sum()}")
        results[name] = {
            'trend_deg_per_decade': round(sl * 10, 5),
            'p_value': round(p, 5),
            'mean_lat': round(float(np.nanmean(vals)), 3),
            'valid_months': int(valid.sum()),
        }

# N-S SLA differential for GS
sla_north = sla_arr[:, lat >= 40, :].mean(axis=(1, 2))
sla_south = sla_arr[:, lat <= 38, :].mean(axis=(1, 2))
ns_diff = sla_north - sla_south
valid_ns = ~np.isnan(ns_diff)
if valid_ns.sum() > 30:
    sl_ns, _, _, p_ns, _ = linregress(time_years[valid_ns], ns_diff[valid_ns])
    print(f"  GS N-S SLA diff trend: {sl_ns * 10 * 100:.2f} cm/dec, p={p_ns:.5f}")
    results['GS_NS_SLA_diff'] = {
        'trend_cm_per_decade': round(sl_ns * 10 * 100, 3),
        'p_value': round(p_ns, 5),
    }

# ── 7. 绘图 ──
fig = plt.figure(figsize=(16, 16))

# (a) 流速趋势空间分布
ax1 = fig.add_subplot(3, 1, 1)
lon2d, lat2d = np.meshgrid(lon_gs - 360, lat)  # 转回负经度显示
clevels = np.linspace(-0.08, 0.08, 33)
cs = ax1.contourf(lon2d, lat2d, trend_speed, levels=clevels, cmap='RdBu_r', extend='both')
plt.colorbar(cs, ax=ax1, label='Speed trend (m/s per decade)')
speed_clim = np.nanmean(speed_arr, axis=0)
ax1.contour(lon2d, lat2d, speed_clim, levels=[0.15, 0.2, 0.3, 0.4], colors='k', linewidths=0.8)
ax1.set_title('(a) Gulf Stream Geostrophic Speed Trend (1993-2021)')
ax1.set_xlabel('Longitude (°W)')
ax1.set_ylabel('Latitude (°N)')

# (b) 流速极大 vs SLA 梯度
ax2 = fig.add_subplot(3, 1, 2)
dates = pd.to_datetime(time_arr)

for name, vals, color, label_prefix in [
    ('vel', gs_vel, 'darkred', 'Velocity max'),
    ('sla', gs_sla, 'navy', 'SLA gradient'),
]:
    valid = ~np.isnan(vals)
    if valid.sum() < 30:
        continue
    smooth = pd.Series(vals).rolling(12, center=True, min_periods=6).mean().values
    sl, ic, _, p, _ = linregress(time_years[valid], vals[valid])
    ax2.plot(dates, smooth, color=color, linewidth=2,
             label=f'{label_prefix} ({sl * 10:+.3f}°/dec, p={p:.3f})')
    ax2.plot(dates, sl * time_years + ic, '--', color=color, linewidth=1.5)

ax2.set_ylabel('GS Axis Latitude (°N)')
ax2.set_title('(b) Gulf Stream: Velocity Maximum vs SLA Gradient Maximum')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# (c) 前后十年经向流速剖面
ax3 = fig.add_subplot(3, 1, 3)
n_early = min(120, len(ugos_arr))
early = ugos_arr[:n_early, :, :].mean(axis=(0, 2))
late = ugos_arr[-n_early:, :, :].mean(axis=(0, 2))
ax3.plot(lat, early, 'b-', linewidth=2, label='1993-2002 mean')
ax3.plot(lat, late, 'r-', linewidth=2, label='2012-2021 mean')
ax3.fill_between(lat, early, late, where=late > early, color='red', alpha=0.2)
ax3.fill_between(lat, early, late, where=late < early, color='blue', alpha=0.2)
ax3.axhline(0, color='gray', linewidth=0.5)
ax3.set_xlabel('Latitude (°N)')
ax3.set_ylabel('Eastward velocity (m/s)')
ax3.set_title('(c) GS Meridional Velocity Profile: Early vs Late Decade')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "fig7_gulf_stream_velocity.png", dpi=300, bbox_inches='tight')
print(f"\n图已保存: {FIG / 'fig7_gulf_stream_velocity.png'}")
plt.close()

# ── 8. 保存 ──
with open(OUT / "gulf_stream_stats.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
