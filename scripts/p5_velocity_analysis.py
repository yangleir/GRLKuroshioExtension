"""P5: Velocity-based KE jet axis tracking and trend analysis
- Compute geostrophic velocity trends in KE region (south decel / north accel pattern)
- Track jet axis via velocity maximum method
- Compare with SLA gradient method
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
                files.append(all_nc[len(all_nc)//2])

print(f"共 {len(files)} 个月度采样文件")

# ── 2. 提取 KE 区域 ugos ──
lon_min, lon_max = 142, 170
lat_min, lat_max = 28, 42

ugos_list = []
speed_list = []
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

    u = ke['ugos'].isel(time=0).values
    v = ke['vgos'].isel(time=0).values
    spd = np.sqrt(u**2 + v**2)
    ugos_list.append(u)
    speed_list.append(spd)
    times.append(ke.time.values[0])
    ds.close()

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(files)}")

ugos_arr = np.array(ugos_list)
speed_arr = np.array(speed_list)
time_arr = np.array(times)
lat = ke.latitude.values
lon_ke = ke.longitude.values
dlat = np.abs(np.diff(lat).mean())

print(f"数据形状: {ugos_arr.shape}, 时间: {str(time_arr[0])[:10]} → {str(time_arr[-1])[:10]}")

# ── 3. 流速趋势空间分布 ──
print("\n计算 KE 区域流速线性趋势...")
time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_arr])

ny, nx = len(lat), len(lon_ke)
trend_speed = np.full((ny, nx), np.nan)
trend_ugos = np.full((ny, nx), np.nan)

for i in range(ny):
    for j in range(nx):
        # speed trend
        y = speed_arr[:, i, j]
        valid = ~np.isnan(y)
        if valid.sum() > 30:
            sl, _, _, _, _ = linregress(time_years[valid], y[valid])
            trend_speed[i, j] = sl * 10  # per decade
        # ugos trend
        y2 = ugos_arr[:, i, j]
        valid2 = ~np.isnan(y2)
        if valid2.sum() > 30:
            sl2, _, _, _, _ = linregress(time_years[valid2], y2[valid2])
            trend_ugos[i, j] = sl2 * 10

print(f"流速趋势范围: {np.nanmin(trend_speed):.4f} ~ {np.nanmax(trend_speed):.4f} m/s/decade")

# ── 4. 流速极大法追踪射流轴 ──
print("\n用 ugos 极大法追踪 KE 轴位置...")

ke_axis_vel = np.full(len(time_arr), np.nan)
for t in range(len(time_arr)):
    frame = ugos_arr[t]
    jet_lats = []
    for j in range(frame.shape[1]):
        col = frame[:, j]
        if np.isnan(col).sum() > len(col) * 0.3:
            continue
        valid_mask = ~np.isnan(col)
        col_interp = np.interp(np.arange(len(col)), np.where(valid_mask)[0], col[valid_mask])
        col_s = gaussian_filter1d(col_interp, sigma=2)
        # 在 32-40°N 找 ugos 极大值
        mask = (lat >= 32) & (lat <= 40)
        col_masked = col_s.copy()
        col_masked[~mask] = -np.inf
        idx = np.argmax(col_masked)
        if col_masked[idx] > 0.05:  # 至少 5 cm/s
            # parabolic interpolation
            if 0 < idx < len(col_s) - 1:
                y0, y1, y2 = col_s[idx-1], col_s[idx], col_s[idx+1]
                denom = 2 * (2 * y1 - y0 - y2)
                if abs(denom) > 1e-10:
                    offset = (y0 - y2) / denom
                    jet_lats.append(lat[idx] + offset * dlat)
                else:
                    jet_lats.append(lat[idx])
            else:
                jet_lats.append(lat[idx])
    if len(jet_lats) >= frame.shape[1] * 0.3:
        ke_axis_vel[t] = np.median(jet_lats)

valid_v = ~np.isnan(ke_axis_vel)
sl_v, ic_v, r_v, p_v, se_v = linregress(time_years[valid_v], ke_axis_vel[valid_v])
print(f"流速极大法 KE 轴趋势: {sl_v*10:+.4f}°/dec, p={p_v:.4f}")
print(f"均值: {np.nanmean(ke_axis_vel):.2f}°N, 有效月: {valid_v.sum()}")

# ── 5. 加载 SLA 梯度法结果对比 ──
ke_sla_ds = xr.open_dataset(OUT / 'ke_axis_position.nc')
# 对齐时间（SLA 是 1993-2025, velocity 是 1993-2021）
ke_sla_vals = ke_sla_ds['ke_axis_latitude'].sel(time=slice(str(time_arr[0])[:10], str(time_arr[-1])[:10])).values
ke_sla_time = ke_sla_ds['ke_axis_latitude'].sel(time=slice(str(time_arr[0])[:10], str(time_arr[-1])[:10])).time.values

# SLA 梯度法趋势 (1993-2021 同期)
sla_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in ke_sla_time])
valid_s = ~np.isnan(ke_sla_vals)
sl_s, _, _, p_s, _ = linregress(sla_years[valid_s], ke_sla_vals[valid_s])
print(f"SLA 梯度法趋势 (1993-2021): {sl_s*10:+.4f}°/dec, p={p_s:.4f}")

# ── 6. 绘图 ──
fig = plt.figure(figsize=(16, 16))

# (a) 流速趋势空间分布
ax1 = fig.add_subplot(3, 1, 1)
lon2d, lat2d = np.meshgrid(lon_ke, lat)
clevels = np.linspace(-0.08, 0.08, 33)
cs = ax1.contourf(lon2d, lat2d, trend_speed, levels=clevels, cmap='RdBu_r', extend='both')
plt.colorbar(cs, ax=ax1, label='Speed trend (m/s per decade)')
# 叠加气候态流速等值线
speed_clim = np.nanmean(speed_arr, axis=0)
ax1.contour(lon2d, lat2d, speed_clim, levels=[0.2, 0.3, 0.4, 0.5], colors='k', linewidths=0.8)
ax1.set_title('(a) Geostrophic Speed Trend (1993-2021) with Climatological Speed Contours')
ax1.set_xlabel('Longitude (°E)')
ax1.set_ylabel('Latitude (°N)')

# (b) 流速极大法 vs SLA 梯度法
ax2 = fig.add_subplot(3, 1, 2)
dates_v = pd.to_datetime(time_arr)
smooth_v = pd.Series(ke_axis_vel).rolling(12, center=True, min_periods=6).mean().values
ax2.plot(dates_v, ke_axis_vel, color='darkred', linewidth=0.3, alpha=0.3)
ax2.plot(dates_v, smooth_v, color='darkred', linewidth=2, label=f'Velocity max ({sl_v*10:+.3f}°/dec, p={p_v:.3f})')
ax2.plot(dates_v, sl_v * time_years + ic_v, '--', color='darkred', linewidth=1.5)

# SLA 梯度法叠加
dates_s = pd.to_datetime(ke_sla_time)
smooth_s = pd.Series(ke_sla_vals).rolling(12, center=True, min_periods=6).mean().values
ax2.plot(dates_s, smooth_s, color='navy', linewidth=2, alpha=0.7, label=f'SLA gradient ({sl_s*10:+.3f}°/dec, p={p_s:.3f})')

ax2.set_ylabel('KE Axis Latitude (°N)')
ax2.set_title('(b) KE Jet Axis: Velocity Maximum vs SLA Gradient Maximum')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# (c) 经向流速剖面对比：前10年 vs 后10年
ax3 = fig.add_subplot(3, 1, 3)
# 142-170°E 平均的经向 ugos 剖面
early = ugos_arr[:120, :, :].mean(axis=(0, 2))  # ~前10年
late = ugos_arr[-120:, :, :].mean(axis=(0, 2))   # ~后10年
ax3.plot(lat, early, 'b-', linewidth=2, label='1993-2002 mean')
ax3.plot(lat, late, 'r-', linewidth=2, label='2012-2021 mean')
ax3.fill_between(lat, early, late, where=late > early, color='red', alpha=0.2, label='Acceleration')
ax3.fill_between(lat, early, late, where=late < early, color='blue', alpha=0.2, label='Deceleration')
ax3.axhline(0, color='gray', linewidth=0.5)
ax3.set_xlabel('Latitude (°N)')
ax3.set_ylabel('Eastward velocity (m/s)')
ax3.set_title('(c) Meridional Profile of Zonal Velocity: Early vs Late Decade')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "fig6_velocity_analysis.png", dpi=300, bbox_inches='tight')
print(f"\n图已保存: {FIG / 'fig6_velocity_analysis.png'}")
plt.close()

# ── 7. 保存统计 ──
stats = {
    "velocity_max_method": {
        "trend_deg_per_decade": round(sl_v * 10, 5),
        "p_value": round(p_v, 5),
        "mean_lat": round(float(np.nanmean(ke_axis_vel)), 3),
        "valid_months": int(valid_v.sum()),
    },
    "sla_gradient_1993_2021": {
        "trend_deg_per_decade": round(sl_s * 10, 5),
        "p_value": round(p_s, 5),
    },
}
with open(OUT / "velocity_analysis_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
