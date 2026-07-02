"""Fix Gulf Stream figure: (a) speed trend color fill + (c) legend position"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import pandas as pd

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})

FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")
ROOT = Path("/Volumes/Backup Plus/ssh/dataset-duacs-rep-global-merged-allsat-phy-l4")

# ── 收集文件 ──
files = []
for year in range(1993, 2022):
    for month in range(1, 13):
        day_dir = ROOT / str(year) / f"{month:02d}"
        if not day_dir.exists(): continue
        target = f"dt_global_allsat_phy_l4_{year}{month:02d}15"
        cands = sorted(day_dir.glob(f"{target}*.nc"))
        if cands: files.append(cands[0])
        else:
            all_nc = sorted(day_dir.glob("*.nc"))
            if len(all_nc) >= 15: files.append(all_nc[14])
            elif all_nc: files.append(all_nc[len(all_nc)//2])

# ── 读 GS 区域（285-315°E = 75-45°W）──
print("读取 Gulf Stream 数据（285-315°E）...")
ugos_list, speed_list, sla_list, times = [], [], [], []
for i, fp in enumerate(files):
    ds = xr.open_dataset(fp)
    lon_data = ds.longitude.values
    if lon_data.max() > 180:
        gs = ds.sel(latitude=slice(33, 47), longitude=slice(285, 315))
    else:
        gs = ds.sel(latitude=slice(33, 47), longitude=slice(-75, -45))
    if len(gs.longitude) == 0:
        ds.close(); continue
    # 统一经度到 285-315 (0-360)
    if gs.longitude.values.min() < 0:
        gs = gs.assign_coords(longitude=(gs.longitude % 360))
        gs = gs.sortby('longitude')
    u = gs['ugos'].isel(time=0).values
    v = gs['vgos'].isel(time=0).values
    ugos_list.append(u)
    speed_list.append(np.sqrt(u**2 + v**2))
    sla_list.append(gs['sla'].isel(time=0).values)
    times.append(gs.time.values[0])
    lat = gs.latitude.values
    lon_gs = gs.longitude.values
    ds.close()
    if (i+1) % 100 == 0: print(f"  {i+1}/{len(files)}")

ugos_arr = np.array(ugos_list)
speed_arr = np.array(speed_list)
sla_arr = np.array(sla_list)
time_arr = np.array(times)
lat = gs.latitude.values
lon_gs = gs.longitude.values
dlat = np.abs(np.diff(lat).mean())
time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_arr])

print(f"数据: {ugos_arr.shape}, 经度 {lon_gs.min():.1f}-{lon_gs.max():.1f}")

# ── 计算速度趋势 ──
print("计算速度趋势...")
ny, nx = len(lat), len(lon_gs)
trend_speed = np.full((ny, nx), np.nan)
for i in range(ny):
    for j in range(nx):
        y = speed_arr[:, i, j]
        valid = ~np.isnan(y)
        if valid.sum() > 30:
            sl, _, _, _, _ = linregress(time_years[valid], y[valid])
            trend_speed[i, j] = sl * 10

speed_clim = np.nanmean(speed_arr, axis=0)
print(f"趋势范围: {np.nanmin(trend_speed):.4f} ~ {np.nanmax(trend_speed):.4f}")
print(f"有效格点: {np.sum(~np.isnan(trend_speed))}/{ny*nx}")

# ── 追踪射流轴 ──
def track_axis(data, lat_arr, dlat_val, method='max'):
    nt = data.shape[0]
    axis = np.full(nt, np.nan)
    for t in range(nt):
        frame = data[t]
        lats = []
        for j in range(frame.shape[1]):
            col = frame[:, j]
            valid = ~np.isnan(col)
            if valid.sum() < 5: continue
            col_i = np.interp(np.arange(len(col)), np.where(valid)[0], col[valid])
            col_s = gaussian_filter1d(col_i, sigma=2)
            mask = (lat_arr >= lat_arr.min()+2) & (lat_arr <= lat_arr.max()-2)
            if method == 'max':
                col_m = np.where(mask, col_s, -np.inf)
                idx = np.argmax(col_m)
                if col_m[idx] > 0.03: lats.append(lat_arr[idx])
            else:
                grad = np.gradient(col_s, dlat_val)
                gm = np.where(mask, grad, -np.inf)
                idx = np.argmax(gm)
                if gm[idx] > 0: lats.append(lat_arr[idx])
        if len(lats) >= frame.shape[1] * 0.3:
            axis[t] = np.median(lats)
    return axis

vel_gs = track_axis(ugos_arr, lat, dlat, 'max')
sla_gs = track_axis(sla_arr, lat, dlat, 'grad')

valid_v = ~np.isnan(vel_gs)
valid_s = ~np.isnan(sla_gs)
sl_v, ic_v, _, p_v, _ = linregress(time_years[valid_v], vel_gs[valid_v])
sl_s, ic_s, _, p_s, _ = linregress(time_years[valid_s], sla_gs[valid_s])
print(f"GS vel max: {sl_v*10:+.3f}°/dec, p={p_v:.3f}")
print(f"GS SLA grad: {sl_s*10:+.3f}°/dec, p={p_s:.3f}")

# ── 绘图 ──
# 经度转为 °W 显示
lon_display = lon_gs - 360  # 285-315 → -75 to -45

early = ugos_arr[:120].mean(axis=(0, 2))
late = ugos_arr[-120:].mean(axis=(0, 2))

fig = plt.figure(figsize=(10, 14))
gs_grid = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.35)

# (a) Speed trend
ax = fig.add_subplot(gs_grid[0])
lon2d, lat2d = np.meshgrid(lon_display, lat)
clevels = np.linspace(-0.08, 0.08, 33)
cs = ax.contourf(lon2d, lat2d, trend_speed, levels=clevels, cmap='RdBu_r', extend='both')
cb = plt.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('Speed trend (m/s per decade)', fontsize=12)
cb.ax.tick_params(labelsize=10)
ax.contour(lon2d, lat2d, speed_clim, levels=[0.2, 0.3, 0.4, 0.5], colors='k', linewidths=1)
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
ax.set_title('(a) Gulf Stream Geostrophic Speed Trend (1993–2021)', fontweight='bold')

# (b) Vel max vs SLA gradient
ax = fig.add_subplot(gs_grid[1])
dates = pd.to_datetime(time_arr)
smooth_v = pd.Series(vel_gs).rolling(12, center=True, min_periods=6).mean().values
smooth_s = pd.Series(sla_gs).rolling(12, center=True, min_periods=6).mean().values
ax.plot(dates, smooth_v, 'r-', linewidth=2.5, label=f'Velocity max: {sl_v*10:+.2f}°/dec (p={p_v:.3f})')
ax.plot(dates, sl_v*time_years+ic_v, 'r--', linewidth=1.5)
ax.plot(dates, smooth_s, 'b-', linewidth=2.5, label=f'SLA gradient: {sl_s*10:+.2f}°/dec (p={p_s:.3f})')
ax.plot(dates, sl_s*time_years+ic_s, 'b--', linewidth=1.5)
ax.set_ylabel('GS Axis Latitude (°N)')
ax.set_title('(b) Gulf Stream: Velocity Maximum vs SLA Gradient Maximum', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# (c) Meridional velocity profile — legend 放左上角不遮挡峰值
ax = fig.add_subplot(gs_grid[2])
ax.plot(lat, early, 'b-', linewidth=2.5, label='1993–2002 mean')
ax.plot(lat, late, 'r-', linewidth=2.5, label='2012–2021 mean')
ax.fill_between(lat, early, late, where=late > early, color='red', alpha=0.15, label='Acceleration')
ax.fill_between(lat, early, late, where=late < early, color='blue', alpha=0.15, label='Deceleration')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Latitude (°N)')
ax.set_ylabel('Eastward Velocity (m/s)')
ax.set_title('(c) Meridional Zonal Velocity Profile: Early vs Late Decade', fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.savefig(FIG / "fig2_gs_velocity_v3.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "fig2_gs_velocity_v3.png", dpi=300, bbox_inches='tight')
print(f"\n保存: fig2_gs_velocity_v3")
plt.close()

import shutil
shutil.copy(FIG / "fig2_gs_velocity_v3.pdf", "/Users/zhulin/aitest/黑潮延伸体/manuscript/fig7.pdf")
print("已复制到 manuscript/fig7.pdf")
