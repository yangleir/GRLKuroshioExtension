"""重出 fig7（Gulf Stream，1993-2024 统一时段），图例用 Bretherton 校正 p 值。
布局与 fig2_gs_velocity_v4 相同：a 速度趋势场、b velocity-max 序列（MY/NRT 分界线）、c 早晚剖面。
同时缓存 GS 轴序列到 output/gs_axis_unified.nc。
输出 figures/fig2_gs_velocity_v5.{png,pdf} 并拷贝为 manuscript/fig7.{png,pdf}
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import pandas as pd
import shutil
from pathlib import Path
from scipy.stats import linregress, t as t_dist
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})

FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")
MAN = Path("/Users/zhulin/aitest/黑潮延伸体/manuscript")
OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
DRIVE = Path("/Volumes/Backup Plus/ssh")
ROOT = DRIVE / "dataset-duacs-rep-global-merged-allsat-phy-l4"


def bretherton_trend(x, y):
    sl, ic, r, p_raw, se = linregress(x, y)
    residuals = y - (sl * x + ic)
    N = len(y)
    r1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    N_eff = max(N * (1 - r1) / (1 + r1), 3)
    se_c = se * np.sqrt(N / N_eff)
    p_c = 2 * (1 - t_dist.cdf(abs(sl / se_c), df=max(N_eff - 2, 1)))
    return sl, ic, p_c


def fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.2f}"


# ── MY 月中快照 ──
files_my = []
for year in range(1993, 2022):
    for month in range(1, 13):
        day_dir = ROOT / str(year) / f"{month:02d}"
        if not day_dir.exists():
            continue
        target = f"dt_global_allsat_phy_l4_{year}{month:02d}15"
        cands = sorted(day_dir.glob(f"{target}*.nc"))
        if cands:
            files_my.append(cands[0])
        else:
            all_nc = sorted(day_dir.glob("*.nc"))
            if len(all_nc) >= 15:
                files_my.append(all_nc[14])
            elif all_nc:
                files_my.append(all_nc[len(all_nc) // 2])

ugos_list, speed_list, times = [], [], []
lat = None
for fp in files_my:
    d = xr.open_dataset(fp)
    gs = d.sel(latitude=slice(33, 47), longitude=slice(285, 315))
    if len(gs.longitude) == 0:
        gs = d.sel(latitude=slice(33, 47), longitude=slice(-75, -45))
    u = gs['ugos'].isel(time=0).values
    v = gs['vgos'].isel(time=0).values
    ugos_list.append(u)
    speed_list.append(np.sqrt(u ** 2 + v ** 2))
    times.append(gs.time.values[0])
    lat = gs.latitude.values
    lon = gs.longitude.values
    d.close()

# ── NRT 月均（至 2024-12）──
nrt = xr.open_dataset(DRIVE / "cmems_ugos_vgos_nrt_daily_GS_2022_2025.nc")
gs_sub = nrt.sel(latitude=slice(33, 47), longitude=slice(-75, -45))
if len(gs_sub.longitude) == 0:
    gs_sub = nrt.sel(latitude=slice(33, 47), longitude=slice(285, 315))
u_m = gs_sub['ugos'].resample(time='MS').mean().sel(time=slice(None, '2024-12-31'))
v_m = gs_sub['vgos'].resample(time='MS').mean().sel(time=slice(None, '2024-12-31'))
for t in range(len(u_m.time)):
    u = u_m.isel(time=t).values
    v = v_m.isel(time=t).values
    ugos_list.append(u)
    speed_list.append(np.sqrt(u ** 2 + v ** 2))
    times.append(u_m.time.values[t])
nrt.close()

ugos_arr = np.array(ugos_list)
speed_arr = np.array(speed_list)
time_arr = np.array(times)
time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_arr])
print(f"GS 统一: {ugos_arr.shape}, {str(time_arr[0])[:10]} → {str(time_arr[-1])[:10]}")

# ── 追踪 velocity max ──
band = (lat >= lat.min() + 2) & (lat <= lat.max() - 2)
axis_gs = np.full(len(time_arr), np.nan)
for t in range(len(time_arr)):
    frame = ugos_arr[t]
    lats = []
    for j in range(frame.shape[1]):
        col = frame[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 5:
            continue
        col_i = np.interp(np.arange(len(col)), np.where(valid)[0], col[valid])
        col_s = gaussian_filter1d(col_i, sigma=2)
        col_m = np.where(band, col_s, -np.inf)
        idx = np.argmax(col_m)
        if col_m[idx] > 0.03:
            lats.append(lat[idx])
    if len(lats) >= frame.shape[1] * 0.3:
        axis_gs[t] = np.median(lats)

valid = ~np.isnan(axis_gs)
sl_v, ic_v, p_v = bretherton_trend(time_years[valid], axis_gs[valid])
print(f"GS vel max: {sl_v*10:+.3f}°/dec ({fmt_p(p_v)})")

# 缓存序列
xr.Dataset({"gs_axis_latitude": ("time", axis_gs)},
           coords={"time": time_arr},
           attrs={"method": "velocity maximum, Gaussian sigma=2, search 35-45N, threshold 0.03 m/s",
                  "trend_deg_per_decade": float(sl_v * 10),
                  "p_bretherton": float(p_v)}).to_netcdf(OUT / "gs_axis_unified.nc")

# ── 速度趋势场 ──
ny, nx = len(lat), len(lon)
trend_speed = np.full((ny, nx), np.nan)
for i in range(ny):
    for j in range(nx):
        y = speed_arr[:, i, j]
        vv = ~np.isnan(y)
        if vv.sum() > 30:
            s, _, _, _, _ = linregress(time_years[vv], y[vv])
            trend_speed[i, j] = s * 10
speed_clim = np.nanmean(speed_arr, axis=0)

# ── 绘图 ──
early = ugos_arr[:120].mean(axis=(0, 2))
late = ugos_arr[-120:].mean(axis=(0, 2))

lon_display = lon - 360 if lon.max() > 180 else lon

fig = plt.figure(figsize=(10, 14))
gs_grid = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.35)

ax = fig.add_subplot(gs_grid[0])
lon2d, lat2d = np.meshgrid(lon_display, lat)
cs = ax.contourf(lon2d, lat2d, trend_speed, levels=np.linspace(-0.08, 0.08, 33), cmap='RdBu_r', extend='both')
cb = plt.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('Speed trend (m/s per decade)')
ax.contour(lon2d, lat2d, speed_clim, levels=[0.2, 0.3, 0.4, 0.5], colors='k', linewidths=1)
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
ax.set_title('(a) Gulf Stream Geostrophic Speed Trend (1993–2024)', fontweight='bold')

ax = fig.add_subplot(gs_grid[1])
dates = pd.to_datetime(time_arr)
sm_v = pd.Series(axis_gs).rolling(12, center=True, min_periods=6).mean().values
ax.plot(dates, sm_v, 'r-', linewidth=2.5, label=f'Velocity max: {sl_v*10:+.2f}°/dec ({fmt_p(p_v)})')
ax.plot(dates, sl_v * time_years + ic_v, 'r--', linewidth=1.5)
ax.axvline(pd.Timestamp("2022-01-01"), color='gray', linewidth=1, linestyle=':', alpha=0.5)
ax.set_ylabel('GS Axis Latitude (°N)')
ax.set_title('(b) Gulf Stream: Velocity Maximum (1993–2024)', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax = fig.add_subplot(gs_grid[2])
ax.plot(lat, early, 'b-', linewidth=2.5, label='1993–2002 mean')
ax.plot(lat, late, 'r-', linewidth=2.5, label='2015–2024 mean')
ax.fill_between(lat, early, late, where=late > early, color='red', alpha=0.15, label='Acceleration')
ax.fill_between(lat, early, late, where=late < early, color='blue', alpha=0.15, label='Deceleration')
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Latitude (°N)')
ax.set_ylabel('Eastward Velocity (m/s)')
ax.set_title('(c) Meridional Velocity Profile: Early vs Late Decade', fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.savefig(FIG / "fig2_gs_velocity_v5.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "fig2_gs_velocity_v5.png", dpi=300, bbox_inches='tight')
plt.close()

shutil.copy(FIG / "fig2_gs_velocity_v5.pdf", MAN / "fig7.pdf")
shutil.copy(FIG / "fig2_gs_velocity_v5.png", MAN / "fig7.png")
print("已拷贝到 manuscript/fig7.{pdf,png}")
