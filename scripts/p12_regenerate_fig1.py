"""Regenerate Fig 1 (KE) with unified MY+NRT data (1993-2024)"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy.stats import linregress, t as t_dist
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import pandas as pd

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})

FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")
ROOT = Path("/Volumes/Backup Plus/ssh/dataset-duacs-rep-global-merged-allsat-phy-l4")

# ── 收集 MY 文件 ──
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

# ── 读 MY KE ──
ugos_list, speed_list, sla_list, times = [], [], [], []
for i, fp in enumerate(files):
    ds = xr.open_dataset(fp)
    ke = ds.sel(latitude=slice(30, 42), longitude=slice(142, 170))
    if len(ke.longitude) == 0: ds.close(); continue
    u = ke['ugos'].isel(time=0).values
    v = ke['vgos'].isel(time=0).values
    ugos_list.append(u)
    speed_list.append(np.sqrt(u**2 + v**2))
    sla_list.append(ke['sla'].isel(time=0).values)
    times.append(ke.time.values[0])
    ds.close()

lat = ke.latitude.values
lon_ke = ke.longitude.values

# ── 读 NRT KE ──
nrt = xr.open_dataset("/Volumes/Backup Plus/ssh/cmems_ugos_vgos_nrt_daily_KE_2022_2025.nc")
nrt_ke = nrt.sel(latitude=slice(30, 42), longitude=slice(142, 170))
nrt_ugos_m = nrt_ke['ugos'].resample(time='MS').mean()
nrt_vgos_m = nrt_ke['vgos'].resample(time='MS').mean()
for t in range(len(nrt_ugos_m.time)):
    u = nrt_ugos_m.isel(time=t).values
    v = nrt_vgos_m.isel(time=t).values
    ugos_list.append(u)
    speed_list.append(np.sqrt(u**2 + v**2))
    sla_list.append(np.full_like(u, np.nan))
    times.append(nrt_ugos_m.time.values[t])
nrt.close()

ugos_arr = np.array(ugos_list)
speed_arr = np.array(speed_list)
sla_arr = np.array(sla_list)
time_arr = np.array(times)
dlat = np.abs(np.diff(lat).mean())
time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_arr])

print(f"KE 统一: {ugos_arr.shape}, {str(time_arr[0])[:10]} → {str(time_arr[-1])[:10]}")

# ── 计算速度趋势（用全部 1993-2024 数据）──
ny, nx = len(lat), len(lon_ke)
trend_speed = np.full((ny, nx), np.nan)
for i in range(ny):
    for j in range(nx):
        y = speed_arr[:, i, j]
        valid = ~np.isnan(y)
        if valid.sum() > 30:
            sl, _, _, _, _ = linregress(time_years[valid], y[valid])
            trend_speed[i, j] = sl * 10
speed_clim = np.nanmean(speed_arr, axis=0)

# ── 追踪 ──
def track(data, lat_arr, method='max'):
    nt = data.shape[0]
    axis = np.full(nt, np.nan)
    for t in range(nt):
        frame = data[t]; lats = []
        for j in range(frame.shape[1]):
            col = frame[:, j]; valid = ~np.isnan(col)
            if valid.sum() < 5: continue
            col_i = np.interp(np.arange(len(col)), np.where(valid)[0], col[valid])
            col_s = gaussian_filter1d(col_i, sigma=2)
            mask = (lat_arr >= 32) & (lat_arr <= 40)
            if method == 'max':
                cm = np.where(mask, col_s, -np.inf); idx = np.argmax(cm)
                if cm[idx] > 0.03: lats.append(lat_arr[idx])
            else:
                grad = np.gradient(col_s, dlat)
                gm = np.where(mask, grad, -np.inf); idx = np.argmax(gm)
                if gm[idx] > 0: lats.append(lat_arr[idx])
        if len(lats) >= frame.shape[1]*0.3: axis[t] = np.median(lats)
    return axis

vel_axis = track(ugos_arr, lat, 'max')
sla_axis = track(sla_arr, lat, 'grad')

def bretherton_trend(x, y):
    """OLS trend with Bretherton et al. (1999) autocorrelation-corrected p-value (same as p10)."""
    sl, ic, r, p_raw, se = linregress(x, y)
    residuals = y - (sl * x + ic)
    N = len(y)
    r1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    N_eff = max(N * (1 - r1) / (1 + r1), 3)
    se_corrected = se * np.sqrt(N / N_eff)
    t_stat = sl / se_corrected
    p_corrected = 2 * (1 - t_dist.cdf(abs(t_stat), df=max(N_eff - 2, 1)))
    return sl, ic, p_corrected

def fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.2f}"

valid_v = ~np.isnan(vel_axis); valid_s = ~np.isnan(sla_axis)
sl_v, ic_v, p_v = bretherton_trend(time_years[valid_v], vel_axis[valid_v])
sl_s, ic_s, p_s = bretherton_trend(time_years[valid_s], sla_axis[valid_s])
print(f"Vel max: {sl_v*10:+.2f}°/dec ({fmt_p(p_v)})")
print(f"SLA grad: {sl_s*10:+.2f}°/dec ({fmt_p(p_s)})")

# ── 绘图 ──
early = ugos_arr[:120].mean(axis=(0, 2))
late = ugos_arr[-120:].mean(axis=(0, 2))

fig = plt.figure(figsize=(10, 14))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.35)

ax = fig.add_subplot(gs[0])
lon2d, lat2d = np.meshgrid(lon_ke, lat)
cs = ax.contourf(lon2d, lat2d, trend_speed, levels=np.linspace(-0.08, 0.08, 33), cmap='RdBu_r', extend='both')
cb = plt.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('Speed trend (m/s per decade)')
ax.contour(lon2d, lat2d, speed_clim, levels=[0.2, 0.3, 0.4, 0.5], colors='k', linewidths=1)
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
ax.set_title('(a) KE Geostrophic Speed Trend (1993–2024)', fontweight='bold')

ax = fig.add_subplot(gs[1])
dates = pd.to_datetime(time_arr)
sm_v = pd.Series(vel_axis).rolling(12, center=True, min_periods=6).mean().values
sm_s = pd.Series(sla_axis).rolling(12, center=True, min_periods=6).mean().values
ax.plot(dates, sm_v, 'r-', linewidth=2.5, label=f'Velocity max: {sl_v*10:+.2f}°/dec ({fmt_p(p_v)})')
ax.plot(dates, sl_v*time_years+ic_v, 'r--', linewidth=1.5)
ax.plot(dates, sm_s, 'b-', linewidth=2.5, label=f'SLA gradient: {sl_s*10:+.2f}°/dec ({fmt_p(p_s)})')
ax.plot(dates, sl_s*time_years+ic_s, 'b--', linewidth=1.5)
ax.axvline(pd.Timestamp("2022-01-01"), color='gray', linewidth=1, linestyle=':', alpha=0.5)
ax.set_ylabel('KE Axis Latitude (°N)')
ax.set_title('(b) KE: Velocity Maximum vs SLA Gradient Maximum (1993–2024)', fontweight='bold')
ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(5)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax = fig.add_subplot(gs[2])
ax.plot(lat, early, 'b-', linewidth=2.5, label='1993–2002 mean')
ax.plot(lat, late, 'r-', linewidth=2.5, label='2015–2024 mean')
ax.fill_between(lat, early, late, where=late>early, color='red', alpha=0.15, label='Acceleration')
ax.fill_between(lat, early, late, where=late<early, color='blue', alpha=0.15, label='Deceleration')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('Latitude (°N)'); ax.set_ylabel('Eastward Velocity (m/s)')
ax.set_title('(c) Meridional Velocity Profile: Early vs Late Decade', fontweight='bold')
ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

plt.savefig(FIG / "fig1_ke_velocity_v3.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "fig1_ke_velocity_v3.png", dpi=300, bbox_inches='tight')
print(f"保存: fig1_ke_velocity_v3")
plt.close()

import shutil
shutil.copy(FIG / "fig1_ke_velocity_v3.pdf", "/Users/zhulin/aitest/黑潮延伸体/manuscript/fig6.pdf")
print("已复制到 manuscript/fig6.pdf")
