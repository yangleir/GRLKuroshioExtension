"""Remake Fig 1 (KE velocity) and Fig 2 (Gulf Stream) with:
- Equal panel widths
- Larger fonts (≥12pt labels, ≥10pt ticks)
- Publication quality for Nature Communications
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import pandas as pd

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'font.family': 'sans-serif',
})

OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")
ROOT = Path("/Volumes/Backup Plus/ssh/dataset-duacs-rep-global-merged-allsat-phy-l4")

# ── 数据加载（复用 P5 的逻辑）──
print("收集日数据...")
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

def load_region(files, lat_range, lon_range):
    ugos_list, sla_list, speed_list, times = [], [], [], []
    for i, fp in enumerate(files):
        ds = xr.open_dataset(fp)
        lon = ds.longitude.values
        if lon.max() > 180:
            ke = ds.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))
        else:
            ke = ds.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))
        if len(ke.longitude) == 0:
            ds.close()
            continue
        u = ke['ugos'].isel(time=0).values
        v = ke['vgos'].isel(time=0).values
        s = ke['sla'].isel(time=0).values
        ugos_list.append(u)
        speed_list.append(np.sqrt(u**2 + v**2))
        sla_list.append(s)
        times.append(ke.time.values[0])
        ds.close()
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(files)}")
    lat = ke.latitude.values
    lon_arr = ke.longitude.values
    return (np.array(ugos_list), np.array(speed_list), np.array(sla_list),
            np.array(times), lat, lon_arr)

def track_vel_max(ugos_arr, lat, dlat):
    axis = np.full(ugos_arr.shape[0], np.nan)
    for t in range(len(axis)):
        frame = ugos_arr[t]
        lats = []
        for j in range(frame.shape[1]):
            col = frame[:, j]
            valid = ~np.isnan(col)
            if valid.sum() < 5: continue
            col_i = np.interp(np.arange(len(col)), np.where(valid)[0], col[valid])
            col_s = gaussian_filter1d(col_i, sigma=2)
            mask = (lat >= 32) & (lat <= 40) if lat.max() > 40 else (lat >= lat.min()+2) & (lat <= lat.max()-2)
            col_m = np.where(mask, col_s, -np.inf)
            idx = np.argmax(col_m)
            if col_m[idx] > 0.03:
                lats.append(lat[idx])
        if len(lats) >= frame.shape[1] * 0.3:
            axis[t] = np.median(lats)
    return axis

def track_sla_grad(sla_arr, lat, dlat):
    axis = np.full(sla_arr.shape[0], np.nan)
    for t in range(len(axis)):
        frame = sla_arr[t]
        lats = []
        for j in range(frame.shape[1]):
            col = frame[:, j]
            valid = ~np.isnan(col)
            if valid.sum() < 5: continue
            col_i = np.interp(np.arange(len(col)), np.where(valid)[0], col[valid])
            col_s = gaussian_filter1d(col_i, sigma=2)
            grad = np.gradient(col_s, dlat)
            mask = (lat >= 32) & (lat <= 40) if lat.max() > 40 else (lat >= lat.min()+2) & (lat <= lat.max()-2)
            grad_m = np.where(mask, grad, -np.inf)
            idx = np.argmax(grad_m)
            if grad_m[idx] > 0:
                lats.append(lat[idx])
        if len(lats) >= frame.shape[1] * 0.3:
            axis[t] = np.median(lats)
    return axis

def compute_speed_trend(speed_arr, time_years, lat, lon_arr):
    ny, nx = len(lat), len(lon_arr)
    trend = np.full((ny, nx), np.nan)
    for i in range(ny):
        for j in range(nx):
            y = speed_arr[:, i, j]
            valid = ~np.isnan(y)
            if valid.sum() > 30:
                sl, _, _, _, _ = linregress(time_years[valid], y[valid])
                trend[i, j] = sl * 10
    return trend

def plot_three_panel(ugos_arr, speed_arr, sla_arr, time_arr, lat, lon_arr,
                     vel_axis, sla_axis, time_years, region_name, output_name):
    """Three equal-width panels with large fonts"""
    trend_speed = compute_speed_trend(speed_arr, time_years, lat, lon_arr)
    speed_clim = np.nanmean(speed_arr, axis=0)

    valid_v = ~np.isnan(vel_axis)
    valid_s = ~np.isnan(sla_axis)
    sl_v, ic_v, _, p_v, _ = linregress(time_years[valid_v], vel_axis[valid_v])
    sl_s, ic_s, _, p_s, _ = linregress(time_years[valid_s], sla_axis[valid_s])

    early = ugos_arr[:120].mean(axis=(0, 2))
    late = ugos_arr[-120:].mean(axis=(0, 2))

    fig = plt.figure(figsize=(10, 14))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.35)

    # (a) Speed trend map
    ax = fig.add_subplot(gs[0])
    lon2d, lat2d = np.meshgrid(lon_arr, lat)
    clevels = np.linspace(-0.08, 0.08, 33)
    cs = ax.contourf(lon2d, lat2d, trend_speed, levels=clevels, cmap='RdBu_r', extend='both')
    cb = plt.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('Speed trend (m/s per decade)', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    ax.contour(lon2d, lat2d, speed_clim, levels=[0.2, 0.3, 0.4, 0.5], colors='k', linewidths=1)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(f'(a) {region_name} Geostrophic Speed Trend (1993–2021)', fontweight='bold')

    # (b) Velocity max vs SLA gradient
    ax = fig.add_subplot(gs[1])
    dates = pd.to_datetime(time_arr)
    smooth_v = pd.Series(vel_axis).rolling(12, center=True, min_periods=6).mean().values
    smooth_s = pd.Series(sla_axis).rolling(12, center=True, min_periods=6).mean().values
    ax.plot(dates, smooth_v, 'r-', linewidth=2.5,
            label=f'Velocity max: {sl_v*10:+.2f}°/dec (p={p_v:.3f})')
    ax.plot(dates, sl_v * time_years + ic_v, 'r--', linewidth=1.5)
    ax.plot(dates, smooth_s, 'b-', linewidth=2.5,
            label=f'SLA gradient: {sl_s*10:+.2f}°/dec (p={p_s:.3f})')
    ax.plot(dates, sl_s * time_years + ic_s, 'b--', linewidth=1.5)
    ax.set_ylabel('KE Axis Latitude (°N)')
    ax.set_title(f'(b) {region_name}: Velocity Maximum vs SLA Gradient Maximum', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # (c) Meridional velocity profile
    ax = fig.add_subplot(gs[2])
    ax.plot(lat, early, 'b-', linewidth=2.5, label='1993–2002 mean')
    ax.plot(lat, late, 'r-', linewidth=2.5, label='2012–2021 mean')
    ax.fill_between(lat, early, late, where=late > early, color='red', alpha=0.15, label='Acceleration')
    ax.fill_between(lat, early, late, where=late < early, color='blue', alpha=0.15, label='Deceleration')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Latitude (°N)')
    ax.set_ylabel('Eastward Velocity (m/s)')
    ax.set_title(f'(c) Meridional Zonal Velocity Profile: Early vs Late Decade', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.savefig(FIG / f"{output_name}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(FIG / f"{output_name}.png", dpi=300, bbox_inches='tight')
    print(f"保存: {FIG / output_name}.pdf/png")
    plt.close()

# ── KE (Fig 1) ──
print("\n=== Fig 1: Kuroshio Extension ===")
ugos_ke, speed_ke, sla_ke, time_ke, lat_ke, lon_ke = load_region(files, (30, 42), (142, 170))
dlat_ke = np.abs(np.diff(lat_ke).mean())
ty_ke = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_ke])
vel_ke = track_vel_max(ugos_ke, lat_ke, dlat_ke)
sla_axis_ke = track_sla_grad(sla_ke, lat_ke, dlat_ke)
plot_three_panel(ugos_ke, speed_ke, sla_ke, time_ke, lat_ke, lon_ke,
                 vel_ke, sla_axis_ke, ty_ke, 'Kuroshio Extension', 'fig1_ke_velocity_v2')

# ── Gulf Stream (Fig 2) ──
print("\n=== Fig 2: Gulf Stream ===")
ugos_gs, speed_gs, sla_gs, time_gs, lat_gs, lon_gs = load_region(files, (33, 47), (-75, -45))
if len(lon_gs) == 0:
    # 经度 0-360 格式
    ugos_gs, speed_gs, sla_gs, time_gs, lat_gs, lon_gs = load_region(files, (33, 47), (285, 315))
dlat_gs = np.abs(np.diff(lat_gs).mean())
ty_gs = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_gs])
vel_gs = track_vel_max(ugos_gs, lat_gs, dlat_gs)
sla_axis_gs = track_sla_grad(sla_gs, lat_gs, dlat_gs)
plot_three_panel(ugos_gs, speed_gs, sla_gs, time_gs, lat_gs, lon_gs,
                 vel_gs, sla_axis_gs, ty_gs, 'Gulf Stream', 'fig2_gs_velocity_v2')

# Copy to manuscript
import shutil
shutil.copy(FIG / "fig1_ke_velocity_v2.pdf", "/Users/zhulin/aitest/黑潮延伸体/manuscript/fig6.pdf")
shutil.copy(FIG / "fig2_gs_velocity_v2.pdf", "/Users/zhulin/aitest/黑潮延伸体/manuscript/fig7.pdf")
print("\n已复制到 manuscript/fig6.pdf, fig7.pdf")
