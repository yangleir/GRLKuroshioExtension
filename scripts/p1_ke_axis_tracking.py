"""P0+P1: KE axis position tracking from monthly SLA/ADT data
- Method A: SSH maximum gradient (∂SLA/∂y) to locate jet axis
- Produces: KE axis latitude time series (1993-2025)
- Note: Uses monthly global data; switch path when full download completes
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
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# ── 1. 数据路径 ──
# 优先使用完整的 1993-2025 数据（移动硬盘）；fallback 到 data/ 里的子集
full_path = Path("/Volumes/Backup Plus/ssh/cmems_sla_monthly_global_0.125deg_1993_2025.nc")
subset_path = Path("/Users/zhulin/aitest/黑潮延伸体/data/cmems_sla_monthly_global_2022_2025.nc")

if full_path.exists():
    data_path = full_path
    print(f"使用完整数据: {data_path}")
else:
    data_path = subset_path
    print(f"完整数据未就绪，使用子集: {data_path}")

ds = xr.open_dataset(data_path)
print(f"时间: {str(ds.time.values[0])[:10]} → {str(ds.time.values[-1])[:10]}, {len(ds.time)} 步")
print(f"网格: {len(ds.latitude)}×{len(ds.longitude)}")

# ── 2. 提取 KE 区域 ──
# 经度统一到 0-360 再切片（如果是 -180~180 格式）
lon = ds.longitude.values
if lon.min() < 0:
    ds = ds.assign_coords(longitude=(ds.longitude % 360))
    ds = ds.sortby('longitude')

# KE 核心区域
lon_min, lon_max = 142, 170
lat_min, lat_max = 30, 42

sla_ke = ds['sla'].sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
print(f"\nKE 区域: {lon_min}-{lon_max}°E, {lat_min}-{lat_max}°N")
print(f"KE 数据形状: {sla_ke.shape}")

lat = sla_ke.latitude.values
lon_ke = sla_ke.longitude.values

# ── 3. KE 轴追踪：SLA 经向梯度极大法 ──
print("\n计算 KE 轴位置（SLA 经向梯度极大法）...")
dlat = np.abs(np.diff(lat).mean())

ke_axis_lat = np.full(len(sla_ke.time), np.nan)

for t in range(len(sla_ke.time)):
    sla_t = sla_ke.isel(time=t).values  # (lat, lon)

    jet_lats = []
    for j in range(sla_t.shape[1]):
        col = sla_t[:, j]
        if np.isnan(col).sum() > len(col) * 0.3:
            continue
        col_smooth = gaussian_filter1d(np.nan_to_num(col, nan=np.nanmean(col)), sigma=2)
        grad = np.gradient(col_smooth, dlat)
        # 在 32-40°N 范围内找梯度极大值（KE 射流位置）
        mask = (lat >= 32) & (lat <= 40)
        grad_masked = grad.copy()
        grad_masked[~mask] = -np.inf
        idx_max = np.argmax(grad_masked)
        if grad_masked[idx_max] > 0:
            jet_lats.append(lat[idx_max])

    if len(jet_lats) >= len(lon_ke) * 0.3:
        ke_axis_lat[t] = np.median(jet_lats)

valid = ~np.isnan(ke_axis_lat)
print(f"有效月数: {valid.sum()}/{len(ke_axis_lat)}")
print(f"KE 轴纬度范围: {np.nanmin(ke_axis_lat):.2f} ~ {np.nanmax(ke_axis_lat):.2f}°N")
print(f"KE 轴纬度均值: {np.nanmean(ke_axis_lat):.2f}°N")

# ── 4. 趋势分析 ──
time_vals = sla_ke.time.values
time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25
                        for t in time_vals])

slope, intercept, r, p, se = linregress(time_years[valid], ke_axis_lat[valid])
trend_deg_per_decade = slope * 10

print(f"\nKE 轴线性趋势: {trend_deg_per_decade:.3f}°/decade")
print(f"p 值: {p:.6f} ({'显著' if p < 0.05 else '不显著'})")
print(f"R²: {r**2:.4f}")

# ── 5. 保存结果 ──
ke_axis_ds = xr.Dataset({
    'ke_axis_latitude': ('time', ke_axis_lat),
}, coords={'time': time_vals})
ke_axis_ds.attrs['method'] = 'SLA meridional gradient maximum (Gaussian smoothed, sigma=2)'
ke_axis_ds.attrs['region'] = f'{lon_min}-{lon_max}E, {lat_min}-{lat_max}N'
ke_axis_ds.attrs['trend_deg_per_decade'] = trend_deg_per_decade
ke_axis_ds.attrs['trend_p_value'] = p
ke_axis_ds.to_netcdf(OUT / 'ke_axis_position.nc')
print(f"\n保存: {OUT / 'ke_axis_position.nc'}")

# ── 6. 绘图：KE 轴纬度时间序列 ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Panel (a): KE 轴纬度时间序列
ax = axes[0]
dates = pd.to_datetime(time_vals)
ax.plot(dates, ke_axis_lat, 'k-', linewidth=0.8, alpha=0.6, label='Monthly')

# 12-month running mean
if valid.sum() > 12:
    ke_smooth = pd.Series(ke_axis_lat).rolling(12, center=True, min_periods=6).mean().values
    ax.plot(dates, ke_smooth, 'b-', linewidth=2, label='12-month running mean')

# 趋势线
trend_line = slope * time_years + intercept
ax.plot(dates, trend_line, 'r--', linewidth=1.5,
        label=f'Trend: {trend_deg_per_decade:+.3f}°/decade (p={p:.4f})')

ax.axvline(pd.Timestamp("2017-08-01"), color='green', linewidth=1, linestyle=':', alpha=0.7)
ax.text(pd.Timestamp("2017-10-01"), ax.get_ylim()[1] * 0.98, 'LM 2017', color='green', fontsize=8, va='top')

ax.set_ylabel('KE Jet Axis Latitude (°N)')
ax.set_title('(a) Kuroshio Extension Jet Axis Position')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (b): 年均 KE 轴位置
ax = axes[1]
yearly = ke_axis_ds['ke_axis_latitude'].resample(time='YE').mean()
years_dt = pd.to_datetime(yearly.time.values)
yearly_vals = yearly.values

colors = ['#d73027' if v > np.nanmean(yearly_vals) else '#4575b4' for v in yearly_vals]
ax.bar(years_dt, yearly_vals - np.nanmean(yearly_vals), color=colors, width=300, alpha=0.7)
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('Anomaly from Mean (°)')
ax.set_title(f'(b) Annual Mean KE Axis Latitude Anomaly (mean={np.nanmean(yearly_vals):.2f}°N)')
ax.grid(True, alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(FIG / "fig2_ke_axis_timeseries.png", dpi=300, bbox_inches='tight')
print(f"图已保存: {FIG / 'fig2_ke_axis_timeseries.png'}")
plt.close()

# ── 7. 统计摘要 ──
stats = {
    "data_source": str(data_path),
    "time_range": f"{str(time_vals[0])[:10]} to {str(time_vals[-1])[:10]}",
    "valid_months": int(valid.sum()),
    "total_months": len(ke_axis_lat),
    "mean_lat_N": round(float(np.nanmean(ke_axis_lat)), 3),
    "std_lat": round(float(np.nanstd(ke_axis_lat)), 3),
    "trend_deg_per_decade": round(trend_deg_per_decade, 4),
    "trend_p_value": round(p, 6),
    "trend_r_squared": round(r**2, 4),
}
with open(OUT / "ke_axis_statistics.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
