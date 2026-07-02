"""P2b: Method comparison — why different methods give contradictory "northward shift"
- Remove regional SLA trend, then re-track KE axis
- Compare: gradient method vs SLA-weighted method vs detrended versions
- EEMD decomposition of KE axis time series
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

# ── 1. 读取数据 ──
data_path = "/Volumes/Backup Plus/ssh/cmems_sla_monthly_global_0.125deg_1993_2025.nc"
ds = xr.open_dataset(data_path)
lon = ds.longitude.values
if lon.min() < 0:
    ds = ds.assign_coords(longitude=(ds.longitude % 360))
    ds = ds.sortby('longitude')

sla_ke = ds['sla'].sel(longitude=slice(142, 170), latitude=slice(30, 42))
time_vals = sla_ke.time.values
lat = sla_ke.latitude.values
lon_ke = sla_ke.longitude.values
dlat = np.abs(np.diff(lat).mean())

time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_vals])

print(f"数据: {len(time_vals)} 月, {len(lat)}x{len(lon_ke)} 网格")

# ── 2. 去除空间趋势场 ──
print("计算逐格点 SLA 线性趋势...")
sla_vals = sla_ke.values  # (time, lat, lon)
trend_field = np.full((len(lat), len(lon_ke)), np.nan)
intercept_field = np.full_like(trend_field, np.nan)

for i in range(len(lat)):
    for j in range(len(lon_ke)):
        y = sla_vals[:, i, j]
        valid = ~np.isnan(y)
        if valid.sum() > 30:
            sl, ic, _, _, _ = linregress(time_years[valid], y[valid])
            trend_field[i, j] = sl
            intercept_field[i, j] = ic

# 去趋势 SLA
sla_detrended = np.full_like(sla_vals, np.nan)
for t in range(len(time_vals)):
    sla_detrended[t] = sla_vals[t] - (trend_field * time_years[t] + intercept_field)

print(f"趋势场范围: {np.nanmin(trend_field)*1000:.2f} ~ {np.nanmax(trend_field)*1000:.2f} mm/yr")

# ── 3. 方法 A: SLA 梯度极大法（原始 + 去趋势） ──
def track_ke_axis_gradient(sla_data, lat_arr, dlat_val, lat_search=(32, 40)):
    nt = sla_data.shape[0]
    ke_lat = np.full(nt, np.nan)
    for t in range(nt):
        sla_t = sla_data[t]
        jet_lats = []
        for j in range(sla_t.shape[1]):
            col = sla_t[:, j]
            if np.isnan(col).sum() > len(col) * 0.3:
                continue
            col_s = gaussian_filter1d(np.nan_to_num(col, nan=np.nanmean(col)), sigma=2)
            grad = np.gradient(col_s, dlat_val)
            mask = (lat_arr >= lat_search[0]) & (lat_arr <= lat_search[1])
            grad_m = grad.copy()
            grad_m[~mask] = -np.inf
            idx = np.argmax(grad_m)
            if grad_m[idx] > 0:
                jet_lats.append(lat_arr[idx])
        if len(jet_lats) >= sla_t.shape[1] * 0.3:
            ke_lat[t] = np.median(jet_lats)
    return ke_lat

print("\n追踪 KE 轴（4种方案）...")
ke_grad_orig = track_ke_axis_gradient(sla_vals, lat, dlat)
ke_grad_detrended = track_ke_axis_gradient(sla_detrended, lat, dlat)

# ── 4. 方法 B: SLA 加权纬度（原始 + 去趋势） ──
def sla_weighted_lat(sla_data, lat_arr, lat_range=(32, 40)):
    mask = (lat_arr >= lat_range[0]) & (lat_arr <= lat_range[1])
    sla_sub = sla_data[:, mask, :]
    lat_sub = lat_arr[mask]
    nt = sla_sub.shape[0]
    wlat = np.full(nt, np.nan)
    for t in range(nt):
        s = sla_sub[t]
        s_pos = np.where(s > 0, s, 0)
        total = np.nansum(s_pos)
        if total > 0:
            wlat[t] = np.nansum(s_pos * lat_sub[:, None]) / total
    return wlat

ke_wt_orig = sla_weighted_lat(sla_vals, lat)
ke_wt_detrended = sla_weighted_lat(sla_detrended, lat)

# ── 5. 各序列趋势 ──
print("\n=== 趋势对比 ===")
series = {
    'Gradient (original)': ke_grad_orig,
    'Gradient (detrended)': ke_grad_detrended,
    'Weighted (original)': ke_wt_orig,
    'Weighted (detrended)': ke_wt_detrended,
}
results = {}
for name, vals in series.items():
    valid = ~np.isnan(vals)
    if valid.sum() < 30:
        continue
    sl, ic, r, p, se = linregress(time_years[valid], vals[valid])
    print(f"  {name}: {sl*10:+.4f}°/dec, p={p:.4f} {'★' if p < 0.05 else ''}")
    results[name] = {'trend_per_decade': round(sl * 10, 5), 'p_value': round(p, 5)}

# ── 6. 绘图 ──
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
dates = pd.to_datetime(time_vals)

# Panel (a): 原始数据的两种方法对比
ax = axes[0]
ax.plot(dates, pd.Series(ke_grad_orig).rolling(12, center=True, min_periods=6).mean(),
        'b-', linewidth=2, label='Gradient method (12-mo mean)')
ax.plot(dates, pd.Series(ke_wt_orig).rolling(12, center=True, min_periods=6).mean(),
        'r-', linewidth=2, label='SLA-weighted method (12-mo mean)')

for name, vals, color in [('Gradient', ke_grad_orig, 'b'), ('Weighted', ke_wt_orig, 'r')]:
    valid = ~np.isnan(vals)
    sl, ic, _, p, _ = linregress(time_years[valid], vals[valid])
    ax.plot(dates, sl * time_years + ic, f'{color}--', linewidth=1,
            label=f'{name} trend: {sl*10:+.3f}°/dec (p={p:.3f})')

ax.set_ylabel('Latitude (°N)')
ax.set_title('(a) Original SLA: Two Methods Give Contradictory Trends')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel (b): 去区域趋势后
ax = axes[1]
ax.plot(dates, pd.Series(ke_grad_detrended).rolling(12, center=True, min_periods=6).mean(),
        'b-', linewidth=2, label='Gradient (detrended, 12-mo)')
ax.plot(dates, pd.Series(ke_wt_detrended).rolling(12, center=True, min_periods=6).mean(),
        'r-', linewidth=2, label='Weighted (detrended, 12-mo)')

for name, vals, color in [('Gradient', ke_grad_detrended, 'b'), ('Weighted', ke_wt_detrended, 'r')]:
    valid = ~np.isnan(vals)
    sl, ic, _, p, _ = linregress(time_years[valid], vals[valid])
    ax.plot(dates, sl * time_years + ic, f'{color}--', linewidth=1,
            label=f'{name} trend: {sl*10:+.3f}°/dec (p={p:.3f})')

ax.set_ylabel('Latitude (°N)')
ax.set_title('(b) After Removing Regional SLA Trend: Do Methods Converge?')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel (c): 差值（Weighted - Gradient）与 N-S SLA 差
ax = axes[2]
diff_orig = ke_wt_orig - ke_grad_orig
diff_smooth = pd.Series(diff_orig).rolling(12, center=True, min_periods=6).mean().values

sla_n = ds['sla'].sel(longitude=slice(142, 170), latitude=slice(35, 40)).mean(dim=['latitude', 'longitude']).values
sla_s = ds['sla'].sel(longitude=slice(142, 170), latitude=slice(30, 35)).mean(dim=['latitude', 'longitude']).values
ns_diff = sla_n - sla_s
ns_smooth = pd.Series(ns_diff).rolling(12, center=True, min_periods=6).mean().values

ax2 = ax.twinx()
ax.plot(dates, diff_smooth, 'k-', linewidth=2, label='Method diff (Weighted - Gradient)')
ax2.plot(dates, ns_smooth * 100, 'r-', linewidth=2, alpha=0.7, label='N-S SLA diff (cm)')
ax.set_ylabel('Latitude difference (°)')
ax2.set_ylabel('N-S SLA diff (cm)', color='r')
ax.set_title('(c) Method Disagreement Tracks N-S SLA Differential Rise')
ax.set_xlabel('Year')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

for a in axes:
    a.xaxis.set_major_locator(mdates.YearLocator(5))
    a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    a.set_xlim(pd.Timestamp("1993-01-01"), pd.Timestamp("2026-01-01"))

plt.tight_layout()
plt.savefig(FIG / "fig2_method_comparison.png", dpi=300, bbox_inches='tight')
print(f"\n图已保存: {FIG / 'fig2_method_comparison.png'}")
plt.close()

# ── 7. 保存结果 ──
with open(OUT / "method_comparison_stats.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
