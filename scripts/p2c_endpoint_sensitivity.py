"""P2c: Endpoint sensitivity + unified method reanalysis
- Re-run KE axis tracking on monthly data with improved method (from P1b)
- Sliding endpoint sensitivity: trend vs cutoff year
- Multi-method trend uncertainty quantification
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

# ── 1. 读取月均数据 ──
data_path = "/Volumes/Backup Plus/ssh/cmems_sla_monthly_global_0.125deg_1993_2025.nc"
ds = xr.open_dataset(data_path)
lon = ds.longitude.values
if lon.min() < 0:
    ds = ds.assign_coords(longitude=(ds.longitude % 360))
    ds = ds.sortby('longitude')

sla_ke = ds['sla'].sel(longitude=slice(142, 170), latitude=slice(30, 42))
time_vals = sla_ke.time.values
lat = sla_ke.latitude.values
dlat = np.abs(np.diff(lat).mean())
sla_arr = sla_ke.values

# ── 2. 改进的梯度法（与 P1b 一致：np.interp + parabolic） ──
def track_axis_improved(data, lat_arr, dlat_val, lat_search=(32, 40)):
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
            col_interp = np.interp(np.arange(len(col)), np.where(valid_mask)[0], col[valid_mask])
            col_s = gaussian_filter1d(col_interp, sigma=2)
            grad = np.gradient(col_s, dlat_val)
            mask = (lat_arr >= lat_search[0]) & (lat_arr <= lat_search[1])
            grad_m = grad.copy()
            grad_m[~mask] = -np.inf
            idx = np.argmax(grad_m)
            if grad_m[idx] > 0 and 0 < idx < len(grad) - 1:
                y0, y1, y2 = grad[idx-1], grad[idx], grad[idx+1]
                denom = 2 * (2 * y1 - y0 - y2)
                if abs(denom) > 1e-10:
                    offset = (y0 - y2) / denom
                    jet_lats.append(lat_arr[idx] + offset * dlat_val)
                else:
                    jet_lats.append(lat_arr[idx])
        if len(jet_lats) >= frame.shape[1] * 0.3:
            axis_lat[t] = np.median(jet_lats)
    return axis_lat

print("重跑 KE 轴追踪（改进方法，月均 0.125° 数据）...")
ke_axis = track_axis_improved(sla_arr, lat, dlat)
valid = ~np.isnan(ke_axis)
print(f"有效月数: {valid.sum()}/{len(ke_axis)}")

time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_vals])
sl_full, ic_full, r_full, p_full, se_full = linregress(time_years[valid], ke_axis[valid])
print(f"1993-2025 全段: {sl_full*10:+.4f}°/dec, p={p_full:.4f}")

# ── 3. 端点敏感性分析 ──
print("\n=== 端点敏感性分析 ===")
cutoff_years = list(range(2010, 2026))
trends = []
p_vals = []
ci_lower = []
ci_upper = []

for cy in cutoff_years:
    cutoff_date = pd.Timestamp(f"{cy}-12-31")
    mask_time = pd.to_datetime(time_vals) <= cutoff_date
    mask = valid & mask_time
    if mask.sum() < 60:
        trends.append(np.nan)
        p_vals.append(np.nan)
        ci_lower.append(np.nan)
        ci_upper.append(np.nan)
        continue
    sl, ic, r, p, se = linregress(time_years[mask], ke_axis[mask])
    trends.append(sl * 10)
    p_vals.append(p)
    ci_lower.append((sl - 1.96 * se) * 10)
    ci_upper.append((sl + 1.96 * se) * 10)
    sig = '★' if p < 0.05 else ''
    print(f"  1993-{cy}: {sl*10:+.4f}°/dec, p={p:.4f} {sig}")

trends = np.array(trends)
p_vals = np.array(p_vals)
ci_lower = np.array(ci_lower)
ci_upper = np.array(ci_upper)

# ── 4. 起点敏感性 ──
print("\n=== 起点敏感性分析 ===")
start_years = list(range(1993, 2006))
trends_start = []
p_vals_start = []

for sy in start_years:
    start_date = pd.Timestamp(f"{sy}-01-01")
    mask_time = pd.to_datetime(time_vals) >= start_date
    mask = valid & mask_time
    if mask.sum() < 60:
        trends_start.append(np.nan)
        p_vals_start.append(np.nan)
        continue
    ty = np.array([(pd.Timestamp(t) - pd.Timestamp(f"{sy}-01-01")).days / 365.25 for t in time_vals])
    sl, ic, r, p, se = linregress(ty[mask], ke_axis[mask])
    trends_start.append(sl * 10)
    p_vals_start.append(p)
    sig = '★' if p < 0.05 else ''
    print(f"  {sy}-2025: {sl*10:+.4f}°/dec, p={p:.4f} {sig}")

# ── 5. 绘图 ──
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# (a) 改进方法 KE 轴时间序列
ax = axes[0]
dates = pd.to_datetime(time_vals)
ax.plot(dates, ke_axis, 'gray', linewidth=0.5, alpha=0.4)
smooth = pd.Series(ke_axis).rolling(12, center=True, min_periods=6).mean().values
ax.plot(dates, smooth, 'b-', linewidth=2, label='12-month running mean')
ax.plot(dates, sl_full * time_years + ic_full, 'r--', linewidth=1.5,
        label=f'Full trend: {sl_full*10:+.4f}°/dec (p={p_full:.3f})')
ax.axvline(pd.Timestamp("2017-08-01"), color='green', linewidth=1, linestyle=':', alpha=0.7)
ax.set_ylabel('Latitude (°N)')
ax.set_title('(a) KE Axis (Improved Gradient Method, 0.125° Monthly, 1993-2025)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) 端点敏感性
ax = axes[1]
colors = ['#d73027' if p < 0.05 else '#4575b4' for p in p_vals]
ax.bar(cutoff_years, trends, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.errorbar(cutoff_years, trends, yerr=[trends - ci_lower, ci_upper - trends],
            fmt='none', color='black', capsize=3)
ax.axhline(0, color='gray', linewidth=1)
ax.axhline(0.45, color='green', linewidth=1, linestyle='--', alpha=0.5, label='Fan (2025): +0.45°/dec')
ax.axhline(-0.45, color='green', linewidth=1, linestyle='--', alpha=0.5)
ax.set_xlabel('Cutoff Year')
ax.set_ylabel('Trend (°/decade)')
ax.set_title('(b) Endpoint Sensitivity: Trend vs Cutoff Year (start = 1993)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(2010.5, max(trends)*0.9, 'Red = p<0.05\nBlue = p≥0.05', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# (c) 起点敏感性
ax = axes[2]
colors_s = ['#d73027' if p < 0.05 else '#4575b4' for p in p_vals_start]
ax.bar(start_years, trends_start, color=colors_s, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axhline(0, color='gray', linewidth=1)
ax.axhline(0.45, color='green', linewidth=1, linestyle='--', alpha=0.5, label='Fan (2025): +0.45°/dec')
ax.set_xlabel('Start Year')
ax.set_ylabel('Trend (°/decade)')
ax.set_title('(c) Start-point Sensitivity: Trend vs Start Year (end = 2025)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "fig_endpoint_sensitivity.png", dpi=300, bbox_inches='tight')
print(f"\n图已保存: {FIG / 'fig_endpoint_sensitivity.png'}")
plt.close()

# ── 6. 保存 ──
stats = {
    "full_1993_2025": {"trend_deg_per_dec": round(sl_full*10, 5), "p_value": round(p_full, 5)},
    "endpoint_sensitivity": {str(y): {"trend": round(t, 5), "p": round(p, 5)}
                             for y, t, p in zip(cutoff_years, trends, p_vals) if not np.isnan(t)},
}
with open(OUT / "endpoint_sensitivity_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2, default=str))
