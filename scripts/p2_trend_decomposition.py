"""P2: KE axis trend decomposition and multi-method comparison
- Compare SLA gradient method vs SSH mean latitude method
- Area-mean SLA in KE region as alternative KE index
- Decompose into trend + decadal + residual
- Cross-correlation with CMEMS KEI
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import pandas as pd
import json

OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")

# ── 1. 读取数据 ──
data_path = "/Volumes/Backup Plus/ssh/cmems_sla_monthly_global_0.125deg_1993_2025.nc"
ds = xr.open_dataset(data_path)

# 经度统一
lon = ds.longitude.values
if lon.min() < 0:
    ds = ds.assign_coords(longitude=(ds.longitude % 360))
    ds = ds.sortby('longitude')

# ── 2. 多种 KE 指标 ──
print("=== 计算多种 KE 指标 ===")

# 指标 A: SLA 梯度极大法（已有，从 P1 加载）
ke_axis_ds = xr.open_dataset(OUT / 'ke_axis_position.nc')
ke_axis_grad = ke_axis_ds['ke_axis_latitude'].values
time_vals = ke_axis_ds.time.values

# 指标 B: KE 区域平均 SLA（北区 - 南区 梯度）
sla_north = ds['sla'].sel(longitude=slice(142, 170), latitude=slice(35, 40)).mean(dim=['latitude', 'longitude'])
sla_south = ds['sla'].sel(longitude=slice(142, 170), latitude=slice(30, 35)).mean(dim=['latitude', 'longitude'])
sla_ke_mean = ds['sla'].sel(longitude=slice(142, 170), latitude=slice(30, 40)).mean(dim=['latitude', 'longitude'])
sla_ns_diff = (sla_north - sla_south).values  # 正值 = 北侧 SLA 更高 = 北移

# 指标 C: SSH 加权纬度（SLA 加权平均纬度）
sla_ke = ds['sla'].sel(longitude=slice(142, 170), latitude=slice(32, 40))
lat_ke = sla_ke.latitude
sla_pos = sla_ke.where(sla_ke > 0)  # 只用正 SLA
weighted_lat = (sla_pos * lat_ke).sum(dim=['latitude', 'longitude']) / sla_pos.sum(dim=['latitude', 'longitude'])
ke_axis_weighted = weighted_lat.values

# 指标 D: CMEMS KEI
kei_path = list(Path("/Users/zhulin/aitest/黑潮延伸体/data").rglob("*kuroshio*.nc"))[0]
kei_ds = xr.open_dataset(kei_path)
kei_raw = kei_ds['kuroshio']

# 对齐时间（KEI 时间可能不完全匹配月均 SLA 时间）
kei_interp = kei_raw.interp(time=time_vals, method='nearest')
kei_vals = kei_interp.values

print(f"时间范围: {str(time_vals[0])[:10]} → {str(time_vals[-1])[:10]}")

# ── 3. 各指标趋势 ──
time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_vals])

indicators = {
    'Gradient Max Lat': ke_axis_grad,
    'N-S SLA Diff (m)': sla_ns_diff,
    'SLA-weighted Lat': ke_axis_weighted,
    'Area-mean SLA (m)': sla_ke_mean.values,
    'CMEMS KEI': kei_vals,
}

print("\n=== 各指标线性趋势 ===")
trends = {}
for name, vals in indicators.items():
    valid = ~np.isnan(vals)
    if valid.sum() < 30:
        print(f"  {name}: 有效数据不足")
        continue
    sl, ic, r, p, se = linregress(time_years[valid], vals[valid])
    trend_dec = sl * 10
    print(f"  {name}: trend={trend_dec:.4f}/decade, p={p:.4f} {'*' if p < 0.05 else ''}")
    trends[name] = {'trend_per_decade': round(trend_dec, 5), 'p_value': round(p, 5), 'r_squared': round(r**2, 4)}

# ── 4. 低通滤波分离年代际信号 ──
def lowpass_filter(data, cutoff_years=7, fs=12):
    """Butterworth 低通滤波，截止周期 cutoff_years 年"""
    valid = ~np.isnan(data)
    if valid.sum() < 100:
        return np.full_like(data, np.nan)
    data_filled = pd.Series(data).interpolate(limit_direction='both').values
    nyq = fs / 2
    cutoff_freq = 1 / (cutoff_years * 12)
    b, a = butter(3, cutoff_freq / nyq, btype='low')
    filtered = filtfilt(b, a, data_filled)
    result = np.where(valid, filtered, np.nan)
    return result

ke_lowpass = lowpass_filter(ke_axis_grad, cutoff_years=7)
ke_highpass = ke_axis_grad - ke_lowpass

# 去年代际后的趋势
valid_lp = ~np.isnan(ke_highpass)
sl_hp, _, _, p_hp, _ = linregress(time_years[valid_lp], ke_highpass[valid_lp])
print(f"\n去年代际(>7yr)后残差趋势: {sl_hp*10:.4f}°/decade, p={p_hp:.4f}")

# ── 5. 分段分析：前半段 vs 后半段 ──
mid = len(time_vals) // 2
valid1 = ~np.isnan(ke_axis_grad[:mid])
valid2 = ~np.isnan(ke_axis_grad[mid:])

if valid1.sum() > 10 and valid2.sum() > 10:
    sl1, _, _, p1, _ = linregress(time_years[:mid][valid1], ke_axis_grad[:mid][valid1])
    sl2, _, _, p2, _ = linregress(time_years[mid:][valid2], ke_axis_grad[mid:][valid2])
    print(f"\n前半段 ({str(time_vals[0])[:4]}-{str(time_vals[mid])[:4]}): {sl1*10:.3f}°/decade, p={p1:.4f}")
    print(f"后半段 ({str(time_vals[mid])[:4]}-{str(time_vals[-1])[:4]}): {sl2*10:.3f}°/decade, p={p2:.4f}")

# ── 6. 绘图：多面板综合分析 ──
fig, axes = plt.subplots(4, 1, figsize=(14, 16))

dates = pd.to_datetime(time_vals)

# (a) KE 轴纬度 + 年代际分量
ax = axes[0]
ax.plot(dates, ke_axis_grad, 'gray', linewidth=0.5, alpha=0.5, label='Monthly')
ax.plot(dates, ke_lowpass, 'b-', linewidth=2, label='Low-pass (>7yr)')
sl, ic, r, p, _ = linregress(time_years[~np.isnan(ke_axis_grad)], ke_axis_grad[~np.isnan(ke_axis_grad)])
ax.plot(dates, sl * time_years + ic, 'r--', linewidth=1.5, label=f'Linear trend: {sl*10:.3f}°/dec (p={p:.2f})')
ax.axvline(pd.Timestamp("2017-08-01"), color='green', linewidth=1, linestyle=':', alpha=0.7)
ax.set_ylabel('Latitude (°N)')
ax.set_title('(a) KE Jet Axis Latitude: Monthly + Decadal Component')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (b) N-S SLA 差值
ax = axes[1]
ns_smooth = lowpass_filter(sla_ns_diff, cutoff_years=3)
ax.fill_between(dates, sla_ns_diff, 0,
                where=sla_ns_diff > 0, color='#d73027', alpha=0.3)
ax.fill_between(dates, sla_ns_diff, 0,
                where=sla_ns_diff <= 0, color='#4575b4', alpha=0.3)
ax.plot(dates, ns_smooth, 'k-', linewidth=1.5, label='3-yr low-pass')
sl_ns, ic_ns, _, p_ns, _ = linregress(time_years[~np.isnan(sla_ns_diff)], sla_ns_diff[~np.isnan(sla_ns_diff)])
ax.plot(dates, sl_ns * time_years + ic_ns, 'r--', linewidth=1, label=f'Trend: {sl_ns*10*100:.2f} cm/dec (p={p_ns:.3f})')
ax.set_ylabel('N-S SLA Diff (m)')
ax.set_title('(b) North (35-40°N) minus South (30-35°N) SLA in KE Region')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (c) KE 区域平均 SLA（海平面上升信号）
ax = axes[2]
sla_mean_vals = sla_ke_mean.values
sla_smooth = lowpass_filter(sla_mean_vals, cutoff_years=3)
ax.plot(dates, sla_mean_vals * 100, 'gray', linewidth=0.5, alpha=0.5)
ax.plot(dates, sla_smooth * 100, 'b-', linewidth=2, label='3-yr low-pass')
sl_m, ic_m, _, p_m, _ = linregress(time_years[~np.isnan(sla_mean_vals)], sla_mean_vals[~np.isnan(sla_mean_vals)])
ax.plot(dates, (sl_m * time_years + ic_m) * 100, 'r--', linewidth=1.5,
        label=f'Trend: {sl_m*10*1000:.1f} mm/yr (p={p_m:.4f})')
ax.set_ylabel('SLA (cm)')
ax.set_title('(c) KE Region Mean SLA (142-170°E, 30-40°N)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (d) CMEMS KEI + KE 轴纬度 标准化叠加
ax = axes[3]
# 标准化
ke_norm = (ke_axis_grad - np.nanmean(ke_axis_grad)) / np.nanstd(ke_axis_grad)
kei_norm = (kei_vals - np.nanmean(kei_vals)) / np.nanstd(kei_vals)
ke_norm_smooth = lowpass_filter(ke_norm, cutoff_years=3)
kei_norm_smooth = lowpass_filter(kei_norm, cutoff_years=3)
ax.plot(dates, ke_norm_smooth, 'b-', linewidth=2, label='KE axis lat (norm, 3yr LP)')
ax.plot(dates, kei_norm_smooth, 'r-', linewidth=2, label='CMEMS KEI (norm, 3yr LP)')
# 相关系数
valid_both = ~np.isnan(ke_norm) & ~np.isnan(kei_norm)
if valid_both.sum() > 30:
    corr = np.corrcoef(ke_norm[valid_both], kei_norm[valid_both])[0, 1]
    ax.set_title(f'(d) Standardized KE Axis Lat vs CMEMS KEI (r={corr:.3f})')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('Standardized Index')
ax.set_xlabel('Year')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(pd.Timestamp("1993-01-01"), pd.Timestamp("2026-01-01"))

plt.tight_layout()
plt.savefig(FIG / "fig3_trend_decomposition.png", dpi=300, bbox_inches='tight')
print(f"\n图已保存: {FIG / 'fig3_trend_decomposition.png'}")
plt.close()

# ── 7. 保存统计 ──
with open(OUT / "trend_decomposition_stats.json", "w") as f:
    json.dump(trends, f, indent=2)
print(json.dumps(trends, indent=2))
