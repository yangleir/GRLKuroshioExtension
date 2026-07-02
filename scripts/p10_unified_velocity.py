"""Unified velocity analysis: MY (1993-2021) + NRT (2022-2024)
Recompute three-method trends on 31-year record.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress, t as t_dist
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import pandas as pd
import json


def bretherton_trend(x, y):
    """OLS trend with Bretherton et al. (1999) autocorrelation-corrected p-value.
    Returns: slope, intercept, p_raw, p_corrected, N_eff, r1_residual
    """
    sl, ic, r, p_raw, se = linregress(x, y)
    residuals = y - (sl * x + ic)
    N = len(y)
    r1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    N_eff = max(N * (1 - r1) / (1 + r1), 3)
    se_corrected = se * np.sqrt(N / N_eff)
    t_stat = sl / se_corrected
    p_corrected = 2 * (1 - t_dist.cdf(abs(t_stat), df=max(N_eff - 2, 1)))
    return sl, ic, p_raw, p_corrected, N_eff, r1

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})

OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")
ROOT = Path("/Volumes/Backup Plus/ssh/dataset-duacs-rep-global-merged-allsat-phy-l4")

def track_methods(ugos_arr, sla_arr, lat, dlat, lat_search=(32, 40)):
    nt = ugos_arr.shape[0]
    vel_max, vel_centroid, sla_grad = [np.full(nt, np.nan) for _ in range(3)]
    for t in range(nt):
        for method, arr, result in [('ugos', ugos_arr, None), ('sla', sla_arr, None)]:
            frame = ugos_arr[t] if method == 'ugos' else sla_arr[t]
            max_lats, centroid_lats, grad_lats = [], [], []
            for j in range(frame.shape[1]):
                col = frame[:, j]
                valid = ~np.isnan(col)
                if valid.sum() < 5: continue
                col_i = np.interp(np.arange(len(col)), np.where(valid)[0], col[valid])
                col_s = gaussian_filter1d(col_i, sigma=2)
                mask = (lat >= lat_search[0]) & (lat <= lat_search[1])
                if method == 'ugos':
                    col_m = np.where(mask, col_s, -np.inf)
                    idx = np.argmax(col_m)
                    if col_m[idx] > 0.03:
                        max_lats.append(lat[idx])
                    u_pos = np.where(mask & (col_s > 0), col_s, 0)
                    total = np.sum(u_pos)
                    if total > 0:
                        centroid_lats.append(np.sum(u_pos * lat) / total)
                else:
                    grad = np.gradient(col_s, dlat)
                    gm = np.where(mask, grad, -np.inf)
                    idx = np.argmax(gm)
                    if gm[idx] > 0:
                        grad_lats.append(lat[idx])
            if method == 'ugos':
                if len(max_lats) >= frame.shape[1]*0.3:
                    vel_max[t] = np.median(max_lats)
                if len(centroid_lats) >= frame.shape[1]*0.3:
                    vel_centroid[t] = np.median(centroid_lats)
            else:
                if len(grad_lats) >= frame.shape[1]*0.3:
                    sla_grad[t] = np.median(grad_lats)
    return vel_max, vel_centroid, sla_grad

def load_region_monthly(files, lat_range, lon_range):
    ugos_list, sla_list, times = [], [], []
    lat_out = None
    for i, fp in enumerate(files):
        ds = xr.open_dataset(fp)
        ke = ds.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))
        if len(ke.longitude) == 0:
            ds.close(); continue
        ugos_list.append(ke['ugos'].isel(time=0).values)
        sla_list.append(ke['sla'].isel(time=0).values)
        times.append(ke.time.values[0])
        lat_out = ke.latitude.values
        ds.close()
    return np.array(ugos_list), np.array(sla_list), np.array(times), lat_out

def load_nrt_monthly(nrt_path, lat_range, lon_range):
    ds = xr.open_dataset(nrt_path)
    ke = ds.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))
    # 月均采样：每月 15 日附近
    monthly = ke.resample(time='MS').first()  # 每月第一天
    ugos_list, times = [], []
    for t in range(len(monthly.time)):
        ugos_list.append(monthly['ugos'].isel(time=t).values)
        times.append(monthly.time.values[t])
    ds.close()
    return np.array(ugos_list), np.array(times), monthly.latitude.values

# ── KE region ──
print("=== KE: MY (1993-2021) ===")
files_my = []
for year in range(1993, 2022):
    for month in range(1, 13):
        day_dir = ROOT / str(year) / f"{month:02d}"
        if not day_dir.exists(): continue
        target = f"dt_global_allsat_phy_l4_{year}{month:02d}15"
        cands = sorted(day_dir.glob(f"{target}*.nc"))
        if cands: files_my.append(cands[0])
        else:
            all_nc = sorted(day_dir.glob("*.nc"))
            if len(all_nc) >= 15: files_my.append(all_nc[14])
            elif all_nc: files_my.append(all_nc[len(all_nc)//2])

ugos_my, sla_my, times_my, lat_ke = load_region_monthly(files_my, (30, 42), (142, 170))
print(f"  MY: {len(times_my)} 月, {str(times_my[0])[:10]} → {str(times_my[-1])[:10]}")

print("=== KE: NRT (2022-2024) ===")
nrt_ke = xr.open_dataset("/Volumes/Backup Plus/ssh/cmems_ugos_vgos_nrt_daily_KE_2022_2025.nc")
nrt_ke_sub = nrt_ke.sel(latitude=slice(30, 42), longitude=slice(142, 170))
nrt_monthly = nrt_ke_sub['ugos'].resample(time='MS').mean()
ugos_nrt = nrt_monthly.values
times_nrt = nrt_monthly.time.values
lat_nrt = nrt_monthly.latitude.values
nrt_ke.close()
print(f"  NRT: {len(times_nrt)} 月, {str(times_nrt[0])[:10]} → {str(times_nrt[-1])[:10]}")

# 合并
ugos_all = np.concatenate([ugos_my, ugos_nrt], axis=0)
# NRT 没有 SLA，用 NaN 填充
sla_nrt_pad = np.full_like(ugos_nrt, np.nan)
sla_all = np.concatenate([sla_my, sla_nrt_pad], axis=0)
times_all = np.concatenate([times_my, times_nrt])
dlat = np.abs(np.diff(lat_ke).mean())

print(f"\n合并: {len(times_all)} 月, {str(times_all[0])[:10]} → {str(times_all[-1])[:10]}")

# 追踪
print("追踪 KE 轴（三方法）...")
vel_max, vel_centroid, sla_grad = track_methods(ugos_all, sla_all, lat_ke, dlat)

time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in times_all])

print("\n=== KE 统一时段趋势 (Bretherton-corrected) ===")
results = {}
for name, vals in [('Velocity max', vel_max), ('Velocity centroid', vel_centroid), ('SLA gradient', sla_grad)]:
    valid = ~np.isnan(vals)
    if valid.sum() < 30:
        print(f"  {name}: 数据不足"); continue
    sl, ic, p_raw, p_corr, N_eff, r1 = bretherton_trend(time_years[valid], vals[valid])
    print(f"  {name}: {sl*10:+.4f}°/dec, p_raw={p_raw:.5f}, p_corrected={p_corr:.5f}, N_eff={N_eff:.1f}, r1={r1:.3f}")
    results[name] = {
        'trend_per_decade': round(sl*10, 5),
        'p_value_raw': round(p_raw, 5),
        'p_value_corrected': round(p_corr, 5),
        'N_eff': round(N_eff, 1),
        'r1_residual': round(r1, 4),
        'n_valid': int(valid.sum())
    }

# 绘图
fig, ax = plt.subplots(figsize=(14, 6))
dates = pd.to_datetime(times_all)
for name, vals, color in [('Velocity max', vel_max, 'red'), ('Velocity centroid', vel_centroid, 'darkgreen'), ('SLA gradient', sla_grad, 'navy')]:
    valid = ~np.isnan(vals)
    if valid.sum() < 30: continue
    smooth = pd.Series(vals).rolling(12, center=True, min_periods=6).mean().values
    sl, ic, p_raw, p_corr, N_eff, r1 = bretherton_trend(time_years[valid], vals[valid])
    p_label = f'p={p_corr:.3f}' if p_corr >= 0.001 else 'p<0.001'
    ax.plot(dates, smooth, color=color, linewidth=2.5, label=f'{name}: {sl*10:+.3f}°/dec ({p_label})')
    ax.plot(dates, sl*time_years + ic, '--', color=color, linewidth=1.5, alpha=0.5)

ax.axvline(pd.Timestamp("2022-01-01"), color='gray', linewidth=1, linestyle=':', alpha=0.5, label='MY→NRT boundary')
ax.set_ylabel('KE Axis Latitude (°N)')
ax.set_title('Three Methods: Unified MY+NRT Period (1993–2024)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(FIG / "fig_unified_three_methods.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "fig_unified_three_methods.png", dpi=300, bbox_inches='tight')
print(f"\n图: {FIG / 'fig_unified_three_methods.png'}")
plt.close()

with open(OUT / "unified_velocity_stats.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
