"""速度法（velocity max + centroid）滑动端点敏感性 → SI 端点稳健性图。

复刻 p10 的 KE 统一管线（MY 月中快照 1993-2021 + NRT 月均 2022-2024），
对 cutoff = 2010..2024 逐年算 Bretherton 校正趋势与 95% CI。
同时缓存 KE 轴序列 output/ke_axis_unified.nc（后续免扫盘）。
输出 figures/figS4_endpoint_velocity.{png,pdf}, output/endpoint_velocity_stats.json
"""
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from pathlib import Path
from scipy.stats import linregress, t as t_dist
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})

OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")
MAN = Path("/Users/zhulin/aitest/黑潮延伸体/manuscript")
DRIVE = Path("/Volumes/Backup Plus/ssh")
ROOT = DRIVE / "dataset-duacs-rep-global-merged-allsat-phy-l4"


def bretherton_trend_ci(x, y):
    sl, ic, r, p_raw, se = linregress(x, y)
    residuals = y - (sl * x + ic)
    N = len(y)
    r1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    N_eff = max(N * (1 - r1) / (1 + r1), 3)
    se_c = se * np.sqrt(N / N_eff)
    df = max(N_eff - 2, 1)
    p_c = 2 * (1 - t_dist.cdf(abs(sl / se_c), df=df))
    ci = t_dist.ppf(0.975, df=df) * se_c
    return sl, p_c, ci


# ── KE 统一序列（MY + NRT）──
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

ugos_list, times = [], []
lat = None
for fp in files_my:
    ds = xr.open_dataset(fp)
    ke = ds.sel(latitude=slice(30, 42), longitude=slice(142, 170))
    if len(ke.longitude) == 0:
        ds.close()
        continue
    ugos_list.append(ke['ugos'].isel(time=0).values)
    times.append(ke.time.values[0])
    lat = ke.latitude.values
    ds.close()

nrt = xr.open_dataset(DRIVE / "cmems_ugos_vgos_nrt_daily_KE_2022_2025.nc")
ke_sub = nrt.sel(latitude=slice(30, 42), longitude=slice(142, 170))
u_m = ke_sub['ugos'].resample(time='MS').mean().sel(time=slice(None, '2024-12-31'))
for t in range(len(u_m.time)):
    ugos_list.append(u_m.isel(time=t).values)
    times.append(u_m.time.values[t])
nrt.close()

ugos_arr = np.array(ugos_list)
time_arr = np.array(times)
time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_arr])
print(f"KE 统一: {ugos_arr.shape}, {str(time_arr[0])[:10]} → {str(time_arr[-1])[:10]}")

band = (lat >= 32) & (lat <= 40)
nt = len(time_arr)
vel_max = np.full(nt, np.nan)
vel_cen = np.full(nt, np.nan)
for t in range(nt):
    frame = ugos_arr[t]
    max_lats, cen_lats = [], []
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
            max_lats.append(lat[idx])
        u_pos = np.where(band & (col_s > 0), col_s, 0)
        tot = np.sum(u_pos)
        if tot > 0:
            cen_lats.append(np.sum(u_pos * lat) / tot)
    if len(max_lats) >= frame.shape[1] * 0.3:
        vel_max[t] = np.median(max_lats)
    if len(cen_lats) >= frame.shape[1] * 0.3:
        vel_cen[t] = np.median(cen_lats)

# 缓存
xr.Dataset({"vel_max_latitude": ("time", vel_max),
            "vel_centroid_latitude": ("time", vel_cen)},
           coords={"time": time_arr},
           attrs={"method": "p10 pipeline: MY mid-month 1993-2021 + NRT monthly mean 2022-2024",
                  "region": "142-170E, search band 32-40N"}).to_netcdf(OUT / "ke_axis_unified.nc")

# ── 滑动端点 ──
cutoffs = list(range(2010, 2025))
results = {"Velocity max": {}, "Velocity centroid": {}}
for name, series in [("Velocity max", vel_max), ("Velocity centroid", vel_cen)]:
    for cy in cutoffs:
        sel = (time_years <= (cy - 1993 + 1)) & ~np.isnan(series)
        sl, p_c, ci = bretherton_trend_ci(time_years[sel], series[sel])
        results[name][cy] = {"trend": round(sl * 10, 4), "p_corrected": round(p_c, 5),
                             "ci95": round(ci * 10, 4)}
        print(f"{name} cutoff {cy}: {sl*10:+.3f} ± {ci*10:.3f} °/dec, p={p_c:.4f}")

with open(OUT / "endpoint_velocity_stats.json", "w") as f:
    json.dump(results, f, indent=2)

# ── 绘图 ──
fig, ax = plt.subplots(figsize=(12, 6))
for name, color, offset in [("Velocity max", 'tab:red', -0.12), ("Velocity centroid", 'darkgreen', 0.12)]:
    x = np.array(cutoffs, dtype=float) + offset
    tr = np.array([results[name][c]["trend"] for c in cutoffs])
    ci = np.array([results[name][c]["ci95"] for c in cutoffs])
    sig = np.array([results[name][c]["p_corrected"] < 0.05 for c in cutoffs])
    ax.errorbar(x, tr, yerr=ci, fmt='o', color=color, capsize=3, label=f'{name} ± 95% CI')
    ax.scatter(x[sig], tr[sig], s=90, facecolors=color, edgecolors='k', zorder=5)
ax.axhline(0, color='gray', linewidth=1)
ax.set_xlabel('Cutoff year (start fixed at 1993)')
ax.set_ylabel('Trend (°/decade)')
ax.set_title('Endpoint sensitivity of KE velocity-based trends (Bretherton-corrected)', fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "figS4_endpoint_velocity.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "figS4_endpoint_velocity.png", dpi=300, bbox_inches='tight')
plt.close()

shutil.copy(FIG / "figS4_endpoint_velocity.pdf", MAN / "figS4.pdf")
shutil.copy(FIG / "figS4_endpoint_velocity.png", MAN / "figS4.png")
print("已拷贝 manuscript/figS4.{pdf,png}")
