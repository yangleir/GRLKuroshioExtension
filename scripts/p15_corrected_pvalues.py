"""Bretherton (1999) 校正 p 值补算：
1. 月度 SLA gradient 与 SLA-weighted 指数（1993-2025，复刻 p2b 逻辑）
2. GS velocity-max（MY 1993-2021 月中快照 + NRT 2022-2024 月均，复刻 p10/p11 逻辑）
输出 output/corrected_pvalues.json
"""
import json
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from scipy.stats import linregress, t as t_dist
from scipy.ndimage import gaussian_filter1d

OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
DRIVE = Path("/Volumes/Backup Plus/ssh")
ROOT = DRIVE / "dataset-duacs-rep-global-merged-allsat-phy-l4"


def bretherton_trend(x, y):
    sl, ic, r, p_raw, se = linregress(x, y)
    residuals = y - (sl * x + ic)
    N = len(y)
    r1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    N_eff = max(N * (1 - r1) / (1 + r1), 3)
    se_corrected = se * np.sqrt(N / N_eff)
    t_stat = sl / se_corrected
    p_corrected = 2 * (1 - t_dist.cdf(abs(t_stat), df=max(N_eff - 2, 1)))
    return sl, p_raw, p_corrected, N_eff, r1


results = {}

# ── 1. 月度 SLA 指数 ──────────────────────────────
print("=== Monthly SLA indices (1993-2025) ===")
ds = xr.open_dataset(DRIVE / "cmems_sla_monthly_global_0.125deg_1993_2025.nc")
ke = ds['sla'].sel(latitude=slice(30, 42), longitude=slice(142, 170)).load()
lat = ke.latitude.values
dlat = np.abs(np.diff(lat).mean())
nt = ke.shape[0]
time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25
                       for t in ke.time.values])

grad_axis = np.full(nt, np.nan)
wt_axis = np.full(nt, np.nan)
band = (lat >= 32) & (lat <= 40)
for t in range(nt):
    frame = ke.isel(time=t).values
    # gradient index (p2b: per-longitude argmax of signed gradient, median)
    jet_lats = []
    for j in range(frame.shape[1]):
        col = frame[:, j]
        if np.isnan(col).all():
            continue
        col_s = gaussian_filter1d(np.nan_to_num(col, nan=np.nanmean(col)), sigma=2)
        grad = np.gradient(col_s, dlat)
        grad_m = np.where(band, grad, -np.inf)
        idx = np.argmax(grad_m)
        if grad_m[idx] > 0:
            jet_lats.append(lat[idx])
    if len(jet_lats) >= frame.shape[1] * 0.3:
        grad_axis[t] = np.median(jet_lats)
    # weighted index (p2b: positive SLA weighting over 32-40N)
    sub = frame[band, :]
    pos = np.where(np.nan_to_num(sub, nan=0) > 0, sub, 0)
    tot = np.nansum(pos)
    if tot > 0:
        wt_axis[t] = np.nansum(pos * lat[band][:, None]) / tot

for name, series in [("SLA gradient (monthly)", grad_axis), ("SLA weighted (monthly)", wt_axis)]:
    valid = ~np.isnan(series)
    sl, p_raw, p_corr, N_eff, r1 = bretherton_trend(time_years[valid], series[valid])
    print(f"  {name}: {sl*10:+.4f}°/dec p_raw={p_raw:.5f} p_corr={p_corr:.5f} N_eff={N_eff:.1f}")
    results[name] = {"trend_per_decade": round(sl * 10, 5), "p_raw": round(p_raw, 5),
                     "p_corrected": round(p_corr, 5), "N_eff": round(N_eff, 1),
                     "r1": round(r1, 4), "n_valid": int(valid.sum())}
ds.close()

# ── 2. GS velocity-max (MY + NRT) ────────────────
print("=== GS velocity max (1993-2024) ===")
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
lat_gs = None
for fp in files_my:
    d = xr.open_dataset(fp)
    gs = d.sel(latitude=slice(33, 47), longitude=slice(285, 315))
    if len(gs.longitude) == 0:
        gs = d.sel(latitude=slice(33, 47), longitude=slice(-75, -45))
    ugos_list.append(gs['ugos'].isel(time=0).values)
    times.append(gs.time.values[0])
    lat_gs = gs.latitude.values
    d.close()

nrt = xr.open_dataset(DRIVE / "cmems_ugos_vgos_nrt_daily_GS_2022_2025.nc")
gs_sub = nrt.sel(latitude=slice(33, 47), longitude=slice(-75, -45))
if len(gs_sub.longitude) == 0:
    gs_sub = nrt.sel(latitude=slice(33, 47), longitude=slice(285, 315))
nrt_m = gs_sub['ugos'].resample(time='MS').mean().sel(time=slice(None, '2024-12-31'))
for t in range(len(nrt_m.time)):
    ugos_list.append(nrt_m.isel(time=t).values)
    times.append(nrt_m.time.values[t])
nrt.close()

ugos_arr = np.array(ugos_list)
time_years_gs = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25
                          for t in times])
band_gs = (lat_gs >= lat_gs.min() + 2) & (lat_gs <= lat_gs.max() - 2)
axis_gs = np.full(len(times), np.nan)
for t in range(len(times)):
    frame = ugos_arr[t]
    lats = []
    for j in range(frame.shape[1]):
        col = frame[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 5:
            continue
        col_i = np.interp(np.arange(len(col)), np.where(valid)[0], col[valid])
        col_s = gaussian_filter1d(col_i, sigma=2)
        col_m = np.where(band_gs, col_s, -np.inf)
        idx = np.argmax(col_m)
        if col_m[idx] > 0.03:
            lats.append(lat_gs[idx])
    if len(lats) >= frame.shape[1] * 0.3:
        axis_gs[t] = np.median(lats)

valid = ~np.isnan(axis_gs)
sl, p_raw, p_corr, N_eff, r1 = bretherton_trend(time_years_gs[valid], axis_gs[valid])
print(f"  GS vel-max: {sl*10:+.4f}°/dec p_raw={p_raw:.5f} p_corr={p_corr:.5f} N_eff={N_eff:.1f}")
results["GS velocity max (unified)"] = {
    "trend_per_decade": round(sl * 10, 5), "p_raw": round(p_raw, 5),
    "p_corrected": round(p_corr, 5), "N_eff": round(N_eff, 1),
    "r1": round(r1, 4), "n_valid": int(valid.sum())}

with open(OUT / "corrected_pvalues.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
