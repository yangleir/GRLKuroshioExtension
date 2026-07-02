"""ADT vs SLA 梯度法对照（修正符号版，替代 p1b 的 fig1）。

关键修正：ADT（绝对场）的急流锋面 = 最陡下降处（argmax of -dADT/dy）；
p1b 旧版对 ADT 取正梯度极大是符号错误。
SLA 保留带符号梯度极大——那是论文批判的异常场指数本身的定义。

若修正后 ADT gradient 为正趋势（与速度法一致）而 SLA gradient 为负，
则干净地证明偏差源于异常场（MSSH 锚定），而非梯度方法本身 → SI Figure S2。

数据：MY daily 0.25°，每月 15 日采样，1993-2021（NRT 档案无 ADT/SLA）。
输出：figures/figS2_adt_vs_sla.{png,pdf}，output/adt_vs_sla_corrected.json
"""
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path
from scipy.stats import linregress, t as t_dist
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})

OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")
ROOT = Path("/Volumes/Backup Plus/ssh/dataset-duacs-rep-global-merged-allsat-phy-l4")


def bretherton_trend(x, y):
    sl, ic, r, p_raw, se = linregress(x, y)
    residuals = y - (sl * x + ic)
    N = len(y)
    r1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    N_eff = max(N * (1 - r1) / (1 + r1), 3)
    se_c = se * np.sqrt(N / N_eff)
    p_c = 2 * (1 - t_dist.cdf(abs(sl / se_c), df=max(N_eff - 2, 1)))
    return sl, ic, p_raw, p_c, N_eff


def fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.2f}"


# ── 每月 15 日采样 ──
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
                files.append(all_nc[len(all_nc) // 2])

adt_list, sla_list, times = [], [], []
for fp in files:
    ds = xr.open_dataset(fp)
    ke = ds.sel(latitude=slice(30, 42), longitude=slice(142, 170))
    if len(ke.longitude) == 0:
        ds.close()
        continue
    adt_list.append(ke['adt'].isel(time=0).values)
    sla_list.append(ke['sla'].isel(time=0).values)
    times.append(ke.time.values[0])
    ds.close()

adt_arr = np.array(adt_list)
sla_arr = np.array(sla_list)
time_arr = np.array(times)
lat = ke.latitude.values
dlat = np.abs(np.diff(lat).mean())
time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_arr])
print(f"数据: {adt_arr.shape}, {str(time_arr[0])[:10]} → {str(time_arr[-1])[:10]}")


def track_axis(data, sign, lat_search=(32, 40)):
    """sign=-1: 最陡下降（绝对场锋面）; sign=+1: 正梯度极大（异常场指数）"""
    nt = data.shape[0]
    axis = np.full(nt, np.nan)
    band = (lat >= lat_search[0]) & (lat <= lat_search[1])
    for t in range(nt):
        frame = data[t]
        lats = []
        for j in range(frame.shape[1]):
            col = frame[:, j]
            valid = ~np.isnan(col)
            if valid.sum() < 5:
                continue
            col_i = np.interp(np.arange(len(col)), np.where(valid)[0], col[valid])
            col_s = gaussian_filter1d(col_i, sigma=2)
            grad = np.gradient(col_s, dlat) * sign
            gm = np.where(band, grad, -np.inf)
            idx = np.argmax(gm)
            if gm[idx] > 0:
                lats.append(lat[idx])
        if len(lats) >= frame.shape[1] * 0.3:
            axis[t] = np.median(lats)
    return axis


adt_axis = track_axis(adt_arr, sign=-1)   # 锋面 = 最陡下降
sla_axis = track_axis(sla_arr, sign=+1)   # 异常场指数（论文批判对象）

results = {}
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
for ax, name, series, color in [
        (axes[0], "ADT gradient (absolute field, steepest descent)", adt_axis, 'darkred'),
        (axes[1], "SLA gradient (anomaly field)", sla_axis, 'navy')]:
    valid = ~np.isnan(series)
    sl, ic, p_raw, p_c, N_eff = bretherton_trend(time_years[valid], series[valid])
    print(f"{name}: {sl*10:+.4f}°/dec p_raw={p_raw:.5f} p_corr={p_c:.5f} N_eff={N_eff:.1f} n={valid.sum()}")
    results[name] = {"trend_per_decade": round(sl * 10, 5), "p_raw": round(p_raw, 5),
                     "p_corrected": round(p_c, 5), "N_eff": round(N_eff, 1),
                     "n_valid": int(valid.sum())}
    dates = pd.to_datetime(time_arr)
    smooth = pd.Series(series).rolling(12, center=True, min_periods=6).mean().values
    ax.plot(dates, series, color=color, alpha=0.25, linewidth=0.8)
    ax.plot(dates, smooth, color=color, linewidth=2.5,
            label=f'{name.split("(")[0].strip()}: {sl*10:+.2f}°/dec ({fmt_p(p_c)})')
    ax.plot(dates, sl * time_years + ic, '--', color=color, linewidth=1.5)
    ax.set_ylabel('KE Axis Latitude (°N)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

axes[0].set_title('(a) ADT gradient method (absolute field): jet front = steepest SSH decrease',
                  fontweight='bold')
axes[1].set_title('(b) SLA gradient method (anomaly field): same data, MSSH removed',
                  fontweight='bold')
axes[1].xaxis.set_major_locator(mdates.YearLocator(5))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(FIG / "figS2_adt_vs_sla.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "figS2_adt_vs_sla.png", dpi=300, bbox_inches='tight')
plt.close()

with open(OUT / "adt_vs_sla_corrected.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
