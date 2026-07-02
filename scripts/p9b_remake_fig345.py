"""Remake Fig 3 (SLA bias profiles), Fig 4 (method comparison), Fig 5 (three methods)
with larger fonts for NC publication.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# ══════════════════════════════════════
# Fig 3: SLA bias meridional profiles
# ══════════════════════════════════════
print("=== Fig 3: SLA bias profiles ===")

def read_ke_profiles(years, months=[3,6,9,12]):
    adt_list, sla_list = [], []
    lat_p = None
    for yr in years:
        for mo in months:
            day_dir = ROOT / str(yr) / f"{mo:02d}"
            nc_files = sorted(day_dir.glob("*.nc"))
            if not nc_files: continue
            for fp in nc_files[::5]:
                ds = xr.open_dataset(fp)
                ke = ds.sel(latitude=slice(28, 44), longitude=slice(142, 170))
                if len(ke.longitude) == 0:
                    ds.close(); continue
                adt_list.append(ke['adt'].isel(time=0).mean(dim='longitude').values)
                sla_list.append(ke['sla'].isel(time=0).mean(dim='longitude').values)
                lat_p = ke.latitude.values
                ds.close()
    return lat_p, np.nanmean(adt_list, axis=0), np.nanmean(sla_list, axis=0)

lat_p, adt_early, sla_early = read_ke_profiles([1994, 1995, 1996])
_, adt_late, sla_late = read_ke_profiles([2019, 2020, 2021])
mssh = np.nanmean([adt_early, adt_late], axis=0) - np.nanmean([sla_early, sla_late], axis=0)
dlat_p = np.abs(np.diff(lat_p).mean())
mask = (lat_p >= 32) & (lat_p <= 40)

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

# (a) ADT
ax = axes[0]
ax.plot(adt_early, lat_p, 'b-', linewidth=2.5, label='1994–1996')
ax.plot(adt_late, lat_p, 'r-', linewidth=2.5, label='2019–2021')
# 急流锋面 = ADT 最陡下降处（向北递减），取 -grad 的极大
grad_e = np.gradient(gaussian_filter1d(adt_early, 2), dlat_p)
grad_l = np.gradient(gaussian_filter1d(adt_late, 2), dlat_p)
idx_e = np.argmax(np.where(mask, -grad_e, -np.inf))
idx_l = np.argmax(np.where(mask, -grad_l, -np.inf))
ax.axhline(lat_p[idx_e], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axhline(lat_p[idx_l], color='r', linestyle='--', alpha=0.5, linewidth=1.5)
ax.annotate('', xy=(adt_early[idx_l]*0.95, lat_p[idx_l]),
            xytext=(adt_early[idx_e]*0.95, lat_p[idx_e]),
            arrowprops=dict(arrowstyle='->', color='green', lw=3))
ax.set_xlabel('ADT (m)')
ax.set_ylabel('Latitude (°N)')
ax.set_title('(a) SSH/ADT:\njet shifts north', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# (b) MSSH
ax = axes[1]
ax.plot(mssh, lat_p, 'k-', linewidth=2.5)
# MSSH 锋面同样取最陡下降处
grad_m = np.gradient(gaussian_filter1d(mssh, 2), dlat_p)
idx_m = np.argmax(np.where(mask, -grad_m, -np.inf))
ax.axhline(lat_p[idx_m], color='k', linestyle='--', alpha=0.5, linewidth=1.5,
           label=f'MSSH front: {lat_p[idx_m]:.1f}°N')
ax.set_xlabel('MSSH (m)')
ax.set_title('(b) MSSH:\nanchored to historical mean', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# (c) SLA
ax = axes[2]
ax.plot(sla_early, lat_p, 'b-', linewidth=2.5, label='1994–1996')
ax.plot(sla_late, lat_p, 'r-', linewidth=2.5, label='2019–2021')
grad_se = np.gradient(gaussian_filter1d(sla_early, 2), dlat_p)
grad_sl = np.gradient(gaussian_filter1d(sla_late, 2), dlat_p)
idx_se = np.argmax(np.where(mask, grad_se, -np.inf))
idx_sl = np.argmax(np.where(mask, grad_sl, -np.inf))
ax.axhline(lat_p[idx_se], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axhline(lat_p[idx_sl], color='r', linestyle='--', alpha=0.5, linewidth=1.5)
if lat_p[idx_sl] < lat_p[idx_se]:
    ax.annotate('', xy=(sla_early[idx_sl]*0.9, lat_p[idx_sl]),
                xytext=(sla_early[idx_se]*0.9, lat_p[idx_se]),
                arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    ax.text(sla_early.min()*0.4, (lat_p[idx_se]+lat_p[idx_sl])/2,
            'Apparent\nsouthward!', color='orange', fontsize=13, fontweight='bold', ha='center')
ax.set_xlabel('SLA (m)')
ax.set_title('(c) SLA = SSH − MSSH:\nbias reverses trend', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.suptitle('Self-Concealing Mechanism: SSH Shifts North, SLA Gradient Shifts South',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG / "fig3_sla_bias_v2.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "fig3_sla_bias_v2.png", dpi=300, bbox_inches='tight')
print(f"保存: fig3_sla_bias_v2")
plt.close()

# ══════════════════════════════════════
# Fig 4: Method comparison (SLA gradient vs weighted + N-S diff)
# ══════════════════════════════════════
print("\n=== Fig 4: Method comparison ===")

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
time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_vals])

# SLA gradient
ke_grad = np.full(len(time_vals), np.nan)
for t in range(len(time_vals)):
    frame = sla_ke.values[t]
    lats = []
    for j in range(frame.shape[1]):
        col = frame[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 5: continue
        col_i = np.interp(np.arange(len(col)), np.where(valid)[0], col[valid])
        col_s = gaussian_filter1d(col_i, sigma=2)
        grad = np.gradient(col_s, dlat)
        m = (lat >= 32) & (lat <= 40)
        gm = np.where(m, grad, -np.inf)
        idx = np.argmax(gm)
        if gm[idx] > 0: lats.append(lat[idx])
    if len(lats) >= frame.shape[1]*0.3:
        ke_grad[t] = np.median(lats)

# SLA weighted
sla_pos = sla_ke.where(sla_ke > 0)
ke_wt = (sla_pos * sla_ke.latitude).sum(dim=['latitude','longitude']) / sla_pos.sum(dim=['latitude','longitude'])
ke_wt_vals = ke_wt.values

# N-S diff
sla_n = ds['sla'].sel(longitude=slice(142,170), latitude=slice(35,40)).mean(dim=['latitude','longitude']).values
sla_s = ds['sla'].sel(longitude=slice(142,170), latitude=slice(30,35)).mean(dim=['latitude','longitude']).values
ns_diff = sla_n - sla_s

dates = pd.to_datetime(time_vals)

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# (a) Two methods
ax = axes[0]
g_smooth = pd.Series(ke_grad).rolling(12, center=True, min_periods=6).mean().values
w_smooth = pd.Series(ke_wt_vals).rolling(12, center=True, min_periods=6).mean().values
valid_g = ~np.isnan(ke_grad)
valid_w = ~np.isnan(ke_wt_vals)
sl_g, ic_g, _, p_g, _ = linregress(time_years[valid_g], ke_grad[valid_g])
sl_w, ic_w, _, p_w, _ = linregress(time_years[valid_w], ke_wt_vals[valid_w])
ax.plot(dates, g_smooth, 'b-', linewidth=2.5, label=f'SLA gradient: {sl_g*10:+.2f}°/dec (p={p_g:.3f})')
ax.plot(dates, w_smooth, 'r-', linewidth=2.5, label=f'SLA weighted: {sl_w*10:+.2f}°/dec (p={p_w:.3f})')
ax.plot(dates, sl_g*time_years+ic_g, 'b--', linewidth=1.5)
ax.plot(dates, sl_w*time_years+ic_w, 'r--', linewidth=1.5)
ax.set_ylabel('Latitude (°N)')
ax.set_title('(a) SLA Gradient vs SLA-Weighted Methods', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# (b) N-S SLA diff
ax = axes[1]
ns_smooth = pd.Series(ns_diff).rolling(12, center=True, min_periods=6).mean().values
ax.fill_between(dates, ns_diff, 0, where=ns_diff>0, color='#d73027', alpha=0.3)
ax.fill_between(dates, ns_diff, 0, where=ns_diff<=0, color='#4575b4', alpha=0.3)
ax.plot(dates, ns_smooth, 'k-', linewidth=2)
valid_ns = ~np.isnan(ns_diff)
sl_ns, ic_ns, _, p_ns, _ = linregress(time_years[valid_ns], ns_diff[valid_ns])
ax.plot(dates, sl_ns*time_years+ic_ns, 'r--', linewidth=2,
        label=f'Trend: {sl_ns*10*100:+.1f} cm/dec (p={p_ns:.4f})')
ax.set_ylabel('N−S SLA Diff (m)')
ax.set_title('(b) North (35–40°N) minus South (30–35°N) SLA Differential', fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# (c) Method disagreement vs N-S diff
ax = axes[2]
diff_meth = ke_wt_vals - ke_grad
diff_smooth = pd.Series(diff_meth).rolling(12, center=True, min_periods=6).mean().values
ax2 = ax.twinx()
ax.plot(dates, diff_smooth, 'k-', linewidth=2.5, label='Method diff (Weighted − Gradient)')
ax2.plot(dates, ns_smooth*100, 'r-', linewidth=2, alpha=0.7, label='N−S SLA diff (cm)')
ax.set_ylabel('Latitude difference (°)')
ax2.set_ylabel('N−S SLA diff (cm)', color='r')
ax.set_title('(c) Method Disagreement Tracks N−S SLA Differential Rise', fontweight='bold')
lines1, l1 = ax.get_legend_handles_labels()
lines2, l2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, l1+l2, loc='upper left')
ax.grid(True, alpha=0.3)

for a in axes:
    a.xaxis.set_major_locator(mdates.YearLocator(5))
    a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(FIG / "fig4_method_comp_v2.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "fig4_method_comp_v2.png", dpi=300, bbox_inches='tight')
print(f"保存: fig4_method_comp_v2")
plt.close()

# ══════════════════════════════════════
# Fig 5: Three methods (already has good data, just re-plot with larger fonts)
# ══════════════════════════════════════
print("\n=== Fig 5: Three methods ===")

three = pd.read_json(OUT / "three_method_comparison.json")
# Re-plot is complex (needs raw data). Use the centroid script output.
# For speed, just re-read the data from p5d
# Actually the quickest fix: regenerate from p5d with updated rcParams

# The rcParams are already set globally. Just run the core plot logic.
exec(open("/Users/zhulin/aitest/黑潮延伸体/scripts/p5d_centroid_method.py").read())

# Overwrite with v2
import shutil
shutil.copy(FIG / "fig_three_methods.png", FIG / "fig5_three_methods_v2.png")

# 2026-07: 不再自动覆盖 manuscript 图件。
# - manuscript/figB 由 p9c_figB_front_fix.py 生成（锋面符号修正版）
# - manuscript/fig2 用 p2b 的 fig2_method_comparison（与正文数字一致）
# - manuscript/fig_three_methods 用 p10 的 fig_unified_three_methods（1993-2024 统一版）
print("全部完成（manuscript 拷贝已移除，见注释）")
