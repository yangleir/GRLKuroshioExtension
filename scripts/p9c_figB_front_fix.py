"""Regenerate figB (self-concealing mechanism profiles) with corrected front definition.

锋面 = SSH 最陡下降处（argmax of -dADT/dy），修正 p9b 旧版取正梯度极大的符号错误。
面板 c 的 SLA 保留带符号梯度极大——那正是论文批判的异常场指数本身的定义。
输出: figures/figB_v3.{png,pdf} 并拷贝为 manuscript/figB.{png,pdf}
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import shutil

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'font.family': 'sans-serif',
})

FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")
MAN = Path("/Users/zhulin/aitest/黑潮延伸体/manuscript")
ROOT = Path("/Volumes/Backup Plus/ssh/dataset-duacs-rep-global-merged-allsat-phy-l4")


def read_ke_profiles(years, months=(3, 6, 9, 12)):
    adt_list, sla_list = [], []
    lat_p = None
    for yr in years:
        for mo in months:
            day_dir = ROOT / str(yr) / f"{mo:02d}"
            if not day_dir.exists():
                continue
            nc_files = sorted(day_dir.glob("*.nc"))
            for fp in nc_files[::5]:
                ds = xr.open_dataset(fp)
                ke = ds.sel(latitude=slice(28, 44), longitude=slice(142, 170))
                if len(ke.longitude) == 0:
                    ds.close()
                    continue
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

# (a) ADT: 锋面 = 最陡下降处
ax = axes[0]
ax.plot(adt_early, lat_p, 'b-', linewidth=2.5, label='1994–1996')
ax.plot(adt_late, lat_p, 'r-', linewidth=2.5, label='2019–2021')
grad_e = np.gradient(gaussian_filter1d(adt_early, 2), dlat_p)
grad_l = np.gradient(gaussian_filter1d(adt_late, 2), dlat_p)
idx_e = np.argmax(np.where(mask, -grad_e, -np.inf))
idx_l = np.argmax(np.where(mask, -grad_l, -np.inf))
print(f"ADT front: early {lat_p[idx_e]:.1f}°N -> late {lat_p[idx_l]:.1f}°N")
ax.axhline(lat_p[idx_e], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axhline(lat_p[idx_l], color='r', linestyle='--', alpha=0.5, linewidth=1.5)
ax.annotate('', xy=(adt_early[idx_l] * 0.95, lat_p[idx_l]),
            xytext=(adt_early[idx_e] * 0.95, lat_p[idx_e]),
            arrowprops=dict(arrowstyle='->', color='green', lw=3))
ax.set_xlabel('ADT (m)')
ax.set_ylabel('Latitude (°N)')
ax.set_title('(a) SSH/ADT:\njet front shifts north', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# (b) MSSH: 锋面锚定在历史平均位置
ax = axes[1]
ax.plot(mssh, lat_p, 'k-', linewidth=2.5)
grad_m = np.gradient(gaussian_filter1d(mssh, 2), dlat_p)
idx_m = np.argmax(np.where(mask, -grad_m, -np.inf))
print(f"MSSH front: {lat_p[idx_m]:.1f}°N")
ax.axhline(lat_p[idx_m], color='k', linestyle='--', alpha=0.5, linewidth=1.5,
           label=f'MSSH front: {lat_p[idx_m]:.1f}°N')
ax.set_xlabel('MSSH (m)')
ax.set_title('(b) MSSH:\nanchored to historical mean', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# (c) SLA: 带符号梯度极大（被批判的异常场指数定义）
ax = axes[2]
ax.plot(sla_early, lat_p, 'b-', linewidth=2.5, label='1994–1996')
ax.plot(sla_late, lat_p, 'r-', linewidth=2.5, label='2019–2021')
grad_se = np.gradient(gaussian_filter1d(sla_early, 2), dlat_p)
grad_sl = np.gradient(gaussian_filter1d(sla_late, 2), dlat_p)
idx_se = np.argmax(np.where(mask, grad_se, -np.inf))
idx_sl = np.argmax(np.where(mask, grad_sl, -np.inf))
print(f"SLA gradient max: early {lat_p[idx_se]:.1f}°N -> late {lat_p[idx_sl]:.1f}°N")
ax.axhline(lat_p[idx_se], color='b', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axhline(lat_p[idx_sl], color='r', linestyle='--', alpha=0.5, linewidth=1.5)
if lat_p[idx_sl] < lat_p[idx_se]:
    ax.annotate('', xy=(sla_early[idx_sl] * 0.9, lat_p[idx_sl]),
                xytext=(sla_early[idx_se] * 0.9, lat_p[idx_se]),
                arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    ax.text(sla_early.min() * 0.4, (lat_p[idx_se] + lat_p[idx_sl]) / 2,
            'Apparent\nsouthward!', color='orange', fontsize=13, fontweight='bold', ha='center')
ax.set_xlabel('SLA (m)')
ax.set_title('(c) SLA = SSH − MSSH:\nbias reverses trend', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.suptitle('Self-Concealing Mechanism: SSH Front Shifts North, SLA Gradient Shifts South',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG / "figB_v3.pdf", dpi=300, bbox_inches='tight')
plt.savefig(FIG / "figB_v3.png", dpi=300, bbox_inches='tight')
plt.close()

shutil.copy(FIG / "figB_v3.pdf", MAN / "figB.pdf")
shutil.copy(FIG / "figB_v3.png", MAN / "figB.png")
print("已拷贝到 manuscript/figB.{pdf,png}")
