"""P3: CMEMS OMI Kuroshio Extension Index analysis
- Time series plot with stable/unstable state identification
- Sliding window statistics: state duration and frequency changes
- Trend analysis of KEI
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from pathlib import Path

OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# ── 1. 读取黑潮指数 ──
kei_path = list(Path("/Users/zhulin/aitest/黑潮延伸体/data").rglob("*kuroshio*.nc"))[0]
ds = xr.open_dataset(kei_path)
print(ds)

kei = ds['kuroshio']
time = kei.time.values

print(f"\nKEI 时间范围: {str(time[0])[:10]} → {str(time[-1])[:10]}")
print(f"KEI 范围: {float(kei.min()):.3f} ~ {float(kei.max()):.3f}")
print(f"KEI 均值: {float(kei.mean()):.3f}, 标准差: {float(kei.std()):.3f}")

# ── 2. 定义稳定/不稳定态 ──
# CMEMS OMI: 正值 = 稳定态（EKE低, 射流强）, 负值 = 不稳定态
stable = kei > 0
unstable = kei <= 0

n_stable = int(stable.sum())
n_unstable = int(unstable.sum())
print(f"\n稳定态月数: {n_stable} ({100*n_stable/len(kei):.1f}%)")
print(f"不稳定态月数: {n_unstable} ({100*n_unstable/len(kei):.1f}%)")

# ── 3. 滑动窗口统计（10年窗口，1年步长） ──
window_years = 10
window_months = window_years * 12
step_months = 12

center_times = []
stable_fractions = []
mean_kei_vals = []

for start in range(0, len(kei) - window_months + 1, step_months):
    end = start + window_months
    window = kei.isel(time=slice(start, end))
    center_time = time[start + window_months // 2]
    frac = float((window > 0).sum()) / window_months
    center_times.append(center_time)
    stable_fractions.append(frac)
    mean_kei_vals.append(float(window.mean()))

center_times = np.array(center_times)
stable_fractions = np.array(stable_fractions)
mean_kei_vals = np.array(mean_kei_vals)

# ── 4. KEI 线性趋势 ──
import pandas as pd
time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time])
slope, intercept, r, p, se = linregress(time_years, kei.values)
print(f"\nKEI 线性趋势: {slope:.4f}/year (p={p:.4f})")
print(f"趋势线: KEI = {slope:.4f} * year + {intercept:.4f}")

# ── 5. 状态持续时间统计 ──
states = (kei.values > 0).astype(int)  # 1=stable, 0=unstable
durations_stable = []
durations_unstable = []
current_state = states[0]
current_duration = 1

for i in range(1, len(states)):
    if states[i] == current_state:
        current_duration += 1
    else:
        if current_state == 1:
            durations_stable.append(current_duration)
        else:
            durations_unstable.append(current_duration)
        current_state = states[i]
        current_duration = 1

if current_state == 1:
    durations_stable.append(current_duration)
else:
    durations_unstable.append(current_duration)

print(f"\n稳定态段数: {len(durations_stable)}, 平均持续: {np.mean(durations_stable):.1f} 月")
print(f"不稳定态段数: {len(durations_unstable)}, 平均持续: {np.mean(durations_unstable):.1f} 月")

# ── 6. 绘图 ──
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

# Panel (a): KEI 时间序列
ax = axes[0]
ax.fill_between(pd.to_datetime(time), kei.values, 0,
                where=kei.values > 0, color='#d73027', alpha=0.4, label='Stable (KEI>0)')
ax.fill_between(pd.to_datetime(time), kei.values, 0,
                where=kei.values <= 0, color='#4575b4', alpha=0.4, label='Unstable (KEI≤0)')
ax.plot(pd.to_datetime(time), kei.values, 'k-', linewidth=0.8)
trend_line = slope * time_years + intercept
ax.plot(pd.to_datetime(time), trend_line, 'r--', linewidth=1.5,
        label=f'Trend: {slope:.4f}/yr (p={p:.3f})')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(pd.Timestamp("2017-08-01"), color='green', linewidth=1, linestyle=':', label='LM onset (Aug 2017)')
ax.axvline(pd.Timestamp("2018-01-01"), color='purple', linewidth=1, linestyle=':', label='New regime (2018)')
ax.set_ylabel('KEI')
ax.set_title('(a) CMEMS OMI Kuroshio Extension Index (1993–2026)')
ax.legend(loc='upper left', fontsize=8)
ax.set_xlim(pd.Timestamp("1993-01-01"), pd.Timestamp("2026-06-01"))

# Panel (b): 滑动窗口稳定态占比
ax = axes[1]
ax.plot(pd.to_datetime(center_times), stable_fractions * 100, 'k-o', markersize=3)
ax.axhline(50, color='gray', linewidth=0.5, linestyle='--')
ax.fill_between(pd.to_datetime(center_times), stable_fractions * 100, 50,
                where=stable_fractions > 0.5, color='#d73027', alpha=0.3)
ax.fill_between(pd.to_datetime(center_times), stable_fractions * 100, 50,
                where=stable_fractions <= 0.5, color='#4575b4', alpha=0.3)
ax.set_ylabel('Stable State Fraction (%)')
ax.set_title(f'(b) Fraction of Stable State in {window_years}-year Sliding Window')
ax.set_ylim(20, 80)

# Panel (c): 10年窗口平均 KEI
ax = axes[2]
ax.plot(pd.to_datetime(center_times), mean_kei_vals, 'k-o', markersize=3)
ax.axhline(0, color='gray', linewidth=0.5)
ax.fill_between(pd.to_datetime(center_times), mean_kei_vals, 0,
                where=np.array(mean_kei_vals) > 0, color='#d73027', alpha=0.3)
ax.fill_between(pd.to_datetime(center_times), mean_kei_vals, 0,
                where=np.array(mean_kei_vals) <= 0, color='#4575b4', alpha=0.3)
ax.set_ylabel('Mean KEI')
ax.set_title(f'(c) {window_years}-year Running Mean KEI')
ax.set_xlabel('Year')

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "fig4_kei_state_analysis.png", dpi=300, bbox_inches='tight')
print(f"\n图已保存: {FIG / 'fig4_kei_state_analysis.png'}")
plt.close()

# ── 7. 保存统计结果 ──
import json
stats = {
    "kei_trend_per_year": float(slope),
    "kei_trend_p_value": float(p),
    "stable_months": n_stable,
    "unstable_months": n_unstable,
    "stable_fraction_percent": round(100 * n_stable / len(kei), 1),
    "n_stable_episodes": len(durations_stable),
    "mean_stable_duration_months": round(np.mean(durations_stable), 1),
    "n_unstable_episodes": len(durations_unstable),
    "mean_unstable_duration_months": round(np.mean(durations_unstable), 1),
}
with open(OUT / "kei_statistics.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"统计结果已保存: {OUT / 'kei_statistics.json'}")
print(json.dumps(stats, indent=2))
