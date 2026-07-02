"""P4: Wind stress curl trend and zero-curl line tracking
- Compute wind stress from ERA5 u10/v10 using bulk formula
- Calculate wind stress curl
- Track zero-curl line latitude (annual mean)
- Compare with KE axis position
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from pathlib import Path
import pandas as pd
import json

OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")

# ── 1. 读取 ERA5 ──
ds = xr.open_dataset("/Users/zhulin/aitest/黑潮延伸体/data/era5_monthly_wind_npac_1993_2025.nc")
u10 = ds['u10']
v10 = ds['v10']
lat = ds.latitude.values
lon = ds.longitude.values
time = ds.valid_time.values

print(f"ERA5: {len(time)} 月, lat {lat.min()}-{lat.max()}, lon {lon.min()}-{lon.max()}")

# ── 2. 计算风应力（bulk formula） ──
# τ = ρ_a * C_d * |U| * U
# ρ_a = 1.225 kg/m³, C_d = 1.3e-3 (simplified)
rho_a = 1.225
Cd = 1.3e-3

wspd = np.sqrt(u10**2 + v10**2)
tau_x = rho_a * Cd * wspd * u10  # Pa
tau_y = rho_a * Cd * wspd * v10  # Pa
print("风应力计算完成")

# ── 3. 计算风应力旋度 curl(τ) = ∂τ_y/∂x - ∂τ_x/∂y ──
# 转换为米
R = 6.371e6
dlat = np.abs(np.diff(lat).mean()) * np.pi / 180
dlon = np.abs(np.diff(lon).mean()) * np.pi / 180

# ∂τ_y/∂x
cos_lat = np.cos(np.deg2rad(ds.latitude))
dtau_y_dx = tau_y.differentiate('longitude') / (R * cos_lat * dlon * (180/np.pi))
# ∂τ_x/∂y
dtau_x_dy = tau_x.differentiate('latitude') / (R * dlat * (180/np.pi))

curl_tau = dtau_y_dx - dtau_x_dy
print(f"风应力旋度范围: {float(curl_tau.min()):.2e} ~ {float(curl_tau.max()):.2e} N/m³")

# ── 4. 年均风应力旋度趋势 ──
curl_annual = curl_tau.resample(valid_time='YE').mean()
time_annual = curl_annual.valid_time.values
years = np.array([pd.Timestamp(t).year for t in time_annual])
time_years = years - years[0]

# 逐格点趋势
print("计算风应力旋度逐格点趋势...")
curl_vals = curl_annual.values
ny, nx = len(lat), len(lon)
trend_curl = np.full((ny, nx), np.nan)

for i in range(ny):
    for j in range(nx):
        y = curl_vals[:, i, j]
        valid = ~np.isnan(y)
        if valid.sum() > 10:
            sl, _, _, p, _ = linregress(time_years[valid], y[valid])
            trend_curl[i, j] = sl * 10  # per decade

# ── 5. 零线追踪 ──
# 在 130-180°E 范围内，找年均 curl(τ) = 0 的纬度
lon_range = (lon >= 130) & (lon <= 180)
curl_region = curl_annual.sel(longitude=lon[lon_range])

zero_lat = np.full(len(time_annual), np.nan)
for t in range(len(time_annual)):
    lats_zero = []
    for j in range(curl_region.shape[2]):
        col = curl_region.values[t, :, j]
        valid = ~np.isnan(col)
        if valid.sum() < 10:
            continue
        # 找符号变化点（从正到负，即零线位置）
        for i in range(len(lat) - 1):
            if col[i] > 0 and col[i+1] <= 0 and lat[i] > 20 and lat[i] < 50:
                # 线性插值
                frac = col[i] / (col[i] - col[i+1])
                zero_l = lat[i] + frac * (lat[i+1] - lat[i])
                lats_zero.append(zero_l)
                break
    if len(lats_zero) >= 5:
        zero_lat[t] = np.median(lats_zero)

valid_z = ~np.isnan(zero_lat)
if valid_z.sum() > 5:
    sl_z, ic_z, r_z, p_z, _ = linregress(time_years[valid_z], zero_lat[valid_z])
    print(f"\n零线纬度趋势: {sl_z*10:+.3f}°/decade, p={p_z:.4f}")
    print(f"零线纬度范围: {np.nanmin(zero_lat):.2f} ~ {np.nanmax(zero_lat):.2f}°N")

# ── 6. 加载 KE 轴位置对比 ──
ke_ds = xr.open_dataset(OUT / 'ke_axis_position.nc')
ke_annual = ke_ds['ke_axis_latitude'].resample(time='YE').mean()
ke_time = ke_annual.time.values
ke_vals = ke_annual.values

# ── 7. 绘图 ──
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# (a) 风应力旋度趋势空间分布
ax = axes[0]
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ax.remove()
ax = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
lon2d, lat2d = np.meshgrid(lon, lat)
clevels = np.linspace(-3e-9, 3e-9, 25)
cs = ax.contourf(lon2d, lat2d, trend_curl, levels=clevels, cmap='RdBu_r', extend='both',
                 transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.LAND, color='lightgray')
ax.set_extent([120, 240, 10, 60], crs=ccrs.PlateCarree())
cbar = plt.colorbar(cs, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
cbar.set_label('Wind stress curl trend (N/m³ per decade)')
ax.set_title('(a) ERA5 Wind Stress Curl Trend (1993-2025)')

# 叠加零线气候态
curl_clim = curl_tau.mean(dim='valid_time')
ax.contour(lon2d, lat2d, curl_clim.values, levels=[0], colors='black', linewidths=2,
           transform=ccrs.PlateCarree())

# (b) 零线纬度时间序列
ax = axes[1]
dates_annual = pd.to_datetime(time_annual)
ax.plot(dates_annual, zero_lat, 'ko-', markersize=4, linewidth=1, label='Zero-curl line latitude')
if valid_z.sum() > 5:
    ax.plot(dates_annual, sl_z * time_years + ic_z, 'r--', linewidth=1.5,
            label=f'Trend: {sl_z*10:+.3f}°/dec (p={p_z:.3f})')
ax.set_ylabel('Latitude (°N)')
ax.set_title('(b) Wind Stress Curl Zero Line Latitude (130-180°E)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) 零线 vs KE 轴对比
ax = axes[2]
# 标准化
z_norm = (zero_lat - np.nanmean(zero_lat)) / np.nanstd(zero_lat)
ke_norm = (ke_vals - np.nanmean(ke_vals)) / np.nanstd(ke_vals)
ax.plot(dates_annual, z_norm, 'r-o', markersize=4, label='Zero-curl line (normalized)')
ke_dates = pd.to_datetime(ke_time)
ax.plot(ke_dates, ke_norm, 'b-o', markersize=4, label='KE axis lat (normalized)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('Standardized Index')
ax.set_xlabel('Year')
ax.set_title('(c) Wind Stress Curl Zero Line vs KE Axis Position')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

for a in axes[1:]:
    a.xaxis.set_major_locator(mdates.YearLocator(5))
    a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(FIG / "fig4_wind_curl_analysis.png", dpi=300, bbox_inches='tight')
print(f"\n图已保存: {FIG / 'fig4_wind_curl_analysis.png'}")
plt.close()

# ── 8. 保存统计 ──
stats = {
    "zero_line_trend_deg_per_decade": round(sl_z * 10, 4) if valid_z.sum() > 5 else None,
    "zero_line_trend_p_value": round(p_z, 5) if valid_z.sum() > 5 else None,
    "zero_line_mean_lat": round(float(np.nanmean(zero_lat)), 2),
}
with open(OUT / "wind_curl_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
