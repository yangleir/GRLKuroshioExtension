"""Recalculate global SLA trend using monthly 0.125° data (1993-2025)
Replaces the old daily-data version (1993-2021, 0.25°)
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import linregress
from numba import njit, prange
from pathlib import Path
import pandas as pd

OUT = Path("/Users/zhulin/aitest/黑潮延伸体/output")
FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")

data_path = "/Volumes/Backup Plus/ssh/cmems_sla_monthly_global_0.125deg_1993_2025.nc"
ds = xr.open_dataset(data_path)
print(f"数据: {len(ds.time)} 月, {len(ds.latitude)}x{len(ds.longitude)}")
print(f"时间: {str(ds.time.values[0])[:10]} → {str(ds.time.values[-1])[:10]}")

sla = ds['sla'].values.astype(np.float64)
lat = ds.latitude.values
lon = ds.longitude.values
time_vals = ds.time.values
ntime, nlat, nlon = sla.shape

time_years = np.array([(pd.Timestamp(t) - pd.Timestamp("1993-01-01")).days / 365.25 for t in time_vals]).astype(np.float64)

mx = np.mean(time_years)
dx = time_years - mx
x_var = np.dot(dx, dx)

print(f"计算 {nlat}x{nlon} 网格点的 SLA 线性趋势...")

@njit(parallel=True)
def compute_trend(data, dx, x_var, ntime, nlat, nlon):
    slope = np.empty((nlat, nlon), dtype=np.float64)
    for i in prange(nlat):
        for j in range(nlon):
            count = 0
            sum_y = 0.0
            for t in range(ntime):
                v = data[t, i, j]
                if not np.isnan(v):
                    sum_y += v
                    count += 1
            if count < 30:
                slope[i, j] = np.nan
                continue
            my = sum_y / count
            num = 0.0
            for t in range(ntime):
                v = data[t, i, j]
                if not np.isnan(v):
                    num += dx[t] * (v - my)
            slope[i, j] = num / x_var if x_var != 0.0 else np.nan
    return slope

slope_per_year = compute_trend(sla, dx, x_var, ntime, nlat, nlon)
trend_mm_yr = slope_per_year * 1000.0

print(f"趋势范围: {np.nanmin(trend_mm_yr):.2f} ~ {np.nanmax(trend_mm_yr):.2f} mm/yr")
print(f"全球均值: {np.nanmean(trend_mm_yr):.2f} mm/yr")

# 保存
trend_da = xr.DataArray(
    trend_mm_yr,
    coords={"latitude": lat, "longitude": lon},
    dims=["latitude", "longitude"],
    name="sla_trend",
    attrs={"units": "mm/year", "long_name": "SLA linear trend (1993-2025)", "data": "CMEMS 0.125deg monthly"},
)
trend_da.to_dataset().to_netcdf(OUT / "global_sla_trend_monthly_1993_2025.nc")
print(f"保存: {OUT / 'global_sla_trend_monthly_1993_2025.nc'}")

# 经度转换（-180~180 → 0~360 用于绘图）
lon_plot = lon.copy()
if lon_plot.min() < 0:
    lon_plot = lon_plot % 360
    idx = np.argsort(lon_plot)
    lon_plot = lon_plot[idx]
    trend_plot = trend_mm_yr[:, idx]
else:
    trend_plot = trend_mm_yr

# 绘图
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=180))

lon2d, lat2d = np.meshgrid(lon_plot, lat)
clevels = np.linspace(-10, 10, 41)
cs = ax.contourf(
    lon2d, lat2d, trend_plot,
    levels=clevels, cmap='RdBu_r', extend='both',
    transform=ccrs.PlateCarree(),
)
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.LAND, color='lightgray', zorder=2)
ax.set_global()

# KE 区域框
import matplotlib.patches as mpatches
ke_lon = [142, 170, 170, 142, 142]
ke_lat = [30, 30, 40, 40, 30]
ax.plot(ke_lon, ke_lat, 'k-', linewidth=2, transform=ccrs.PlateCarree(), zorder=3)

cbar = plt.colorbar(cs, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
cbar.set_label("SLA Trend (mm/year)", fontsize=12)

ax.set_title("Global Sea Level Anomaly Trend (1993-2025, CMEMS 0.125° Monthly)", fontsize=14)

out_png = FIG / "fig5_global_sla_trend_monthly.png"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"图已保存: {out_png}")
plt.close()

# 也复制到 manuscript/
import shutil
shutil.copy(out_png, "/Users/zhulin/aitest/黑潮延伸体/manuscript/fig5.png")
print("已复制到 manuscript/fig5.png")
