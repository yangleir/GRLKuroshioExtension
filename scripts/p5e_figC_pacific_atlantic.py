"""Fig C: Pacific vs Atlantic wind stress curl trends — side by side maps"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import linregress
from pathlib import Path

FIG = Path("/Users/zhulin/aitest/黑潮延伸体/figures")

def compute_curl_trend(ds, time_dim='valid_time'):
    u10, v10 = ds['u10'], ds['v10']
    lat, lon = ds.latitude.values, ds.longitude.values
    rho_a, Cd = 1.225, 1.3e-3
    wspd = np.sqrt(u10**2 + v10**2)
    tau_x = rho_a * Cd * wspd * u10
    tau_y = rho_a * Cd * wspd * v10
    R = 6.371e6
    dlat = np.abs(np.diff(lat).mean()) * np.pi / 180
    dlon = np.abs(np.diff(lon).mean()) * np.pi / 180
    cos_lat = np.cos(np.deg2rad(ds.latitude))
    curl = (tau_y.differentiate('longitude') / (R * cos_lat * dlon * (180/np.pi))
            - tau_x.differentiate('latitude') / (R * dlat * (180/np.pi)))
    curl_ann = curl.resample({time_dim: 'YE'}).mean()
    vals = curl_ann.values
    ny, nx = len(lat), len(lon)
    trend = np.full((ny, nx), np.nan)
    years = np.arange(vals.shape[0], dtype=float)
    for i in range(ny):
        for j in range(nx):
            y = vals[:, i, j]
            valid = ~np.isnan(y)
            if valid.sum() > 10:
                sl, _, _, _, _ = linregress(years[valid], y[valid])
                trend[i, j] = sl * 10
    clim = curl.mean(dim=time_dim).values
    return lat, lon, trend, clim

print("计算太平洋风应力旋度趋势...")
ds_pac = xr.open_dataset("/Users/zhulin/aitest/黑潮延伸体/data/era5_monthly_wind_npac_1993_2025.nc")
lat_p, lon_p, trend_p, clim_p = compute_curl_trend(ds_pac)

print("计算大西洋风应力旋度趋势...")
ds_atl = xr.open_dataset("/Users/zhulin/aitest/黑潮延伸体/data/era5_monthly_wind_atlantic_1993_2025.nc")
lat_a, lon_a, trend_a, clim_a = compute_curl_trend(ds_atl)

# Plot
fig = plt.figure(figsize=(16, 5))
axes = [
    fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree(central_longitude=180)),
    fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree(central_longitude=-40)),
]

clevels = np.linspace(-3e-9, 3e-9, 25)

# Pacific
ax = axes[0]
lon2d_p, lat2d_p = np.meshgrid(lon_p, lat_p)
cs = ax.contourf(lon2d_p, lat2d_p, trend_p, levels=clevels, cmap='RdBu_r', extend='both',
                 transform=ccrs.PlateCarree())
ax.contour(lon2d_p, lat2d_p, clim_p, levels=[0], colors='black', linewidths=2,
           transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.LAND, color='lightgray')
ax.set_extent([120, 240, 10, 60], crs=ccrs.PlateCarree())
ax.set_title('(a) North Pacific', fontsize=13)
ax.plot([142,170,170,142,142], [30,30,40,40,30], 'k-', linewidth=2, transform=ccrs.PlateCarree())

# Atlantic
ax = axes[1]
lon2d_a, lat2d_a = np.meshgrid(lon_a, lat_a)
cs2 = ax.contourf(lon2d_a, lat2d_a, trend_a, levels=clevels, cmap='RdBu_r', extend='both',
                  transform=ccrs.PlateCarree())
ax.contour(lon2d_a, lat2d_a, clim_a, levels=[0], colors='black', linewidths=2,
           transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.LAND, color='lightgray')
ax.set_extent([-80, 0, 10, 60], crs=ccrs.PlateCarree())
ax.set_title('(b) North Atlantic', fontsize=13)
ax.plot([-75,-45,-45,-75,-75], [33,33,47,47,33], 'k-', linewidth=2, transform=ccrs.PlateCarree())

fig.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Wind stress curl trend (N/m³ per decade)', fontsize=11)

fig.suptitle('Wind Stress Curl Trend (1993-2025): Pacific vs Atlantic', fontsize=14, y=0.98)
plt.savefig(FIG / "figC_pacific_atlantic_wind_v2.png", dpi=300, bbox_inches='tight')
print(f"保存: {FIG / 'figC_pacific_atlantic_wind_v2.png'}")
plt.close()

import shutil
shutil.copy(FIG / "figC_pacific_atlantic_wind_v2.png",
            "/Users/zhulin/aitest/黑潮延伸体/manuscript/figC.png")
print("已复制到 manuscript/figC.png")
