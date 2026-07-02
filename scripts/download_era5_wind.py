"""Download ERA5 monthly mean wind stress for KE wind attribution
Variables: 10m u/v wind, mean sea level pressure
Region: North Pacific (120-240E, 10-60N)
Period: 1993-2025
"""
import cdsapi

c = cdsapi.Client()

out_path = "/Users/zhulin/aitest/黑潮延伸体/data/era5_monthly_wind_npac_1993_2025.nc"

print("下载 ERA5 月均风场数据...")
print("区域: 60N/120E/10N/240E (North Pacific)")
print("时间: 1993-2025")

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': ['monthly_averaged_reanalysis'],
        'variable': [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'mean_sea_level_pressure',
        ],
        'year': [str(y) for y in range(1993, 2026)],
        'month': [f'{m:02d}' for m in range(1, 13)],
        'time': ['00:00'],
        'area': [60, 120, 10, -120],
        'data_format': 'netcdf',
    },
    out_path
)

print(f"下载完成: {out_path}")
