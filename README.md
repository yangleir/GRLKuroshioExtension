# Poleward Shift of the Kuroshio Extension Jet: Disentangling Displacement from Intensification

Analysis code and derived statistics for:

> Yang, L., Liu, Y., & Zhu, L. (2026). *Poleward Shift of the Kuroshio Extension Jet: Disentangling Displacement from Intensification.* Submitted to Geophysical Research Letters.

Using 31 years (1993–2024) of satellite altimetry with three independent jet-tracking methods, the paper shows that the Kuroshio Extension jet is simultaneously displacing poleward (+0.21°/decade, velocity centroid) and intensifying asymmetrically on its poleward flank, that the Gulf Stream shows no corresponding shift, and that sea-level-anomaly-based tracking methods are biased against the true migration by differential sea level rise (a "self-concealing" observational bias).

## Script → figure/statistic mapping

| Paper item | Script | Output |
|---|---|---|
| Figure 1 (KE velocity tracking, 3 panels) | `scripts/p12_regenerate_fig1.py` | `fig1_ke_velocity_v3.*` |
| Figure 2 (Gulf Stream control) | `scripts/p16_fig7_corrected.py` | `fig2_gs_velocity_v5.*`, `outputs/gs_axis_unified.nc` |
| Figure 3 (self-concealing mechanism profiles) | `scripts/p9c_figB_front_fix.py` | `figB_v3.*` |
| Figure 4 (anomaly-method comparison, monthly SLA) | `scripts/p2b_method_comparison.py` | `fig2_method_comparison.png`, `outputs/method_comparison_stats.json` |
| Figure 5 (three tracking methods, unified period) | `scripts/p10_unified_velocity.py` | `fig_unified_three_methods.*`, `outputs/unified_velocity_stats.json` |
| Figure S1 (Coriolis-mediated adjustment framework) | `scripts/p13_coriolis_framework.py` | `figS1.*`, `outputs/coriolis_framework_stats.json` |
| Figure S2 (ADT vs SLA gradient tracking) | `scripts/p17_adt_vs_sla_gradient.py` | `figS2_adt_vs_sla.*`, `outputs/adt_vs_sla_corrected.json` |
| Autocorrelation-corrected p-values (GS, SLA-weighted, monthly gradient) | `scripts/p15_corrected_pvalues.py` | `outputs/corrected_pvalues.json` |
| Wind stress curl / zero-curl line | `scripts/p4_wind_curl_analysis.py` | `outputs/wind_curl_stats.json` |
| Sverdrup gyre boundary trend | `scripts/p7_centroid_sverdrup.py` | `outputs/sverdrup_stats.json` |
| Endpoint sensitivity | `scripts/p2c_endpoint_sensitivity.py` | `outputs/endpoint_sensitivity_stats.json` |

Other `p*.py` scripts are earlier exploratory versions retained for provenance; the table above lists the scripts that produce the figures and statistics quoted in the manuscript.

Trend significance throughout uses ordinary least squares with the effective-degrees-of-freedom correction of Bretherton et al. (1999) (`bretherton_trend()` in `p10`, `p12`, `p15`, `p16`).

## Derived data included here

`outputs/` contains the small derived statistics (JSON) and jet-axis time series (NetCDF) underlying the numbers quoted in the paper, so the headline results can be inspected without re-processing the altimetry archive.

## Source data (not redistributed)

| Dataset | Source | ID |
|---|---|---|
| DUACS L4 daily 0.25° (MY, 1993–2021) | [Copernicus Marine Service](https://marine.copernicus.eu) | `SEALEVEL_GLO_PHY_L4_MY_008_047` |
| DUACS L4 daily 0.25° (NRT, 2022–2024) | Copernicus Marine Service | `SEALEVEL_GLO_PHY_L4_NRT_008_046` |
| DUACS L4 monthly 0.125° SLA (1993–2025) | Copernicus Marine Service | `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m` |
| ERA5 monthly 10-m winds | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu) | `reanalysis-era5-single-levels-monthly-means` |

Scripts read the altimetry archive from a local disk (`/Volumes/Backup Plus/ssh/...`); adjust the `ROOT`/`DRIVE` path constants at the top of each script to your local data location.

## Environment

Python ≥ 3.9 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## License

MIT — see `LICENSE`.
