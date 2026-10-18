# Datasets

All datasets are built by `data/build_datasets.py` from `data/raw/FLSF_ObservedData.csv`
(half-hourly, 1 January 2024 to 1 February 2026, 36,624 records; missing values coded −9999).

| Step | Records | Used for |
|---|---|---|
| Daytime records after the commissioning period (time stamps 06:00–18:30 LST, Fsd > 0, from 1 May 2024) | 16,323 | CNN input series (`cnn_daytime_series.csv`, with the flag `QC_bad`) |
| … with all six model drivers present (Fsd, Fld, Ta, RH, Ws, SWC) | 16,188 | `daytime_record.csv` |
| … with Fsd > 50 W m⁻²: flux-characterization (FC) dataset | 12,334 | Figs. 3, 4, 6, 7, A5, A6; Table 2; closure statistics |
| FC records with LE > 20 W m⁻² | 11,479 | Bowen ratio (Section 3c) and Fig. 5 |
| Modeling dataset: daytime records after the modeling screening (physical limits, per-season double-difference spike test on H, LE and G, implausible midday records; 151 dew records kept) | 14,421 | LSM and PCR calibration and evaluation, Figs. 8, 9, A4, A7; Table 3 |
| Modeling dataset with Fsd > 50 W m⁻² | 11,340 | albedo fit (Eq. 3), Bowen-ratio map and scenarios |
| CNN windows scored in cross-validation (24 consecutive daytime records, windows across fold boundaries and flagged targets removed) | 13,923 | Table 3 (CNN) |
| Hold-out split of the modeling dataset: training 1 May 2024–30 April 2025 / test 1 May 2025–1 February 2026 | 8,077 / 6,344 | Table A2 |

Screening counts of the modeling dataset (`results/lsm_cv/metrics.json`, field `qc`): physical limits on LE 11,
spikes H 664, LE 355, G 903 (overlapping), implausible midday records 36;
1,782 records removed in total (a record flagged by more than one test is counted once).

## Columns added to the raw record

| Column | Definition |
|---|---|
| `SWC` | mean of the two 10-cm probes, Sws_10cma and Sws_10cmb (m³ m⁻³) |
| `Cs` | soil volumetric heat capacity, (1 − φ)C_m + SWC·C_w + (φ − SWC)C_a, φ = 0.44 (J m⁻³ K⁻¹) |
| `dTs_dt` | tendency of the 10-cm soil temperature `Ts` between consecutive records of the same day (K s⁻¹) |
| `Fg_surface` (`Gobs` in the modeling dataset) | storage-corrected surface ground heat flux, Fg + Cs·0.10·dTs_dt (W m⁻²) |
| `albedo` | Fsu/Fsd |

Note: in the raw file the column `Ts` is the measured soil temperature at 10 cm (TDR-315H), called T_soil in
the paper; T_s in the paper is the modeled surface temperature.
