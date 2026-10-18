# Changelog

## v1.0.0 (release accompanying the second revision of JAMC-D-26-0098)

First public release. All datasets, results and figures of the paper are rebuilt from `data/raw/` by
`run_all.py`.

### Differences from the analyses of the first revised manuscript

1. **Soil heat capacity.** The storage correction of the ground heat flux uses the three-phase form
   Cs = (1 − φ)C_m + SWC·C_w + (φ − SWC)C_a with φ = 0.44 (de Vries 1963). The earlier form,
   Cs = (1.94 + 4.18·SWC) × 10⁶ J m⁻³ K⁻¹, omitted the solid volume fraction. The coefficients g1–g3 of
   Eq. (5) are refitted (`calibration/fit_ground_heat_flux.py`).
2. **Ground heat flux in the closure analysis.** Figs. 2, 4, 6 and A5 and Table 2 use the storage-corrected
   surface ground heat flux, as the model does. With the heat-flux-plate signal the conventional closure
   slope is 0.662; with the surface flux it is 0.723.
3. **Albedo constants.** α_noon and Δα of Eq. (3) are fitted to the CNR4 albedo record
   (0.098 and 0.0065, `calibration/fit_albedo.py`), in place of the previously prescribed values of 0.115
   and 0.02.
4. **Concrete storage.** The effective depth d_c = 0.04 m and the sensitivity of the closure to it are
   reported (`analysis/observational_statistics.py`, `concrete_depth`).
5. **Model runs.** Cross-validation, Bowen-ratio map, management scenarios and parameter sensitivity are
   rerun with items 1 to 3. The full-record sweep of the array-roughness multiplier is flat around the value
   used (`results/calibration/roughness_multiplier_sweep.json`), and the training-year sweep of the hold-out
   test selects the same value.
6. **Independent hold-out test** (new): `models/holdout_lsm.py`, `models/holdout_ml.py` (Table A2).
7. **Record coverage** (new): `analysis/record_coverage.py` documents the daytime window (time stamps
   06:00 to 18:30 LST), the soil-water range, the periods of the Fig. A2 series and the panel
   air-temperature contrast of Table 4 (note f), and also the air temperature and precipitation of 2024 and 2025
   (Section 2a, Fig. 2 caption) and the diurnal amplitudes of the panel and concrete temperatures (Section 2d).
8. **Reference height of the wind profile.** z is now the height of the wind measurement, 4.7 m (previously
   10 m). z enters the neutral aerodynamic resistance only through a constant factor, so the array-roughness
   multiplier is rescaled from 1.1 to 1.57 (1.1 × 1.425) and the swept values from 0.9–1.4 to 1.28–2.00; the
   modeled fluxes and skill scores are unchanged. The friction velocity of Fig. 7 and Table 2 rises by 24 %,
   and the elasticities for z0m and d in Fig. A4 change slightly (H: z0m 0.025 to 0.037, d 0.001 to 0.003).
9. **Radiosonde sounding** (new): `data/raw/radiosonde_FLSF_2025-10-15_1435LST.txt` and
   `analysis/boundary_layer_height.py` (convective boundary-layer depth H_BL of Eq. A1, parcel method).
10. **Solver diagnostics** (new): `analysis/lsm_diagnostics.py` reports the energy-balance residual left by the
   Newton–Raphson solver (99.3 % of the records within 0.1 W m⁻²) and the share of calm-wind records
   (2.8 %) for which the wind speed is set to 2.5 m s⁻¹.

| Quantity | First revision | This release |
|---|---|---|
| LSM H R² / RMSE (cross-validation) | 0.89 / 35.3 W m⁻² | 0.90 / 34.7 W m⁻² |
| LSM LE R² / RMSE (cross-validation) | 0.66 / 33.3 W m⁻² | 0.64 / 34.1 W m⁻² |
| LSM G R² / RMSE (cross-validation) | 0.80 / 19.7 W m⁻² | 0.81 / 15.5 W m⁻² |
| Conventional / augmented closure slope | 0.662 / 0.839 | 0.723 / 0.937 |
| Annual β reduction, albedo / SWC / combined scenario | 2.1 / 3.8 / 5.8 % | 2.7 / 3.6 / 6.2 % |
