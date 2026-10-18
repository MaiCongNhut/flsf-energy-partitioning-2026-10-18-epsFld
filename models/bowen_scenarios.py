"""
Seasonal response to the management scenarios (Section 3f, Section 4c, Appendix Eq. A1).

Uses the same calibrated physics as the Bowen-ratio map (models/bowen_map.py). For each season:
  baseline     observed seasonal median albedo and mean soil water content
  albedo       albedo of the non-panel ground raised to 0.20
  irrigation   soil water content raised to 0.40 m3 m-3
  combined     both
Outputs per season and annual mean (mean of the four seasons):
  beta and its reduction (%), beta = mean of per-record H/LE (LE > 20 W m-2, clipped to [-2, 8])
  dH   midday reduction in H: mean modeled H of the 12:00 and 12:30 LST records, baseline minus scenario
  dTa  Eq. A1, dTa = dH * dt / (rho_cp * H_BL), dt = 6 h, rho_cp = 1200 J m-3 K-1, H_BL = 1260 m
  dPV  efficiency gain = 0.45 % per K of cooling (Skoplaki and Palyvos 2009)
Output: results/bowen_scenarios/bowen_scenarios.json, .csv
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths, config as C
from preprocessing import load_observed
from calibration import calibrate_and_predict
import bowen_map as BW

DT_PEAK, RHO_CP, H_BL, PV_COEF = 6 * 3600.0, 1200.0, C.H_BL, 0.45
SCEN = {'albedo': (0.20, None), 'irrigation': (None, 0.40), 'combined': (0.20, 0.40)}


def fluxes(sub, albedo, swc, s, p, ac):
    a = BW._prep(sub, p); rc = BW._rc(a, swc, p, s); H, LE = BW._HLE(a, albedo, rc, p)
    return (H + ac['aH'] * a['sinh'] + ac['bH'] * a['cosh'] + ac['cH'],
            LE + ac['aL'] * a['sinh'] + ac['bL'] * a['cosh'] + ac['cL'])


def beta(H, LE):
    m = LE > 20
    return float(np.nanmean(np.clip(H[m] / LE[m], -2.0, 8.0)))


def main():
    df, _ = load_observed(paths.OBSERVED, G_TARGET='surface'); n = len(df); pt = C.params_template()
    fit = calibrate_and_predict(df, np.arange(n), np.arange(n), pt)
    p = {**pt, 'rc_base_by_season': fit['rc_by_season'], 'use_diurnal_albedo': False}; ac = fit['autocal']
    df['alpha'] = (df['Fsu'] / df['Fsd']).clip(0, 1); bright = df[df['Fsd'] > 50]
    rows = []
    for s in C.SEASON_CODES:
        sub = bright[bright['Season'] == s]; noon = sub['Hour'].values == 12
        a0, w0 = float(sub['alpha'].median()), float(sub['SWC'].mean())
        H0, L0 = fluxes(sub, a0, w0, s, p, ac); b0 = beta(H0, L0)
        row = {'season': s, 'N': int(len(sub)), 'N_noon': int(noon.sum()), 'albedo_baseline': a0, 'swc_baseline': w0,
               'beta_baseline': b0, 'H_noon_baseline': float(np.nanmean(H0[noon]))}
        for tag, (al, sw) in SCEN.items():
            H, L = fluxes(sub, a0 if al is None else al, w0 if sw is None else sw, s, p, ac)
            dH = float(np.nanmean(H0[noon]) - np.nanmean(H[noon])); dT = dH * DT_PEAK / (RHO_CP * H_BL)
            row.update({'beta_' + tag: beta(H, L), 'beta_reduction_pct_' + tag: 100 * (b0 - beta(H, L)) / b0,
                        'dH_noon_' + tag: dH, 'dTa_' + tag: dT, 'dPV_pct_' + tag: PV_COEF * dT})
        rows.append(row)
    out = pd.DataFrame(rows)
    annual = {c: float(out[c].mean()) for c in out.columns if c.startswith(('beta_', 'dH_', 'dTa_', 'dPV_'))}
    annual['beta_reduction_pct_of_annual_mean'] = {t: 100 * (annual['beta_baseline'] - annual['beta_' + t]) / annual['beta_baseline'] for t in SCEN}
    json.dump({'seasons': rows, 'annual': annual,
               'constants': {'dt_s': DT_PEAK, 'rho_cp': RHO_CP, 'H_BL_m': H_BL, 'pv_coef_pct_per_K': PV_COEF}},
              open(paths.out('bowen_scenarios', 'bowen_scenarios.json'), 'w'), indent=2)
    out.to_csv(paths.out('bowen_scenarios', 'bowen_scenarios.csv'), index=False)
    for r in rows:
        print('%s beta %.3f -> %.3f (%.1f %%)  dH %.1f  dTa %.2f  dPV %.2f %%' % (
            r['season'], r['beta_baseline'], r['beta_combined'], r['beta_reduction_pct_combined'],
            r['dH_noon_combined'], r['dTa_combined'], r['dPV_pct_combined']))
    print('annual beta %.3f -> %.3f (%.1f %% of the annual mean); albedo %.1f %%, irrigation %.1f %%; dH %.1f  dTa %.2f  dPV %.2f %%' % (
        annual['beta_baseline'], annual['beta_combined'], annual['beta_reduction_pct_of_annual_mean']['combined'],
        annual['beta_reduction_pct_of_annual_mean']['albedo'], annual['beta_reduction_pct_of_annual_mean']['irrigation'],
        annual['dH_noon_combined'], annual['dTa_combined'], annual['dPV_pct_combined']))


if __name__ == '__main__':
    main()
