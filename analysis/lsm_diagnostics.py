"""
Diagnostics of the land surface model quoted in Sections 2d, 3c, 3d and 3f.

  calibration_stability  spread of the per-season base canopy resistance and of the diurnal-correction
                         coefficients among the five cross-validation folds (results/lsm_cv/metrics.json)
  fig9                   seasonal half-hourly composites of the out-of-fold predictions: midday (10:00-14:30 LST)
                         H residual of the mean; G residual of the mean at 09:00-11:00 LST (minimum) and
                         12:00-15:00 LST (maximum); largest G residual relative to the observed interquartile range
                         at 08:00-17:00 LST; range of the LE residual of the mean; share of records with observed G < 0
  modeled_closure        OLS of modeled H + LE on the observed augmented available energy (FC dataset)
  bowen_map              range of each seasonal beta surface and local derivatives at the observed point
  holdout_mae, cv_mae    MAE of the LSM on the hold-out test period (Table A2) and out of fold (Table 3)
  solver                 energy-balance residual left by the Newton-Raphson solver (full-record calibration):
                         share of records closed to within the tolerance NR_TOL (Section 3c), and the share of
                         records whose wind speed is replaced by U_DEFAULT (Ws <= 0.3 m s-1, Section 2d)
Output: results/lsm_cv/lsm_diagnostics.json
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'flsf')); sys.path.insert(0, HERE)
import paths
import config as C
import physics
from observational_statistics import fc_dataset

SEASONS = ['MAM', 'JJA', 'SON', 'DJF']


def calibration_stability(m):
    folds = m['folds']
    out = {'rc_base': {}, 'diurnal': {}}
    for s in SEASONS:
        v = np.array([folds[k]['rc_base'][s] for k in folds])
        out['rc_base'][s] = {'folds': v.tolist(), 'full_record': m['full_record']['rc_base'][s],
                             'range_over_mean': float((v.max() - v.min()) / v.mean()),
                             'max_dev_from_mean': float(np.max(np.abs(v - v.mean())) / v.mean())}
    out['rc_base_max_range_over_mean'] = max(x['range_over_mean'] for x in out['rc_base'].values())
    for c in ('aH', 'bH', 'cH', 'aL', 'bL', 'cL'):
        v = np.array([folds[k]['diurnal'][c] for k in folds])
        out['diurnal'][c] = {'folds': v.tolist(), 'full_record': m['full_record']['diurnal'][c],
                             'range': float(v.max() - v.min())}
    out['diurnal_max_range'] = max(x['range'] for x in out['diurnal'].values())
    return out


def fig9(p):
    p['hh'] = p['DateTime'].dt.hour + p['DateTime'].dt.minute / 60
    out = {'midday_H_resid_max_abs': {}, 'G_resid_min_0900_1100': {}, 'G_resid_max_1200_1500': {},
           'G_resid_over_IQR_max_0800_1700': {}, 'LE_resid_range': {}}
    for s in SEASONS:
        x = p[p['Season'] == s].groupby('hh')
        mH = x['H_mod'].mean() - x['Fh_obs'].mean()
        out['midday_H_resid_max_abs'][s] = float(mH.loc[10:14.5].abs().max())
        mG = x['G_mod'].mean() - x['Fg_obs'].mean()
        iqr = x['Fg_obs'].quantile(0.75) - x['Fg_obs'].quantile(0.25)
        out['G_resid_min_0900_1100'][s] = float(mG.loc[9:11].min())
        out['G_resid_max_1200_1500'][s] = float(mG.loc[12:15].max())
        out['G_resid_over_IQR_max_0800_1700'][s] = float((mG.abs() / iqr).loc[8:17].max())
        mL = x['LE_mod'].mean() - x['Fe_obs'].mean()
        out['LE_resid_range'][s] = [float(mL.min()), float(mL.max())]
    out['share_G_obs_negative'] = float((p['Fg_obs'] < 0).mean())
    return out


def modeled_closure(p):
    d = fc_dataset().merge(p[['DateTime', 'H_mod', 'LE_mod']], on='DateTime', how='inner')
    ae = d['Fn'] - d['G'] - d['dQc'] - d['Eelec']
    r = stats.linregress(ae, d['H_mod'] + d['LE_mod'])
    o = stats.linregress(ae, d['Fh'] + d['Fe'])
    return {'N': int(len(d)), 'modeled_slope': float(r.slope), 'modeled_R2': float(r.rvalue ** 2),
            'observed_slope_same_rows': float(o.slope), 'observed_R2_same_rows': float(o.rvalue ** 2)}


def bowen_map():
    g = np.load(os.path.join(paths.RESULTS, 'bowen_map', 'bowen_grids.npz'))
    A, S, obs = g['ALB'], g['SWCg'], g['obs']
    out = {}
    for i, s in enumerate(SEASONS):
        Z = g[s]; ai = int(np.argmin(np.abs(A - obs[i][0]))); si = int(np.argmin(np.abs(S - obs[i][1])))
        dS = (Z[si + 2, ai] - Z[si - 2, ai]) / 2.0     # per 0.01 m3 m-3 (grid step 0.005)
        dA = (Z[si, ai + 2] - Z[si, ai - 2]) / 2.0     # per 0.01 albedo
        out[s] = {'beta_min': float(np.nanmin(Z)), 'beta_max': float(np.nanmax(Z)),
                  'obs_alpha': float(obs[i][0]), 'obs_SWC': float(obs[i][1]), 'beta_model_at_obs': float(Z[si, ai]),
                  'dbeta_per_0.01_SWC': float(dS), 'dbeta_per_0.01_alpha': float(dA), 'ratio_SWC_to_alpha': float(dS / dA)}
    return out


def holdout_mae():
    h = pd.read_csv(os.path.join(paths.RESULTS, 'holdout', 'holdout_lsm_predictions.csv'))
    return {f: float((h[f + '_mod'] - h[f + '_obs']).abs().mean()) for f in ('H', 'LE', 'G')}


def solver(m):
    """Residual Rn - H - LE - G - Eelec - dQc after the Newton-Raphson iteration, record by record, with the same
    inputs and record order as physics.run_model."""
    df = pd.read_csv(paths.MODELING)
    p = {**C.params_template(), 'rc_base_by_season': m['full_record']['rc_base']}
    dates = df['Date'].values; Ws = df['WindSpeed'].values; Fn = df['Fn'].values
    res, Rp = np.full(len(df), np.nan), None
    for i in range(len(df)):
        if i > 0 and dates[i] != dates[i - 1]:
            Rp = None
        u = Ws[i] if (not np.isnan(Ws[i]) and Ws[i] > 0.3) else p['u_default']
        r = physics.solve_Ts(df['Fsd'].values[i], df['Fld'].values[i], df['Ta'].values[i] + 273.15,
                             df['RH'].values[i] / 100.0, u, df['SWC'].values[i], p, int(df['Hour'].values[i]),
                             int(df['Month'].values[i]), Fn[i], Rp)
        res[i] = abs(r['residual'])
        Rp = Fn[i] if not np.isnan(Fn[i]) else Rp
    calm = ~(np.isfinite(Ws) & (Ws > 0.3))
    return {'N': int(len(res)), 'tolerance_W_m2': C.NR_TOL, 'max_iterations': C.NR_MAX_ITER,
            'n_residual_at_or_above_tolerance': int((res >= C.NR_TOL).sum()),
            'share_within_tolerance': float((res < C.NR_TOL).mean()), 'max_abs_residual_W_m2': float(res.max()),
            'n_calm_wind_records': int(calm.sum()), 'share_calm_wind_records': float(calm.mean()),
            'u_default_m_s': p['u_default']}


def main():
    m = json.load(open(os.path.join(paths.RESULTS, 'lsm_cv', 'metrics.json')))
    p = pd.read_csv(os.path.join(paths.RESULTS, 'lsm_cv', 'predictions.csv'), parse_dates=['DateTime'])
    res = {'calibration_stability': calibration_stability(m), 'fig9': fig9(p.copy()),
           'modeled_closure': modeled_closure(p), 'bowen_map': bowen_map()}
    res['holdout_mae'] = holdout_mae()
    res['solver'] = solver(m)
    res['cv_mae'] = {f: float((p[m] - p[o]).abs().mean()) for f, o, m in
                     (('H', 'Fh_obs', 'H_mod'), ('LE', 'Fe_obs', 'LE_mod'), ('G', 'Fg_obs', 'G_mod'))}
    json.dump(res, open(paths.out('lsm_cv', 'lsm_diagnostics.json'), 'w'), indent=2)
    print(json.dumps(res, indent=1))
    print('wrote', paths.out('lsm_cv', 'lsm_diagnostics.json'))


if __name__ == '__main__':
    main()
