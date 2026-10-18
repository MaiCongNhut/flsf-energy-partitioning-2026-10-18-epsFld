#!/usr/bin/env python3
"""
Independent chronological hold-out test of the process-based land surface model.

Purpose
-------
The season-stratified five-fold cross-validation re-estimates only the base canopy resistance
and the diurnal residual correction in each fold, while the other coefficients are set once
against the full record. This script builds a test in which one period is withheld from every
fitting step (Section 2f).

Design
------
  Training period : 1 May 2024 00:00 to 30 Apr 2025 23:30   (one complete annual cycle)
  Test period     : 1 May 2025 00:00 to  1 Feb 2026 23:30   (never used for any fitting)

Every flux-calibrated coefficient is re-derived from the training period alone:
  * per-season ground-heat coefficients g1, g2, g3 (Eq. 5), ordinary least squares of the
    storage-corrected surface ground heat flux on [Rn, dRn/dt, 1], exactly as in
    calibration/fit_ground_heat_flux.py but restricted to training rows;
  * the albedo constants of Eq. (3), refitted to the training-period radiometer data;
  * the panel-array roughness multiplier f_m, re-selected by the same sweep used for the
    full-record value (minimum training RMSE of H), over config.FM_GRID (1.28 to 2.00);
  * per-season base canopy resistance (bounded Brent, minimum training RMSE of LE);
  * the diurnal residual correction (OLS on [sin, cos, 1] of training residuals).
Quantities that are not fitted to the eddy-covariance fluxes are kept: physical constants,
eta (metered plant generation) and the concrete-storage coefficients (fitted to the pile
temperature record).

The canopy stress-function slopes were set once against the full record by binned residual
analysis. Their effect on the test skill is bounded by repeating the test with every slope
set to zero (stress functions switched off).

Output: results/holdout/holdout_lsm.json, holdout_lsm_predictions.csv
Usage : python models/holdout_lsm.py sweep     (stage 1: f_m sweep; always recomputed and cached)
        python models/holdout_lsm.py           (stage 2: test; uses the cached sweep, or runs it if absent)
"""
import os, sys, json, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths, config as C
from preprocessing import load_observed
from calibration import calibrate_and_predict, metrics

from scipy.optimize import least_squares
OBS = paths.OBSERVED
OUT = os.path.dirname(paths.out('holdout', 'x'))
SPLIT = pd.Timestamp('2025-05-01 00:00')
SEAS = ['MAM', 'JJA', 'SON', 'DJF']


def fit_g_ohm(df, rows):
    same = df['Date'].eq(df['Date'].shift())
    dFn = ((df['Fn'] - df['Fn'].shift()) / C.DT_SEC).where(same, 0.0).values
    Fn, Gs, seas = df['Fn'].values, df['Gobs'].values, df['Season_short'].values
    sel = np.zeros(len(df), bool); sel[rows] = True
    out = {}
    for s in SEAS:
        m = sel & (seas == s) & np.isfinite(Fn) & np.isfinite(dFn) & np.isfinite(Gs)
        X = np.column_stack([Fn[m], dFn[m], np.ones(m.sum())])
        out[s] = [float(v) for v in np.linalg.lstsq(X, Gs[m], rcond=None)[0]]
    return out


def season_metrics(df, rows, col_obs, pred):
    s = df['Season_short'].values
    return {k: metrics(df[col_obs].values[rows][s[rows] == k], pred[rows][s[rows] == k]) for k in SEAS}


def main():
    t0 = time.time()
    df, qc = load_observed(OBS, G_TARGET='surface')
    dt = pd.to_datetime(df['DateTime'])
    tr = np.where(dt < SPLIT)[0]; te = np.where(dt >= SPLIT)[0]
    print('records: all %d  training %d  test %d' % (len(df), len(tr), len(te)), flush=True)
    res = {'split': str(SPLIT), 'N_all': int(len(df)), 'N_train': int(len(tr)), 'N_test': int(len(te)),
           'train_period': [str(dt.iloc[tr].min()), str(dt.iloc[tr].max())],
           'test_period': [str(dt.iloc[te].min()), str(dt.iloc[te].max())],
           'N_test_by_season': {s: int((df['Season_short'].values[te] == s).sum()) for s in SEAS}}

    # 1. ground-heat coefficients from training rows only
    g_tr = fit_g_ohm(df, tr)
    res['G_ohm_full_record'] = C.G_OHM; res['G_ohm_training'] = g_tr
    print('G_ohm (training):', {k: [round(x, 4) for x in v] for k, v in g_tr.items()}, flush=True)

    # albedo constants of Eq. (3) from training rows only (radiometer, Fsd > 50 W m-2)
    b = df.iloc[tr]; b = b[b['Fsd'] > 50]; alb = (b['Fsu'] / b['Fsd']).clip(0, 1).values; h = b['Hour'].values
    eq3 = lambda x: np.minimum(x[0] + x[1] * (1 - np.cos(np.pi * np.abs(h - 12) / 6)), C.ALBEDO_CAP) - alb
    an, ar = least_squares(eq3, x0=[0.10, 0.01], bounds=([0, 0], [0.3, 0.2])).x
    res['albedo_training'] = {'alpha_noon': float(an), 'd_alpha': float(ar)}
    res['albedo_full_record'] = {'alpha_noon': C.ALBEDO_NOON, 'd_alpha': C.ALBEDO_RANGE}
    print('albedo (training): %.4f %.4f' % (an, ar), flush=True)

    base = C.params_template(); base['G_ohm'] = g_tr; base['albedo'] = float(an); base['albedo_range'] = float(ar)

    # 2. roughness multiplier sweep on training rows only (cached between the two stages)
    cache = os.path.join(OUT, 'holdout_fm_sweep_cache.json')
    sweep_only = len(sys.argv) > 1 and sys.argv[1] == 'sweep'
    if os.path.exists(cache) and not sweep_only:
        sweep = {float(k): v for k, v in json.load(open(cache)).items()}
    else:
        sweep = {}
        for fm in C.FM_GRID:
            p = {**base, 'ra_multiplier': fm}
            r = calibrate_and_predict(df, tr, tr, p)
            sweep[fm] = metrics(df['Fh'].values[tr], r['H'][tr])['RMSE']
            print('  f_m = %.4f  training H RMSE = %.2f  (%.0fs)' % (fm, sweep[fm], time.time() - t0), flush=True)
        json.dump({str(k): v for k, v in sweep.items()}, open(cache, 'w'))
    if sweep_only:
        return
    fm_best = min(sweep, key=sweep.get)
    res['fm_sweep_training_H_RMSE'] = {str(k): v for k, v in sweep.items()}; res['fm_selected'] = fm_best

    # 3. calibrate on training, predict the test period
    out = {}
    for tag, extra in (('holdout', {}),
                       ('holdout_no_stress_slopes', {'swc_rc_coef': 0.0, 'vpd_rc_coef': 0.0,
                                                     'fsd_rc_coef': 0.0, 'rh_rc_coef': 0.0})):
        p = {**base, 'ra_multiplier': fm_best, **extra}
        r = calibrate_and_predict(df, tr, te, p)
        out[tag] = r
        res[tag] = {'rc_base_by_season': r['rc_by_season'], 'diurnal_correction': r['autocal'],
                    'test': {'H': metrics(df['Fh'].values[te], r['H'][te]),
                             'LE': metrics(df['Fe'].values[te], r['LE'][te]),
                             'G': metrics(df['Gobs'].values[te], r['G'][te])},
                    'train': {'H': metrics(df['Fh'].values[tr], r['H'][tr]),
                              'LE': metrics(df['Fe'].values[tr], r['LE'][tr]),
                              'G': metrics(df['Gobs'].values[tr], r['G'][tr])},
                    'test_by_season': {'H': season_metrics(df, te, 'Fh', r['H']),
                                       'LE': season_metrics(df, te, 'Fe', r['LE']),
                                       'G': season_metrics(df, te, 'Gobs', r['G'])}}
        m = res[tag]['test']
        print('%s TEST  H R2=%.3f RMSE=%.1f | LE R2=%.3f RMSE=%.1f | G R2=%.3f RMSE=%.1f  (%.0fs)'
              % (tag, m['H']['R2'], m['H']['RMSE'], m['LE']['R2'], m['LE']['RMSE'],
                 m['G']['R2'], m['G']['RMSE'], time.time() - t0), flush=True)

    r = out['holdout']
    pd.DataFrame({'DateTime': dt.values[te], 'Season': df['Season_short'].values[te],
                  'H_obs': df['Fh'].values[te], 'H_mod': r['H'][te],
                  'LE_obs': df['Fe'].values[te], 'LE_mod': r['LE'][te],
                  'G_obs': df['Gobs'].values[te], 'G_mod': r['G'][te]}
                 ).to_csv(os.path.join(OUT, 'holdout_lsm_predictions.csv'), index=False)
    json.dump(res, open(os.path.join(OUT, 'holdout_lsm.json'), 'w'), indent=2, default=float)
    print('written', OUT, '%.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
