"""
Land surface model: season-stratified five-fold time-block cross-validation (Section 2f, Table 3).

Within each season the modeling dataset is split into five consecutive blocks. In each fold the
per-season base canopy resistance (bounded Brent, minimum RMSE of LE) and the diurnal residual
correction (OLS on [sin, cos, 1]) are fitted on the four training blocks and applied to the fifth.

Outputs (results/lsm_cv/):
  metrics.json          pooled and per-season out-of-fold skill (H, LE, G), physical-model-only skill,
                        per-fold coefficients and the full-record reference calibration
  predictions.csv       out-of-fold predictions (used by Fig. 8, Fig. 9 and the hold-out comparison)
"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths, config as C
from preprocessing import load_observed
from calibration import calibrate_and_predict, metrics, per_season_metrics, make_folds
from physics import run_model


def main():
    df, qc = load_observed(paths.OBSERVED, G_TARGET='surface')
    df = make_folds(df); n = len(df); sa = df['Season_short'].values
    pt = C.params_template()
    Ho, LEo, Go = (np.full(n, np.nan) for _ in range(3))
    Hphys, LEphys = np.full(n, np.nan), np.full(n, np.nan)
    folds = {}
    t = time.time()
    for k in range(1, C.K_FOLDS + 1):
        val = np.where(df['fold'].values == k)[0]; cal = np.where(df['fold'].values != k)[0]
        r = calibrate_and_predict(df, cal, val, pt)
        Ho[val] = r['H'][val]; LEo[val] = r['LE'][val]; Go[val] = r['G'][val]
        o = run_model(df, {**pt, 'rc_base_by_season': r['rc_by_season']}, indices=np.arange(n))
        Hphys[val] = o['H'][val]; LEphys[val] = o['LE'][val]
        folds[k] = {'rc_base': r['rc_by_season'], 'diurnal': r['autocal']}
        print('  fold %d/%d (%.0f s)' % (k, C.K_FOLDS, time.time() - t), flush=True)

    # full-record reference calibration (Section 2d reference values)
    allr = np.arange(n)
    full = calibrate_and_predict(df, allr, allr, pt)

    out = {
        'N': n, 'qc': qc,
        'H': metrics(df['Fh'].values, Ho), 'LE': metrics(df['Fe'].values, LEo), 'G': metrics(df['Gobs'].values, Go),
        'H_season': per_season_metrics(df['Fh'].values, Ho, sa),
        'LE_season': per_season_metrics(df['Fe'].values, LEo, sa),
        'G_season': per_season_metrics(df['Gobs'].values, Go, sa),
        'physical_only': {'H': metrics(df['Fh'].values, Hphys), 'LE': metrics(df['Fe'].values, LEphys)},
        'folds': folds,
        'full_record': {'rc_base': full['rc_by_season'], 'diurnal': full['autocal']},
        'config': {'ALBEDO_NOON': C.ALBEDO_NOON, 'ALBEDO_RANGE': C.ALBEDO_RANGE, 'G_OHM': C.G_OHM,
                   'RA_MULT': C.RA_MULT, 'POROSITY': C.POROSITY},
    }
    json.dump(out, open(paths.out('lsm_cv', 'metrics.json'), 'w'), indent=2, default=float)
    pd.DataFrame({'DateTime': pd.to_datetime(df['DateTime']).values, 'Hour': df['Hour'].values, 'Season': sa,
                  'fold': df['fold'].values,
                  'Fh_obs': df['Fh'].values, 'H_mod': Ho, 'Fe_obs': df['Fe'].values, 'LE_mod': LEo,
                  'Fg_obs': df['Gobs'].values, 'G_mod': Go}).to_csv(paths.out('lsm_cv', 'predictions.csv'), index=False)
    for f in ('H', 'LE', 'G'):
        m = out[f]
        print('%-2s R2 = %.3f  RMSE = %.2f  Bias = %+.2f  Slope = %.3f' % (f, m['R2'], m['RMSE'], m['Bias'], m['Slope']))
    print('physical only: H R2 %.3f RMSE %.1f | LE R2 %.3f RMSE %.1f' % (
        out['physical_only']['H']['R2'], out['physical_only']['H']['RMSE'],
        out['physical_only']['LE']['R2'], out['physical_only']['LE']['RMSE']))


if __name__ == '__main__':
    main()
