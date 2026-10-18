"""
Eq. (3): least-squares fit of the diurnal albedo parameterization
    alpha(t) = alpha_noon + d_alpha * [1 - cos(pi |t - 12| / 6)]   (capped at ALBEDO_CAP)
to the half-hourly CNR4 albedo Fsu/Fsd of the modeling dataset with Fsd > 50 W m-2.
Output: results/calibration/albedo_fit.json. The rounded values are stored in flsf/config.py.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths, config as C

def eq3(h, an, ar):
    return np.minimum(an + ar * (1 - np.cos(np.pi * np.abs(np.asarray(h, float) - 12) / 6)), C.ALBEDO_CAP)

def main():
    d = pd.read_csv(paths.MODELING)
    b = d[d.Fsd > 50].copy(); b['alb'] = (b.Fsu / b.Fsd).clip(0, 1)
    f = least_squares(lambda x: eq3(b.Hour, *x) - b.alb, x0=[0.10, 0.02], bounds=([0, 0], [0.3, 0.2]))
    res = {
        'N': int(len(b)),
        'alpha_noon': float(f.x[0]), 'd_alpha': float(f.x[1]),
        'rmse': float(np.sqrt(np.mean(f.fun ** 2))),
        'observed_mean_10_14_LST': float(b[b.Hour.between(10, 14)].alb.mean()),
        'observed_mean_10_14_LST_by_season': {s: float(v) for s, v in b[b.Hour.between(10, 14)].groupby('Season').alb.mean().items()},
        'observed_hourly_mean': {int(h): float(v) for h, v in b.groupby('Hour').alb.mean().items()},
        'config_values': {'ALBEDO_NOON': C.ALBEDO_NOON, 'ALBEDO_RANGE': C.ALBEDO_RANGE},
    }
    json.dump(res, open(paths.out('calibration', 'albedo_fit.json'), 'w'), indent=2)
    print('alpha_noon = %.4f, d_alpha = %.4f, RMSE = %.4f (N = %d)' % (f.x[0], f.x[1], res['rmse'], res['N']))
    print('config: ALBEDO_NOON = %s, ALBEDO_RANGE = %s' % (C.ALBEDO_NOON, C.ALBEDO_RANGE))

if __name__ == '__main__':
    main()
