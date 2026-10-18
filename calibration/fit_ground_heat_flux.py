"""
Eq. (5): per-season ordinary least squares of the storage-corrected surface ground heat flux
    G_surface = g1 * Rn + g2 * dRn/dt + g3
on the modeling dataset (Rn measured; dRn/dt from consecutive records of the same day).
Output: results/calibration/ground_heat_flux_fit.json. The values are stored in flsf/config.py (G_OHM).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths, config as C

def fit(df):
    df = df.reset_index(drop=True)
    same = pd.to_datetime(df['DateTime']).dt.date.eq(pd.to_datetime(df['DateTime']).dt.date.shift())
    dFn = ((df['Fn'] - df['Fn'].shift()) / C.DT_SEC).where(same, 0.0).values
    Fn, Gs, seas = df['Fn'].values, df['Fg_surface'].values, df['Season'].values
    out = {}
    for s in C.SEASON_CODES:
        m = (seas == s) & np.isfinite(Fn) & np.isfinite(dFn) & np.isfinite(Gs)
        X = np.column_stack([Fn[m], dFn[m], np.ones(m.sum())])
        coef = np.linalg.lstsq(X, Gs[m], rcond=None)[0]
        pred = X @ coef
        r2 = 1 - ((Gs[m] - pred) ** 2).sum() / ((Gs[m] - Gs[m].mean()) ** 2).sum()
        out[s] = {'g1': float(coef[0]), 'g2': float(coef[1]), 'g3': float(coef[2]), 'R2': float(r2), 'N': int(m.sum())}
    return out

def main():
    df = pd.read_csv(paths.MODELING)
    out = fit(df)
    json.dump(out, open(paths.out('calibration', 'ground_heat_flux_fit.json'), 'w'), indent=2)
    for s, v in out.items():
        print('%s: g1 = %.4f  g2 = %.1f  g3 = %.2f   R2 = %.3f  N = %d   (config %s)' % (s, v['g1'], v['g2'], v['g3'], v['R2'], v['N'], C.G_OHM[s]))

if __name__ == '__main__':
    main()
