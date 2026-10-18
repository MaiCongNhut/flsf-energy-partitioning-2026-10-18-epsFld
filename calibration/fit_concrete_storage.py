"""
Eq. (6): concrete-pile heat storage  dQc = c1 * Rn + c2 * dRn/dt + c3,
fitted by least squares to the continuous calorimetric storage flux of the three instrumented piles,
    dQc_obs = rho_c * c_c * d * dTc/dt   (rho_c = 2300 kg m-3, c_c = 880 J kg-1 K-1, d = 0.04 m).
The 10-min pile temperature record (HOBO MX2303) is averaged to 30 min before differencing, and both tendencies use the
30-min step and are set to zero across date changes. Output: results/calibration/concrete_storage_fit.json.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths, config as C


def concrete_storage():
    c = (pd.read_excel(paths.PANEL_CONCRETE, sheet_name='Data')
         .rename(columns={'Date-Time (Taiwan Standard Time)': 'DateTime', 'Concrete_Average': 'Tc'})[['DateTime', 'Tc']]
         .dropna())
    c['DateTime'] = pd.to_datetime(c['DateTime'])
    c = c.set_index('DateTime').resample('30min').mean().dropna().reset_index()
    same = c['DateTime'].dt.date.eq(c['DateTime'].dt.date.shift())
    c['dQc_obs'] = (C.RHO_CONCRETE * C.CC_CONCRETE * C.D_EFF_CONCRETE * c['Tc'].diff() / C.DT_SEC).where(same)
    return c[['DateTime', 'dQc_obs']].dropna()


def radiation():
    r = pd.read_csv(paths.OBSERVED, skiprows=2, header=0).rename(columns={'xlDateTime': 'DateTime'}).replace(-9999, np.nan)
    r['DateTime'] = pd.to_datetime(r['DateTime'])
    r = r.sort_values('DateTime').reset_index(drop=True)
    same = r['DateTime'].dt.date.eq(r['DateTime'].dt.date.shift())
    r['dRn_dt'] = ((r['Fn'] - r['Fn'].shift()) / C.DT_SEC).where(same, 0.0)
    return r[['DateTime', 'Fn', 'dRn_dt']].dropna()


def main():
    m = radiation().merge(concrete_storage(), on='DateTime', how='inner').dropna()
    X = np.column_stack([m['Fn'], m['dRn_dt'], np.ones(len(m))])
    c1, c2, c3 = np.linalg.lstsq(X, m['dQc_obs'].values, rcond=None)[0]
    pred = X @ np.array([c1, c2, c3])
    y = m['dQc_obs'].values
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    res = {'N': int(len(m)), 'start': str(m.DateTime.min()), 'end': str(m.DateTime.max()),
           'c1': float(c1), 'c2_s': float(c2), 'c3': float(c3), 'R2': float(r2),
           'config': {'OHM_A1': C.OHM_A1, 'OHM_A2': C.OHM_A2, 'OHM_A3': C.OHM_A3}}
    json.dump(res, open(paths.out('calibration', 'concrete_storage_fit.json'), 'w'), indent=2)
    print('N = %d (%s to %s)' % (res['N'], res['start'], res['end']))
    print('c1 = %.4f   c2 = %.1f s   c3 = %.2f W m-2   R2 = %.3f   (config %.4f / %.1f / %.2f)'
          % (c1, c2, c3, r2, C.OHM_A1, C.OHM_A2, C.OHM_A3))


if __name__ == '__main__':
    main()
