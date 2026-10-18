"""
Array-roughness multiplier f_m of the aerodynamic resistance: sweep over config.FM_GRID on the full modeling dataset,
calibrating r_c,base and the diurnal correction at each value, and report the RMSE of H (and LE).
Output: results/calibration/roughness_multiplier_sweep.json.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths, config as C
from preprocessing import load_observed
from calibration import calibrate_and_predict, metrics


def main():
    df, _ = load_observed(paths.OBSERVED, G_TARGET='surface')
    allr = np.arange(len(df)); res = {}
    for fm in C.FM_GRID:
        p = {**C.params_template(), 'ra_multiplier': fm}
        r = calibrate_and_predict(df, allr, allr, p)
        res[str(fm)] = {'H_RMSE': metrics(df['Fh'].values, r['H'])['RMSE'], 'LE_RMSE': metrics(df['Fe'].values, r['LE'])['RMSE']}
        print('f_m = %.4f   H RMSE = %.2f   LE RMSE = %.2f' % (fm, res[str(fm)]['H_RMSE'], res[str(fm)]['LE_RMSE']), flush=True)
    best = min(res, key=lambda k: res[k]['H_RMSE'])
    json.dump({'sweep': res, 'best_by_H_RMSE': float(best), 'config_RA_MULT': C.RA_MULT},
              open(paths.out('calibration', 'roughness_multiplier_sweep.json'), 'w'), indent=2)
    print('minimum H RMSE at f_m =', best, '(config', C.RA_MULT, ')')


if __name__ == '__main__':
    main()
