"""
One-at-a-time parameter sensitivity of the land surface model (Fig. A4).

The model is run on the modeling dataset with the full-record calibration (per-season rc_base and
diurnal correction from results/lsm_cv/metrics.json). Each coefficient is perturbed by -10 % and +10 %
about its calibrated value with everything else fixed; the elasticity is
    E = mean(|RMSE(+10 %) - RMSE0|, |RMSE(-10 %) - RMSE0|) / RMSE0 / 0.10
for H, LE and G. Panel (d) of Fig. A4 sweeps the surface emissivity over 0.90-0.99.
Output: results/parameter_sensitivity/parameter_sensitivity.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths, config as C
from preprocessing import load_observed
from physics import run_model
from calibration import metrics

PARAMS = [  # (label, key in the parameter dictionary or special handler)
    ('rc,base (canopy resistance)', 'rc_base_by_season'),
    ('SWC stress slope', 'swc_rc_coef'),
    ('VPD stress slope', 'vpd_rc_coef'),
    ('RH stress slope', 'rh_rc_coef'),
    ('Fsd stress slope', 'fsd_rc_coef'),
    ('z0m (roughness length)', 'z0m'),
    ('fm (array roughness)', 'ra_multiplier'),
    ('αnoon (noon albedo)', 'albedo'),
    ('η (PV efficiency)', 'eta_effective'),
    ('g1 (G/Rn slope)', ('G_ohm', 0)),
    ('g2 (hysteresis)', ('G_ohm', 1)),
    ('g3 (offset)', ('G_ohm', 2)),
    ('d (displacement height)', 'd'),
]


def perturbed(p, key, frac):
    q = dict(p)
    if key == 'rc_base_by_season':
        q[key] = {s: v * (1 + frac) for s, v in p[key].items()}
    elif isinstance(key, tuple):
        q['G_ohm'] = {s: [c * (1 + frac) if j == key[1] else c for j, c in enumerate(v)] for s, v in p['G_ohm'].items()}
    else:
        q[key] = p[key] * (1 + frac)
    return q


def main():
    df, _ = load_observed(paths.OBSERVED, G_TARGET='surface')
    full = json.load(open(os.path.join(paths.RESULTS, 'lsm_cv', 'metrics.json')))['full_record']
    p0 = {**C.params_template(), 'rc_base_by_season': full['rc_base']}
    ac = full['diurnal']
    th = (df['Hour'].values - 12) * np.pi / 6; sh, ch = np.sin(th), np.cos(th)

    def skill(p):
        o = run_model(df, p)
        H = o['H'] + ac['aH'] * sh + ac['bH'] * ch + ac['cH']
        LE = o['LE'] + ac['aL'] * sh + ac['bL'] * ch + ac['cL']
        return {'H': metrics(df['Fh'].values, H), 'LE': metrics(df['Fe'].values, LE),
                'G': metrics(df['Gobs'].values, o['G'])}

    base = skill(p0)
    rows = []
    for label, key in PARAMS:
        r = {'param': label}
        for frac in (-0.10, 0.10):
            s = skill(perturbed(p0, key, frac))
            for f in ('H', 'LE', 'G'):
                r['%s_RMSE_%+d' % (f, round(frac * 100))] = s[f]['RMSE']
        for f in ('H', 'LE', 'G'):
            d = (abs(r[f + '_RMSE_+10'] - base[f]['RMSE']) + abs(r[f + '_RMSE_-10'] - base[f]['RMSE'])) / 2
            r[f + '_elast'] = d / base[f]['RMSE'] / 0.10
        rows.append(r)
        print('%-30s H %.3f  LE %.3f  G %.3f' % (label, r['H_elast'], r['LE_elast'], r['G_elast']), flush=True)
    eps0 = C.EPSILON; emis = []
    for e in (0.90, 0.93, 0.95, 0.97, 0.99):
        C.EPSILON = e
        s = skill(p0)
        emis.append({'epsilon': e, 'H_R2': s['H']['R2'], 'H_RMSE': s['H']['RMSE'], 'H_bias': s['H']['Bias'],
                     'LE_R2': s['LE']['R2'], 'G_R2': s['G']['R2']})
        print('epsilon %.2f  H R2 %.3f  bias %+.2f' % (e, s['H']['R2'], s['H']['Bias']), flush=True)
    C.EPSILON = eps0
    out = {'baseline': base, 'oat': rows, 'emissivity': emis,
           'note': 'full-record calibration; +/-10 % one-at-a-time; elasticity = mean |dRMSE| / RMSE0 / 0.10'}
    p = paths.out('parameter_sensitivity', 'parameter_sensitivity.json')
    json.dump(out, open(p, 'w'), indent=2, default=float)
    print('wrote', p)


if __name__ == '__main__':
    main()
