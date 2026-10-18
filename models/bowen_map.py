"""
Bowen-ratio sensitivity map (Fig. 10) and management scenarios (Section 3f).

The calibrated land surface physics (per-season r_c,base and diurnal correction fitted on the full
modeling dataset) is driven over the bright-daytime records (Fsd > 50 W m-2) of each season while the
albedo (0.05-0.35) and soil water content (0.05-0.50 m3 m-3) are swept. Beta is the mean of the
per-record H/LE ratios (clipped to [-2, 8]) over records with LE > 20 W m-2, computed identically for
observed and modeled fluxes. G, E_elec and dQc are driven by the observed net radiation.

Usage:  python models/bowen_map.py --season MAM   (one season per call; grid cached)
        python models/bowen_map.py --merge        (scenarios and results/bowen_map/bowen_results.json)
        python models/bowen_map.py                (all seasons in one process)
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths, config as C
from preprocessing import load_observed
from calibration import calibrate_and_predict
from physics import month_to_season

RHO, CP, LV, EPS, SIG, KAP, P = C.RHO_AIR, C.CP_AIR, C.LV, C.EPSILON, C.SIGMA, C.KAPPA, C.P_SURF
ALB = np.arange(0.05, 0.35001, 0.005); SWCg = np.arange(0.05, 0.50001, 0.005)   # 61 x 91
NR_ITERS = 18

def _esat(Tk): Tc = Tk - 273.15; return 610.8 * np.exp(17.27 * Tc / (Tc + 237.3))
def _q(e): return 0.622 * e / (P - 0.378 * e)

def _prep(sub, p):
    a = {}
    a['Fsd'] = sub['Fsd'].values.astype(float); a['Fld'] = sub['Fld'].values.astype(float)
    a['Ta'] = sub['Ta'].values.astype(float) + 273.15; a['RH'] = sub['RH'].values.astype(float) / 100.0
    Ws = sub['WindSpeed'].values.astype(float); a['Ws'] = np.where((Ws > 0.3) & np.isfinite(Ws), Ws, p['u_default'])
    a['Hour'] = sub['Hour'].values.astype(float); a['Month'] = sub['Month'].values.astype(int)
    Fn = sub['Fn'].values.astype(float); dates = sub['Date'].values
    dFn = np.zeros(len(Fn)); prev = None; pdte = None
    for i in range(len(Fn)):
        if pdte is not None and dates[i] == pdte and prev is not None and np.isfinite(Fn[i]) and np.isfinite(prev):
            dFn[i] = (Fn[i] - prev) / C.DT_SEC
        prev = Fn[i] if np.isfinite(Fn[i]) else prev; pdte = dates[i]
    a['qa'] = _q(a['RH'] * _esat(a['Ta']))
    Ta_c = a['Ta'] - 273.15; a['Ta_c'] = Ta_c
    esa = 0.6108 * np.exp(17.27 * Ta_c / (Ta_c + 237.3)); a['VPD'] = np.maximum(0.0, esa * (1 - a['RH']))
    z0m, d, zr = p['z0m'], p['d'], p['z_ref']; z0h = 0.1 * z0m; z = max(zr, d + z0m + 1)
    a['ra_neut'] = np.maximum(np.log((z - d) / z0m) * np.log((z - d) / z0h) / (KAP**2 * a['Ws']), 10.0)
    # constant sink terms (from OBSERVED Fn): G(OHM) + E_elec + dQc  — independent of (alpha,SWC,Ts)
    s = month_to_season(int(round(np.median(a['Month']))))
    g1, g2, g3 = p['G_ohm'][s]; G = g1 * sub['Fn'].values + g2 * dFn + g3
    E = np.where(a['Fsd'] >= C.FSD_ELEC_MIN, p['eta_effective'] * a['Fsd'], 0.0)
    dQ = p['ohm_a1'] * sub['Fn'].values + p['ohm_a2'] * dFn + p['ohm_a3']
    a['const'] = G + E + dQ
    ha = (a['Hour'] - 12.0) * np.pi / 6.0; a['sinh'] = np.sin(ha); a['cosh'] = np.cos(ha)
    return a

def _rc(a, sw, p, s):
    m = a['Month']
    sf = np.select([np.isin(m, [6, 7, 8]), m == 9, np.isin(m, [5, 10]), m == 4, np.isin(m, [3, 11])],
                   [0.85, 1.2, 1.1, 1.0, 0.9], default=1.0)
    tf = 1 + 0.01 * np.maximum(0.0, a['Ta_c'] - 25)
    rh = np.clip(1.0 + p['rh_rc_coef'] * (p['rh_ref'] - a['RH']), 0.8, 1.3)
    swf = np.clip(1.0 - p['swc_rc_coef'] * (sw - p['swc_ref']), 0.6, 1.4)
    vp = np.clip(1.0 + p['vpd_rc_coef'] * (a['VPD'] - p['vpd_ref']), 0.7, 2.0)
    fs = np.clip(1.0 + p['fsd_rc_coef'] * (a['Fsd'] - p['fsd_ref']) / 500, 0.8, 2.0)
    return np.clip(p['rc_base_by_season'][s] * sf * tf * rh * swf * vp * fs, C.RC_CLIP[0], C.RC_CLIP[1])

def _ra(a, Ts, p):
    m = a['Month']
    sf = np.select([np.isin(m, [12, 1, 2]), np.isin(m, [3, 11]), np.isin(m, [4, 10])], [1.05, 1.08, 1.02], default=1.0)
    dT = Ts - a['Ta']; stab = np.where(dT > 8, 0.95, np.where(dT < -5, 1.10, 1.0))
    rh = np.clip(1.0 + p['ra_rh_coef'] * (a['RH'] - 0.70), 0.9, 1.3)
    return np.maximum(a['ra_neut'] * sf * stab * rh * p['ra_multiplier'], 15.0)

def _HLE(a, al, rc, p):
    def resid(Ts):
        Rn = (1 - al) * a['Fsd'] + EPS * a['Fld'] - EPS * SIG * Ts**4
        ra = _ra(a, Ts, p); qs = _q(_esat(Ts))
        H = RHO * CP * (Ts - a['Ta']) / ra
        rle = np.where(qs >= a['qa'], ra + rc, ra); LE = RHO * LV * (qs - a['qa']) / rle
        return Rn - H - LE - a['const']
    Ts = a['Ta'] + 3.0
    for _ in range(NR_ITERS):
        r = resid(Ts); rp = resid(Ts + 0.1); dF = (rp - r) / 0.1
        dF = np.where(np.abs(dF) < 1e-9, -1.0, dF)
        Ts = np.clip(Ts - np.clip(r / dF, -5, 5), a['Ta'] - 10, a['Ta'] + 25)
    ra = _ra(a, Ts, p); qs = _q(_esat(Ts))
    H = RHO * CP * (Ts - a['Ta']) / ra
    rle = np.where(qs >= a['qa'], ra + rc, ra); LE = RHO * LV * (qs - a['qa']) / rle
    return H, LE

def beta_grid(a, p, s, ac):
    """LSM-consistent surface Bowen ratio over (alpha, SWC)."""
    out = np.full((len(SWCg), len(ALB)), np.nan)
    for si, sw in enumerate(SWCg):
        rc = _rc(a, sw, p, s)
        for ai, al in enumerate(ALB):
            H, LE = _HLE(a, al, rc, p)
            Hc = H + ac['aH'] * a['sinh'] + ac['bH'] * a['cosh'] + ac['cH']
            Lc = LE + ac['aL'] * a['sinh'] + ac['bL'] * a['cosh'] + ac['cL']
            m = Lc > 20
            out[si, ai] = np.nanmean(np.clip((Hc[m] / Lc[m]), -2.0, 8.0)) if m.sum() > 5 else np.nan
    return out

def _setup():
    df, qc = load_observed(paths.OBSERVED, G_TARGET="surface")
    n = len(df); pt = C.params_template()
    r = calibrate_and_predict(df, np.arange(n), np.arange(n), pt)   # full-record calibration
    ac = r['autocal']; rcs = r['rc_by_season']
    p = {**pt, 'rc_base_by_season': rcs, 'use_diurnal_albedo': False}
    df['alpha'] = (df['Fsu'] / df['Fsd']).clip(0, 1); fc = df[df['Fsd'] > 50]
    return fc, p, ac, rcs


def run_season(s, fc, p, ac):
    sub = fc[fc['Season'] == s]; a = _prep(sub, p)
    g = beta_grid(a, p, s, ac)
    Fh = sub['Fh'].values; Fe = sub['Fe'].values; ok = Fe > 20
    obs_beta = np.nanmean(np.clip((Fh[ok] / Fe[ok]), -2.0, 8.0))
    obs = (float(sub['alpha'].median()), float(sub['SWC'].mean()), float(obs_beta))
    np.savez(paths.out('bowen_map', 'grid_%s.npz' % s), grid=g, obs=np.array(obs))
    print('  %s: beta %.2f-%.2f  obs_beta=%.2f' % (s, np.nanmin(g), np.nanmax(g), obs_beta), flush=True)


def merge(rcs):
    lab = ['MAM', 'JJA', 'SON', 'DJF']; grids = {}; obs_pt = {}
    for s in lab:
        z = np.load(paths.out('bowen_map', 'grid_%s.npz' % s)); grids[s] = z['grid']; obs_pt[s] = tuple(z['obs'])
    np.savez(paths.out('bowen_map', 'bowen_grids.npz'), **{s: grids[s] for s in lab},
             ALB=ALB, SWCg=SWCg, obs=np.array([obs_pt[s] for s in lab]))
    def beta_at(s, al, sw):
        ai = np.argmin(np.abs(ALB - al)); si = np.argmin(np.abs(SWCg - sw)); return grids[s][si, ai]
    base = float(np.nanmean([beta_at(s, *obs_pt[s][:2]) for s in lab]))
    cool = float(np.nanmean([beta_at(s, 0.20, obs_pt[s][1]) for s in lab]))
    irr = float(np.nanmean([beta_at(s, obs_pt[s][0], 0.40) for s in lab]))
    comb = float(np.nanmean([beta_at(s, 0.20, 0.40) for s in lab]))
    scen = {"baseline_beta": base, "cool_roof_a0.20": cool, "irrigation_swc0.40": irr, "combined": comb,
            "cool_reduction_pct": 100 * (base - cool) / base, "irr_reduction_pct": 100 * (base - irr) / base,
            "combined_reduction_pct": 100 * (base - comb) / base}
    seas_scen = {s: {'baseline': float(beta_at(s, *obs_pt[s][:2])), 'albedo_0.20': float(beta_at(s, 0.20, obs_pt[s][1])),
                     'swc_0.40': float(beta_at(s, obs_pt[s][0], 0.40)), 'combined': float(beta_at(s, 0.20, 0.40))} for s in lab}
    json.dump({"observed_point_alpha_swc_beta": {s: obs_pt[s] for s in lab},
               "seasonal_obs_beta": {s: obs_pt[s][2] for s in lab},
               "seasonal_model_at_obs": {s: float(beta_at(s, *obs_pt[s][:2])) for s in lab},
               "seasonal_scenarios": seas_scen,
               "rc_base_by_season": rcs, "scenarios": scen, "grid": "61x91"},
              open(paths.out('bowen_map', 'bowen_results.json'), "w"), indent=2, default=float)
    print("annual-mean beta: baseline=%.3f  albedo 0.20=%.3f (%.1f %%)  SWC 0.40=%.3f (%.1f %%)  combined=%.3f (%.1f %%)" % (
        base, cool, scen['cool_reduction_pct'], irr, scen['irr_reduction_pct'], comb, scen['combined_reduction_pct']))


def main():
    fc, p, ac, rcs = _setup()
    if '--season' in sys.argv:
        run_season(sys.argv[sys.argv.index('--season') + 1], fc, p, ac); return
    if '--merge' not in sys.argv:
        for s in ['MAM', 'JJA', 'SON', 'DJF']:
            run_season(s, fc, p, ac)
    merge(rcs)


if __name__ == "__main__":
    main()
