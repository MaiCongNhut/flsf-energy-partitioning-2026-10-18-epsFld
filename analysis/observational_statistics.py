"""
Observational statistics quoted in Sections 2d, 3a, 3c and Appendix A of the paper.

All statistics use the storage-corrected surface ground heat flux
    G = G_plate + Cs(SWC) * dz * dT_soil/dt,   Cs = (1 - phi) C_mineral + SWC C_water + (phi - SWC) C_air
(flsf/config.py; data/build_datasets.py). The flux-characterization (FC) dataset is the subset of
daytime_record.csv with Fsd > 50 W m-2 (N = 12,334).

  closure        conventional (H + LE on Rn - G) and augmented (on Rn - G - dQc - Eelec) closure, the
                 contribution of each PV-specific sink, seasonal slopes and R2 (Fig. 6, Fig. A5, Table 2),
                 and the mean magnitude of each term
  concrete_depth augmented closure slope as a function of the effective concrete depth d (Section 2d)
  ground_flux    mean G at 10:00-14:30 LST and median G/Rn (Fsd > 50) for G_plate and G_surface on the
                 modeling dataset (Section 2d); seasonal ranges of the 30-day running means of Fig. 2;
                 daily correlations with SWC (Section 3a); peaks of the half-hourly mean composite (Fig. 4)
  bowen          seasonal mean and standard deviation of per-record H/LE (LE > 20 W m-2, clipped to [-2, 8])
  peak_duration  hours during which the composite H stays at or above half of its peak (basis of dt in Eq. A1)

R2 here is the squared Pearson correlation of the regression.
Output: results/observational/observational_statistics.json
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import brentq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths, config as C
sys.path.insert(0, os.path.join(paths.ROOT, 'data'))
from build_datasets import raw_with_surface_g

SEASONS = ['MAM', 'JJA', 'SON', 'DJF']


def fc_dataset():
    """FC dataset with the two PV-specific sinks.

    dQc follows Eq. (6) with dRn/dt from consecutive same-day records (zero at the first record of a day),
    as in the land surface model; Eelec = eta * Fsd.
    """
    d = pd.read_csv(paths.DAYTIME, parse_dates=['DateTime']).sort_values('DateTime').reset_index(drop=True)
    same = d['DateTime'].dt.date.eq(d['DateTime'].dt.date.shift())
    d['dRn'] = ((d['Fn'] - d['Fn'].shift()) / C.DT_SEC).where(same, 0.0)
    d['dQc'] = C.OHM_A1 * d['Fn'] + C.OHM_A2 * d['dRn'] + C.OHM_A3
    d['Eelec'] = np.where(d['Fsd'] > C.FSD_ELEC_MIN, C.ETA_EFFECTIVE * d['Fsd'], 0.0)
    d['HLE'] = d['Fh'] + d['Fe']
    d['G'] = d['Fg_surface']
    d = d[(d['Fsd'] > 50) & np.isfinite(d['G']) & np.isfinite(d['HLE'])].copy()
    d['hh'] = d['DateTime'].dt.hour + d['DateTime'].dt.minute / 60
    return d


def fit(x, y):
    r = stats.linregress(x, y)
    return {'slope': float(r.slope), 'intercept': float(r.intercept), 'R2': float(r.rvalue ** 2), 'N': int(len(x))}


def closure(d):
    rng = d['Fn'] - d['G']
    out = {'conventional': fit(rng, d['HLE']),
           'augmented': fit(rng - d['dQc'] - d['Eelec'], d['HLE']),
           'Eelec_only': fit(rng - d['Eelec'], d['HLE']),
           'dQc_only': fit(rng - d['dQc'], d['HLE']),
           'conventional_with_G_plate': fit(d['Fn'] - d['Fg'], d['HLE'])}
    c, a = out['conventional']['slope'], out['augmented']['slope']
    e, q = out['Eelec_only']['slope'] - c, out['dQc_only']['slope'] - c
    out['gain'] = {'total': a - c, 'Eelec': e, 'dQc': q, 'interaction': a - c - e - q}
    out['by_season'] = {}
    for s in SEASONS:
        x = d[d['Season'] == s]
        out['by_season'][s] = {'augmented': fit(x['Fn'] - x['G'] - x['dQc'] - x['Eelec'], x['HLE']),
                               'conventional': fit(x['Fn'] - x['G'], x['HLE'])}
    out['means'] = {'Eelec': float(d['Eelec'].mean()), 'abs_dQc': float(d['dQc'].abs().mean()),
                    'dQc': float(d['dQc'].mean()), 'G': float(d['G'].mean()), 'G_plate': float(d['Fg'].mean()),
                    'Rn': float(d['Fn'].mean()), 'Rn_minus_G': float(rng.mean()),
                    'sinks_share_of_Rn_minus_G': float((d['Eelec'].mean() + d['dQc'].abs().mean()) / rng.mean())}
    return out


def concrete_depth(d):
    """c1-c3 were fitted to rho_c c_c d dTc/dt, so dQc scales linearly with d (d = 0.04 m adopted)."""
    def slope(depth):
        return stats.linregress(d['Fn'] - d['G'] - d['Eelec'] - depth / C.D_EFF_CONCRETE * d['dQc'], d['HLE']).slope
    k = 2 * np.pi / 86400.0
    delta = np.sqrt(2 * C.K_CONCRETE / (C.RHO_CONCRETE * C.CC_CONCRETE * k))
    return {'slope_by_depth': {'%.2f' % x: float(slope(x)) for x in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)},
            'depth_for_slope_1': float(brentq(lambda x: slope(x) - 1.0, 0.0, 0.2)),
            'diurnal_penetration_depth': float(delta)}


def ground_flux(d):
    m = pd.read_csv(paths.MODELING, parse_dates=['DateTime'])
    w = m['Hour'].between(10, 14); b = m['Fsd'] > 50
    out = {'modeling_dataset': {'N': int(len(m)),
                                'G_plate_mean_10_1430': float(m.loc[w, 'Fg'].mean()),
                                'G_surface_mean_10_1430': float(m.loc[w, 'Gobs'].mean()),
                                'median_Gplate_over_Rn_bright': float((m['Fg'] / m['Fn'])[b].median()),
                                'median_Gsurface_over_Rn_bright': float((m['Gobs'] / m['Fn'])[b].median())}}
    # Fig. 2: daily means of the full 25-month record (Fsd > 50 W m-2, time stamps 06:00-18:30 LST), 30-day running mean
    r = raw_with_surface_g().set_index('DateTime')
    day = r[(r['Fsd'] > 50) & r['Hour'].between(6, 18)]
    daily = pd.DataFrame({'Precip': r['Precip'].resample('1D').sum(min_count=1),
                          'SWC': day['SWC'].resample('1D').mean(), 'H': day['Fh'].resample('1D').mean(),
                          'LE': day['Fe'].resample('1D').mean(), 'G': day['Fg_surface'].resample('1D').mean(),
                          'G_plate': day['Fg'].resample('1D').mean()})
    roll = daily.rolling(30, center=True, min_periods=10).mean()
    seas = pd.Series(roll.index.month, index=roll.index).map(
        lambda mo: 'DJF' if mo in (12, 1, 2) else 'MAM' if mo in (3, 4, 5) else 'JJA' if mo in (6, 7, 8) else 'SON')
    keys = ('SWC', 'H', 'LE', 'G', 'G_plate')
    out['fig2_running_mean_range'] = {s: {k: [float(roll.loc[seas == s, k].min()), float(roll.loc[seas == s, k].max())]
                                          for k in keys} for s in SEASONS}
    # Section 3a correlations: daily means of all 06:00-18:30 LST records after commissioning (from 1 May 2024)
    post = r[r['Hour'].between(6, 18)].loc['2024-05-01':]
    dc = pd.DataFrame({'SWC': post['SWC'].resample('1D').mean(), 'H': post['Fh'].resample('1D').mean(),
                       'LE': post['Fe'].resample('1D').mean(), 'G': post['Fg_surface'].resample('1D').mean(),
                       'Precip_lag1': r['Precip'].resample('1D').sum(min_count=1).shift(1).loc['2024-05-01':]})
    out['daily_corr_with_SWC'] = {}
    for k in ('H', 'LE', 'G', 'Precip_lag1'):
        x = dc[[k, 'SWC']].dropna()
        rr = stats.pearsonr(x[k], x['SWC'])
        out['daily_corr_with_SWC'][k] = {'r': float(rr[0]), 'p': float(rr[1]), 'N_days': int(len(x))}
    # Fig. 4: half-hourly mean composite on the FC dataset
    comp = {}
    for s in SEASONS:
        x = d[d['Season'] == s].groupby('hh')
        comp[s] = {k: {'peak': float(x[k].mean().max()), 'hour': float(x[k].mean().idxmax())}
                   for k in ('Fn', 'Fh', 'Fe', 'G')}
    out['fig4_mean_composite_peak'] = comp
    return out


def bowen(d):
    x = d[d['Fe'] > 20].copy()
    x['beta'] = (x['Fh'] / x['Fe']).clip(-2, 8)
    out = {s: {'mean': float(x.loc[x['Season'] == s, 'beta'].mean()), 'sd': float(x.loc[x['Season'] == s, 'beta'].std()),
               'N': int((x['Season'] == s).sum())} for s in SEASONS}
    out['annual'] = {'mean': float(x['beta'].mean()), 'sd': float(x['beta'].std()), 'N': int(len(x))}
    return out


def peak_duration():
    d = pd.read_csv(paths.DAYTIME, parse_dates=['DateTime'])
    d['hh'] = d['DateTime'].dt.hour + d['DateTime'].dt.minute / 60
    out = {}
    for s in SEASONS:
        c = d[d['Season'] == s].groupby('hh')['Fh'].mean()
        pk = c.max()
        out[s] = {'peak_H': float(pk), 'hours_at_or_above_half_peak': float((c >= 0.5 * pk).sum() * 0.5),
                  'equivalent_duration_h': float(c.clip(lower=0).sum() * 0.5 / pk)}
    return out


def main():
    d = fc_dataset()
    res = {'N_fc': int(len(d)), 'N_fc_by_season': {s: int((d['Season'] == s).sum()) for s in SEASONS},
           'closure': closure(d), 'concrete_depth': concrete_depth(d), 'ground_flux': ground_flux(d),
           'bowen': bowen(d), 'peak_duration': peak_duration()}
    p = paths.out('observational', 'observational_statistics.json')
    json.dump(res, open(p, 'w'), indent=2)
    c = res['closure']
    print('N =', res['N_fc'], res['N_fc_by_season'])
    for k in ('conventional', 'augmented', 'Eelec_only', 'dQc_only', 'conventional_with_G_plate'):
        print('%-26s slope %.3f  R2 %.3f  intercept %.1f' % (k, c[k]['slope'], c[k]['R2'], c[k]['intercept']))
    print('gains', {k: round(v, 3) for k, v in c['gain'].items()})
    for s in SEASONS:
        a = c['by_season'][s]
        print(s, 'aug %.3f (R2 %.2f)  conv %.3f (R2 %.2f)' % (a['augmented']['slope'], a['augmented']['R2'],
                                                            a['conventional']['slope'], a['conventional']['R2']))
    print('means', {k: round(v, 3) for k, v in c['means'].items()})
    print('depth', res['concrete_depth'])
    print(json.dumps(res['ground_flux'], indent=1))
    print('bowen', json.dumps(res['bowen']))
    print('peak', json.dumps(res['peak_duration']))
    print('wrote', p)


if __name__ == '__main__':
    main()
