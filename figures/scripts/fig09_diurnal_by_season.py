"""Fig. 9: seasonal diurnal composites of rainfall, H, LE, G and modeled dQc (out-of-fold LSM predictions).
Observed: black solid line with gray IQR; modeled: season-colored dashed line with colored IQR."""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S
import config as C

S.apply()
p = pd.read_csv(os.path.join(S.paths.RESULTS, 'lsm_cv', 'predictions.csv'), parse_dates=['DateTime'])
m = pd.read_csv(S.paths.MODELING, parse_dates=['DateTime'])[['DateTime', 'Precip', 'Fn']]
d = p.merge(m, on='DateTime', how='left').sort_values('DateTime').reset_index(drop=True)
same = d['DateTime'].dt.date.eq(d['DateTime'].dt.date.shift())
dRn = ((d['Fn'] - d['Fn'].shift()) / C.DT_SEC).where(same, 0.0)
d['dQc_mod'] = C.OHM_A1 * d['Fn'] + C.OHM_A2 * dRn + C.OHM_A3
ROWS = [('Rainfall', 'Precipitation\n(mm h$^{-1}$)', 'Precip', None, (0, 0.85)),
        ('H', '$H$ (W m$^{-2}$)', 'Fh_obs', 'H_mod', (-20, 400)),
        ('LE', '$LE$ (W m$^{-2}$)', 'Fe_obs', 'LE_mod', (-20, 250)),
        ('G', '$G$ (W m$^{-2}$)', 'Fg_obs', 'G_mod', (-60, 150)),
        ('dQc', r'$\Delta Q_\mathrm{c}$ (W m$^{-2}$)', None, 'dQc_mod', (-20, 50))]
q1 = lambda x: x.quantile(0.25)
q3 = lambda x: x.quantile(0.75)
fig = plt.figure(figsize=(S.DOUBLE_COL, 8.2))
gs = fig.add_gridspec(5, 4, height_ratios=[0.7, 1, 1, 1, 1], hspace=0.18, wspace=0.10, bottom=0.08, top=0.96,
                      left=0.10, right=0.985)
amp = {}
for ci, s in enumerate(S.SEASON_ORDER):
    sub = d[d['Season'] == s]; sc = S.SEASON_COLORS[s]
    for ri, (key, yl, oc, mc, lim) in enumerate(ROWS):
        ax = fig.add_subplot(gs[ri, ci]); S.grid(ax)
        if key == 'Rainfall':
            hp = (sub['Precip'] * 2.0).groupby(sub['Hour']).mean().loc[6:18]   # mm per half hour -> mm h-1
            ax.bar(hp.index, hp.values, width=0.85, color=sc, alpha=0.55, edgecolor=sc, lw=0.6)
        else:
            if mc is not None:
                g = sub.groupby('Hour')[mc].agg(['mean', q1, q3]).loc[6:18]; g.columns = ['m', 'a', 'b']
                ax.fill_between(g.index, g['a'], g['b'], color=sc, alpha=0.22, lw=0)
            if oc is not None:
                o = sub.groupby('Hour')[oc].agg(['mean', q1, q3]).loc[6:18]; o.columns = ['m', 'a', 'b']
                ax.fill_between(o.index, o['a'], o['b'], color='#bdbdbd', alpha=0.55, lw=0)
                ax.plot(o.index, o['m'], color='k', lw=1.4, ls='-', zorder=6)
            ax.plot(g.index, g['m'], color=sc, lw=1.6, ls='--', zorder=5)
            if key == 'dQc':
                amp[s] = float(g['m'].max() - g['m'].min())
            ax.axhline(0, color='#999', lw=0.4)
        ax.set_xlim(5.5, 18.5); ax.set_ylim(*lim); ax.set_xticks([6, 9, 12, 15, 18])
        ax.text(0.03, 0.95, '(%s)' % chr(97 + 4 * ri + ci), transform=ax.transAxes, ha='left', va='top', fontsize=7.5,
                fontweight='bold', zorder=8)
        if ri == 0:
            ax.set_title(S.SEASON_LONG[s], fontsize=8.5, fontweight='bold', loc='left')
        if ci == 0:
            ax.set_ylabel(yl, fontsize=7.5)
        else:
            ax.tick_params(labelleft=False)
        if ri == 4:
            ax.set_xlabel('Hour (LST)', fontsize=7.5)
        else:
            ax.tick_params(labelbottom=False)
h = [Line2D([0], [0], color='k', lw=1.5, ls='-'), Line2D([0], [0], color='#555', lw=1.5, ls='--')]
fig.legend(h, ['Observed (IQR shaded)', 'Modeled, season color (IQR shaded)'], loc='lower center', ncol=2,
           bbox_to_anchor=(0.5, 0.005), fontsize=8.5)
S.save(fig, 'Fig09_Diurnal_Composite_Seasonal.png')
import json
json.dump({'dQc_diurnal_amplitude_of_mean': amp}, open(S.paths.out('figures', 'fig09_dQc_amplitude.json'), 'w'), indent=1)
print('dQc amplitude', {k: round(v, 1) for k, v in amp.items()})
