"""Fig. 3: diurnal surface albedo by season on the flux-characterization dataset (Fsd > 50 W m-2, N = 12,334).

Solid lines: seasonal hourly means of Fsu/Fsd; shading: interquartile range; dotted line: model reference
alpha_noon of Eq. (3) (flsf/config.py).
"""
import json
import matplotlib.pyplot as plt
import style as S
import config as C

S.apply()
from observational_statistics import fc_dataset
d = fc_dataset(); d['alb'] = (d['Fsu'] / d['Fsd']).clip(0, 1); d['Hour'] = d['DateTime'].dt.hour
fig, ax = plt.subplots(figsize=(S.ONE_HALF_COL, 3.4), layout='constrained'); S.grid(ax)
stats = {}
for s in S.SEASON_ORDER:
    g = d[d['Season'] == s].groupby('Hour')['alb']
    keep = g.size() >= 20                                  # hours with at least 20 records
    m, q1, q3 = g.mean()[keep], g.quantile(0.25)[keep], g.quantile(0.75)[keep]
    ax.fill_between(m.index, q1, q3, color=S.SEASON_COLORS[s], alpha=0.12, lw=0)
    ax.plot(m.index, m.values, color=S.SEASON_COLORS[s], lw=0.85, label=S.SEASON_LONG[s])
    stats[s] = {int(h): float(v) for h, v in m.items()}
ax.axhline(C.ALBEDO_NOON, color='#555555', lw=1.0, ls=':', zorder=1)   # model reference alpha_noon of Eq. (3)
ax.set_xlim(6, 18); ax.set_xticks(range(6, 19, 2)); ax.set_ylim(0.06, 0.24)
ax.set_xlabel('Hour (LST)'); ax.set_ylabel(r'Surface albedo $\alpha$')
ax.legend(loc='upper right', fontsize=7.5, frameon=True, framealpha=0.92)
S.save(fig, 'Fig03_Diurnal_Albedo.png')
json.dump(stats, open(S.paths.out('figures', 'fig03_hourly_mean_albedo.json'), 'w'), indent=1)
