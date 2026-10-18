"""Fig. 7 and the u* columns of Table 2: friction velocity on the flux-characterization dataset (N = 12,334).

u* is estimated from the measured wind speed with the neutral logarithmic profile and the aerodynamic
parameters of the land surface model, u* = kappa * U / ln((z - d) / z0m) with z = 4.7 m (the height of the wind
measurement), d = 0.5 m and z0m = 0.15 m, because the processed tower record does not carry the eddy-covariance u*.
Output: figures/output/Fig07_FrictionVelocity.png, results/observational/friction_velocity_table2.json
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import style as S
import config as C
from observational_statistics import fc_dataset

S.apply()
d = fc_dataset()
factor = C.KAPPA / np.log((C.Z_REF - C.DISP) / C.Z0M)
d['ustar'] = factor * d['Ws']
tab = {}
fig, ax = plt.subplots(figsize=(S.DOUBLE_COL * 0.8, 3.6), layout='constrained'); S.grid(ax)
for s in S.SEASON_ORDER:
    u = np.sort(d.loc[d['Season'] == s, 'ustar'].dropna().values)
    ax.plot(u, 100 * np.arange(1, len(u) + 1) / len(u), color=S.SEASON_COLORS[s], lw=0.9, label=S.SEASON_LONG[s])
    tab[s] = {'N': int((d['Season'] == s).sum()), 'mean_ustar': float(factor * d.loc[d['Season'] == s, 'Ws'].mean()),
              'pct_0.1_0.2': float(100 * ((u > 0.1) & (u <= 0.2)).mean()),
              'pct_0.2_0.5': float(100 * ((u > 0.2) & (u <= 0.5)).mean()),
              'pct_le_0.1': float(100 * (u <= 0.1).mean())}
for x in (0.1, 0.2, 0.5):
    ax.axvline(x, ls='--', color='#666', lw=1)
ax.set_xlim(0, 0.7); ax.set_ylim(0, 100)
ax.set_xlabel(r'Friction velocity $u_*$ (m s$^{-1}$)'); ax.set_ylabel('Empirical cumulative frequency (%)')
ax.legend(loc='lower right', fontsize=8)
S.save(fig, 'Fig07_FrictionVelocity.png')
json.dump({'ustar_over_U': float(factor), 'by_season': tab}, open(S.paths.out('observational', 'friction_velocity_table2.json'), 'w'), indent=1)
for s, v in tab.items():
    print(s, {k: round(x, 2) for k, x in v.items()})
