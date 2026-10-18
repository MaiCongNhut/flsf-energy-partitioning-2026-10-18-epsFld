"""Fig. A5: conventional closure, H + LE against Rn - G (surface G), on the flux-characterization dataset."""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import style as S
from observational_statistics import fc_dataset

S.apply()
d = fc_dataset(); d['RnG'] = d['Fn'] - d['G']
c = stats.linregress(d['RnG'], d['HLE'])
fig = plt.figure(figsize=(S.DOUBLE_COL, 3.55), layout='constrained')
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0]); a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[0, 1])
S.grid(a0)
a0.scatter(d['RnG'], d['HLE'], color='#2c2c2c', alpha=0.18, s=3, edgecolors='none', rasterized=True, zorder=2)
x = np.linspace(-100, 950, 50)
a0.plot(x, c.slope * x + c.intercept, ls='--', color='#c0392b', lw=1.6, label='OLS fit', zorder=5)
a0.plot(x, x, ls='-', color='k', lw=1.6, label='1:1 (perfect closure)', zorder=6)
a0.text(0.97, 0.04, '$N$ = %s\nslope = %.3f\n$R^2$ = %.2f' % (
    format(len(d), ','), c.slope, c.rvalue ** 2), transform=a0.transAxes, ha='right', va='bottom',
    fontsize=7.5, bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#888', alpha=0.93))
a0.set_xlim(-100, 950); a0.set_ylim(-100, 950); a0.set_aspect('equal', adjustable='box')
a0.set_xlabel(r'Conventional available energy: $R_\mathrm{n} - G$ (W m$^{-2}$)')
a0.set_ylabel(r'Turbulent flux: $H + LE$ (W m$^{-2}$)')
a0.legend(loc='upper left', fontsize=7.5, frameon=True, framealpha=0.93, edgecolor='#aaa')
S.panel(a0, 'a', 'Conventional closure')
S.grid(a1, 'x')
sl, r2 = [], []
for s in S.SEASON_ORDER:
    x_ = d[d['Season'] == s]; q = stats.linregress(x_['RnG'], x_['HLE']); sl.append(q.slope); r2.append(q.rvalue ** 2)
y = np.arange(4)
a1.barh(y, sl, height=0.62, color='#4A6FA5', edgecolor='k', linewidth=0.6, zorder=3)
a1.axvline(1.0, color='k', ls='-', lw=0.8, zorder=1)
for i, (v, q) in enumerate(zip(sl, r2)):
    a1.text(v - 0.02, i, '%.3f' % v, va='center', ha='right', fontsize=7.5, color='white', fontweight='bold', zorder=5)
    a1.text(1.04, i, '$R^2$ = %.2f' % q, va='center', ha='left', fontsize=7.5, zorder=5,
            bbox=dict(fc='white', ec='none', pad=0.5))
a1.set_yticks(y); a1.set_yticklabels([S.SEASON_LONG[s] for s in S.SEASON_ORDER]); a1.invert_yaxis()
a1.set_xlim(0, 1.42); a1.tick_params(axis='y', right=False); a1.set_xlabel('Conventional closure slope')
S.panel(a1, 'b', 'By season')
S.save(fig, 'FigA5_ConventionalEBC.png')
