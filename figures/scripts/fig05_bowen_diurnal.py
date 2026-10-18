"""Fig. 5: seasonal diurnal Bowen ratio on the flux-characterization dataset (Fsd > 50 W m-2), records with
LE > 20 W m-2 and beta clipped to [-2, 8] (the selection of the Bowen-ratio statistics in Section 3c, N = 11,479).
Lines: hourly medians; shading: interquartile range; dashed line: beta = 1."""
import matplotlib.pyplot as plt
import style as S
from observational_statistics import fc_dataset

S.apply()
d = fc_dataset()
d = d[d['Fe'] > 20].copy(); d['beta'] = (d['Fh'] / d['Fe']).clip(-2, 8)
fig, ax = plt.subplots(figsize=(S.ONE_HALF_COL, 3.3), layout='constrained'); S.grid(ax)
for s in S.SEASON_ORDER:
    g = d[d['Season'] == s].groupby('Hour')['beta']; keep = g.size() >= 20
    m, q1, q3 = g.median()[keep], g.quantile(0.25)[keep], g.quantile(0.75)[keep]
    ax.fill_between(m.index, q1, q3, color=S.SEASON_COLORS[s], alpha=0.12, lw=0)
    ax.plot(m.index, m.values, color=S.SEASON_COLORS[s], lw=0.85, label=S.SEASON_LONG[s])
ax.axhline(1, ls='--', color='#444', lw=0.9)
ax.set_xlim(6, 18); ax.set_xticks(range(6, 19, 2)); ax.set_ylim(-0.1, 3.5)
ax.set_xlabel('Hour (LST)'); ax.set_ylabel(r'Bowen ratio $\beta$')
ax.legend(loc='upper right', fontsize=7.5, frameon=True, framealpha=0.92)
S.save(fig, 'Fig05_Diurnal_Bowen.png')
