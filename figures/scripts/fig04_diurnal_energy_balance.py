"""Fig. 4: seasonal diurnal composites of Rn, H, LE and the surface ground heat flux G on the
flux-characterization dataset (Fsd > 50 W m-2, N = 12,334). Lines: half-hourly means; shading: IQR."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import style as S
from observational_statistics import fc_dataset

S.apply()
d = fc_dataset()
fl = [('Fn', 'Rn', r'$R_\mathrm{n}$'), ('Fh', 'H', '$H$'), ('Fe', 'LE', '$LE$'), ('G', 'G', '$G$')]
fig, axes = plt.subplots(1, 4, figsize=(S.DOUBLE_COL, 2.7), sharey=True, layout='constrained')
for j, s in enumerate(S.SEASON_ORDER):
    ax = axes[j]; S.grid(ax); sub = d[d['Season'] == s]
    for col, key, lab in fl:
        g = sub.groupby('hh')[col]; keep = g.size() >= 20      # half-hours with at least 20 records
        m, q1, q3 = g.mean()[keep], g.quantile(0.25)[keep], g.quantile(0.75)[keep]
        c = S.FLUX_COLORS[key]
        ax.fill_between(m.index, q1, q3, color=c, alpha=0.14, lw=0, zorder=2)
        ax.plot(m.index, m.values, color=c, lw=0.8, zorder=4)
    ax.axhline(0, color='#888', lw=0.6)
    ax.set_xlim(5.5, 18.5); ax.set_xticks([6, 9, 12, 15, 18]); ax.set_ylim(-100, 850)
    S.panel(ax, chr(97 + j), s); ax.set_xlabel('Hour (LST)')
axes[0].set_ylabel('Energy flux (W m$^{-2}$)')
h = [Line2D([], [], color=S.FLUX_COLORS[k], lw=0.9) for _, k, _ in fl] + [Patch(fc='#999', alpha=0.25)]
fig.legend(h, [l for _, _, l in fl] + ['IQR'], loc='outside lower center', ncol=5)
S.save(fig, 'Fig04_Seasonal_Diurnal_EB.png')
