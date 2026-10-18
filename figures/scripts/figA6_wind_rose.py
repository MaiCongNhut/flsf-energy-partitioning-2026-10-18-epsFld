"""Fig. A6: daytime wind rose by season on the flux-characterization dataset (Fsd > 50 W m-2, N = 12,334).
Sixteen 22.5-degree sectors centered on north; bar length is the sample frequency (% of the season's records),
stacked by wind-speed bin."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import style as S
from observational_statistics import fc_dataset

S.apply()
d = fc_dataset(); d = d[(d['Ws'] >= 0) & (d['Wd'] >= 0)]
sectors = 16; width = 2 * np.pi / sectors
theta = np.deg2rad(np.arange(sectors) * 360 / sectors)               # sector centers: N, NNE, ...
sector = (((d['Wd'] % 360) + 360 / sectors / 2) // (360 / sectors)).astype(int) % sectors
spd = [0, 1, 2, 3, 4, 5, 7, np.inf]; spd_lab = ['0–1', '1–2', '2–3', '3–4', '4–5', '5–7', '>7']
cols = [plt.cm.YlGnBu(x) for x in np.linspace(0.12, 1.0, len(spd_lab))]
fig, axes = plt.subplots(2, 2, figsize=(S.DOUBLE_COL, 7.0), subplot_kw={'projection': 'polar'})
fig.subplots_adjust(left=0.06, right=0.94, top=0.92, bottom=0.08, hspace=0.36, wspace=0.30)
for i, (ax, s) in enumerate(zip(axes.ravel(), S.SEASON_ORDER)):
    m = d['Season'] == s; sub = d[m]; sec = sector[m]
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    bottom = np.zeros(sectors)
    for b in range(len(spd_lab)):
        k = (sub['Ws'] >= spd[b]) & (sub['Ws'] < spd[b + 1])
        frac = 100 * np.bincount(sec[k], minlength=sectors) / len(sub)
        ax.bar(theta, frac, width=width * 0.92, bottom=bottom, color=cols[b], edgecolor='white', lw=0.4, zorder=3)
        bottom += frac
    ax.set_ylim(0, 28); ax.set_yticks([5, 10, 15, 20, 25]); ax.set_yticklabels(['5 %', '10 %', '15 %', '20 %', '25 %'])
    ax.set_rlabel_position(292.5); ax.tick_params(axis='y', labelsize=7); ax.tick_params(axis='x', labelsize=8)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45))); ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax.grid(color='#bbb', lw=0.5)
    ax.set_title('(%s) %s   ($N$ = %s)' % (chr(97 + i), S.SEASON_LONG[s], format(len(sub), ',')), fontsize=9.5, fontweight='bold', pad=14)
fig.legend([Patch(color=c) for c in cols], ['%s m s$^{-1}$' % l for l in spd_lab], loc='lower center', ncol=7,
           fontsize=7.5, frameon=False, handlelength=1.2, columnspacing=1.0)
S.save(fig, 'FigA6_WindRose.png')
