"""Fig. 10: Bowen-ratio sensitivity in the (albedo, SWC) space by season (results/bowen_map/bowen_grids.npz)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import style as S

S.apply()
g = np.load(os.path.join(S.paths.RESULTS, 'bowen_map', 'bowen_grids.npz'))
ALB, SWC, obs = g['ALB'], g['SWCg'], g['obs']
A, W = np.meshgrid(ALB, SWC)
allb = np.concatenate([g[s][np.isfinite(g[s])] for s in S.SEASON_ORDER])
vmin, vmax = np.nanpercentile(allb, 1), np.nanpercentile(allb, 99)
levels = np.arange(np.floor(vmin * 10) / 10, np.ceil(vmax * 10) / 10 + 0.001, 0.1)
ticks = np.arange(np.ceil(vmin / 0.05) * 0.05, vmax + 1e-9, 0.05)
fig, axes = plt.subplots(2, 2, figsize=(S.DOUBLE_COL, 5.4), layout='constrained')
for i, (ax, s) in enumerate(zip(axes.ravel(), S.SEASON_ORDER)):
    Z = g[s]
    cf = ax.contourf(A, W, Z, levels=np.linspace(vmin, vmax, 100), cmap='RdYlBu_r', extend='both')
    cs = ax.contour(A, W, Z, levels=levels, colors='k', linewidths=0.5, alpha=0.7)
    ax.clabel(cs, fontsize=7, fmt='%.1f')
    ax.plot(obs[i][0], obs[i][1], 's', color='red', ms=7, mec='k', mew=1.1, label=r'Observed FLSF (median $\alpha$, mean SWC)')
    S.panel(ax, chr(97 + i), s)
    ax.set_xlabel(r'Albedo $\alpha$'); ax.set_ylabel('SWC (m$^3$ m$^{-3}$)')
    if i == 0:
        handle = ax.lines[-1]
fig.legend([handle], [handle.get_label()], loc='outside lower center', fontsize=8, frameon=False)
cb = fig.colorbar(cf, ax=axes, shrink=0.85, pad=0.02, ticks=ticks)
cb.set_label(r'Mean Bowen ratio $\beta = H/LE$'); cb.set_ticklabels(['%.2f' % t for t in ticks])
S.save(fig, 'Fig10_Bowen_Sensitivity_Map.png')
