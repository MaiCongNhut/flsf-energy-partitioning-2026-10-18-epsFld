"""Fig. 8: out-of-fold LSM performance (results/lsm_cv/predictions.csv, N = 14,421).

Left: hourly diurnal composites, observed black solid and modeled colored dashed, each with its IQR.
Right: observed-versus-modeled scatter with the 1:1 line (solid) and the OLS fit (dashed).
R2 = 1 - SSE/SST (model skill), as in Table 3.
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import style as S

S.apply()
p = pd.read_csv(os.path.join(S.paths.RESULTS, 'lsm_cv', 'predictions.csv'))
ROWS = [('Sensible heat flux', 'H', 'Fh_obs', 'H_mod', (-30, 350), (-220, 560)),
        ('Latent heat flux', 'LE', 'Fe_obs', 'LE_mod', (-30, 320), (-220, 560)),
        ('Surface ground heat flux', 'G', 'Fg_obs', 'G_mod', (-40, 110), (-150, 170))]
fig, axes = plt.subplots(3, 2, figsize=(S.DOUBLE_COL, 7.6), gridspec_kw={'width_ratios': [1.12, 1.0]},
                         layout='constrained')
lab = iter('abcdef')
q1 = lambda x: x.quantile(0.25)
q3 = lambda x: x.quantile(0.75)
for ri, (name, key, oc, mc, yd, ys) in enumerate(ROWS):
    col = S.FLUX_COLORS[key]
    axd, axs = axes[ri, 0], axes[ri, 1]; S.grid(axd); S.grid(axs)
    g = p.groupby('Hour').agg(om=(oc, 'mean'), oq1=(oc, q1), oq3=(oc, q3), mm=(mc, 'mean'), mq1=(mc, q1), mq3=(mc, q3))
    g = g.loc[6:18]
    axd.fill_between(g.index, g['oq1'], g['oq3'], color='#555555', alpha=0.18, lw=0)
    axd.fill_between(g.index, g['mq1'], g['mq3'], color=col, alpha=0.20, lw=0)
    axd.plot(g.index, g['om'], color='k', lw=1.8, ls='-', label='Observed')
    axd.plot(g.index, g['mm'], color=col, lw=2.0, ls='--', label='Modeled')
    axd.set_xlim(6, 18); axd.set_ylim(*yd); axd.set_xticks(range(6, 19, 2)); axd.axhline(0, color='#999', lw=0.5)
    axd.set_ylabel('$%s$ (W m$^{-2}$)' % key)
    S.panel(axd, next(lab), name)
    axd.legend(loc='upper right', fontsize=7.5, frameon=True, framealpha=0.95)
    if ri == 2:
        axd.set_xlabel('Hour (LST)')
    o, m = p[oc].values, p[mc].values; ok = np.isfinite(o) & np.isfinite(m); o, m = o[ok], m[ok]
    r2 = 1 - np.sum((m - o) ** 2) / np.sum((o - o.mean()) ** 2)
    rmse = np.sqrt(np.mean((m - o) ** 2)); bias = np.mean(m - o); sl, ic = np.polyfit(o, m, 1)
    axs.scatter(o, m, s=2.5, alpha=0.2, color=col, edgecolors='none', rasterized=True, zorder=3)
    xx = np.array(ys); axs.plot(xx, xx, '-', color='k', lw=1.1, zorder=4, label='1:1')
    axs.plot(xx, sl * xx + ic, '--', color='k', lw=1.5, zorder=5, label='OLS fit')
    axs.set_xlim(*ys); axs.set_ylim(*ys); axs.set_ylabel('Modeled $%s$ (W m$^{-2}$)' % key)
    axs.set_xlabel('Observed $%s$ (W m$^{-2}$)' % key)
    S.panel(axs, next(lab), '$%s$: observed vs. modeled' % key)
    bias_s = ('%+.2f' % bias).replace('-', '\u2212')                 # two decimals, true minus sign
    axs.text(0.04, 0.96, '$R^2$ = %.2f\nRMSE = %.1f W m$^{-2}$\nBias = %s W m$^{-2}$\nSlope = %.2f' % (r2, rmse, bias_s, sl),
             transform=axs.transAxes, va='top', ha='left', fontsize=7,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#888', lw=0.6, alpha=0.95), zorder=6)
    axs.legend(loc='lower right', fontsize=7, frameon=True, framealpha=0.95)
S.save(fig, 'Fig08_LSM_Performance.png')
