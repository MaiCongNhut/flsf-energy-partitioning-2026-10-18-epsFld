"""Fig. A4: one-at-a-time parameter sensitivity (results/parameter_sensitivity/parameter_sensitivity.json)."""
import os, json
import matplotlib.pyplot as plt
import style as S

S.apply()
LABEL = {'rc,base (canopy resistance)': r'$r_\mathrm{c,base}$ canopy resistance', 'SWC stress slope': 'SWC stress slope',
         'VPD stress slope': 'VPD stress slope', 'RH stress slope': 'RH stress slope',
         'Fsd stress slope': r'$F_\mathrm{sd}$ stress slope', 'z0m (roughness length)': r'$z_\mathrm{0m}$ roughness length',
         'fm (array roughness)': r'$f_\mathrm{m}$ array roughness', 'αnoon (noon albedo)': r'$\alpha_\mathrm{noon}$ noon albedo',
         'η (PV efficiency)': r'$\eta$ PV efficiency', 'g1 (G/Rn slope)': r'$g_1$ OHM $G/R_\mathrm{n}$ slope',
         'g2 (hysteresis)': r'$g_2$ OHM hysteresis', 'g3 (offset)': r'$g_3$ OHM offset',
         'd (displacement height)': r'$d$ displacement height'}
r = json.load(open(os.path.join(S.paths.RESULTS, 'parameter_sensitivity', 'parameter_sensitivity.json')))
fig, axes = plt.subplots(2, 2, figsize=(S.DOUBLE_COL, 5.6), layout='constrained')
for ax, f, letter, title in ((axes[0, 0], 'H', 'a', 'Sensible heat $H$'), (axes[0, 1], 'LE', 'b', 'Latent heat $LE$'),
                             (axes[1, 0], 'G', 'c', 'Ground heat $G$')):
    rows = sorted(r['oat'], key=lambda q: q[f + '_elast'])
    v = [q[f + '_elast'] for q in rows]
    ax.barh(range(len(rows)), v, color=S.FLUX_COLORS[f], edgecolor='#333', linewidth=0.4, zorder=3)
    for i, x in enumerate(v):
        ax.text(x + max(v) * 0.015, i, '%.3f' % (0.0 if abs(x) < 5e-4 else x), va='center', fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([LABEL.get(q['param'], q['param']) for q in rows], fontsize=7)
    ax.set_xlim(0, max(v) * 1.2); ax.tick_params(axis='y', right=False, length=0)
    ax.set_xlabel('Elasticity, |%ΔRMSE| / %Δparameter'); S.grid(ax, 'x'); S.panel(ax, letter, title)
ax = axes[1, 1]; S.grid(ax)
e = r['emissivity']; eps = [x['epsilon'] for x in e]
ax.plot(eps, [x['H_R2'] for x in e], '-o', color=S.FLUX_COLORS['H'], ms=4, label='$H$ $R^2$')
ax.set_ylabel('$H$ $R^2$', color=S.FLUX_COLORS['H']); ax.tick_params(axis='y', colors=S.FLUX_COLORS['H'], right=False)
ax.set_ylim(0.86, 0.92); ax.set_xlabel(r'Surface emissivity $\varepsilon$')
ax2 = ax.twinx(); ax2.plot(eps, [x['H_bias'] for x in e], '--s', color='#333', ms=4, label='$H$ bias')
ax2.set_ylabel('$H$ bias (W m$^{-2}$)'); ax2.set_ylim(-1.5, 3.0)
ax.axvline(0.97, ls=':', color='#888', lw=1)
ax.text(0.968, 0.865, r'adopted $\varepsilon$ = 0.97', ha='right', fontsize=7, color='#555')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=7, frameon=True, framealpha=0.9)
S.panel(ax, 'd', 'Emissivity robustness')
S.save(fig, 'FigA4_ParamSensitivity.png')
