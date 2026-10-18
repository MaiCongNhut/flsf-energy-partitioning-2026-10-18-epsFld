"""Fig. 2: 25-month time series of precipitation, SWC, H, LE and the surface ground heat flux G.
Daily precipitation bars are clipped at 75 mm d-1 (eight days exceed 100 mm d-1).

Daily means of the bright-daytime records (Fsd > 50 W m-2, time stamps 06:00-18:30 LST) of the full tower record,
with 30-day centered running means; G is the storage-corrected surface flux (Section 2d).
"""
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import style as S
from build_datasets import raw_with_surface_g

S.apply()
r = raw_with_surface_g().set_index('DateTime')
day = r[(r['Fsd'] > 50) & r['Hour'].between(6, 18)]
daily = pd.DataFrame({'Precip': r['Precip'].resample('1D').sum(min_count=1),
                      'SWC': day['SWC'].resample('1D').mean(), 'H': day['Fh'].resample('1D').mean(),
                      'LE': day['Fe'].resample('1D').mean(), 'G': day['Fg_surface'].resample('1D').mean()})
roll = daily.rolling(30, center=True, min_periods=10).mean()
t = daily.index
rows = [('Precip', 'Precipitation', 'Daily total\n(mm d$^{-1}$)', S.FLUX_COLORS['Precip']),
        ('SWC', 'Soil water content (10 cm)', 'SWC (m$^3$ m$^{-3}$)', S.FLUX_COLORS['SWC']),
        ('H', 'Sensible heat flux $H$', '$H$ (W m$^{-2}$)', S.FLUX_COLORS['H']),
        ('LE', 'Latent heat flux $LE$', '$LE$ (W m$^{-2}$)', S.FLUX_COLORS['LE']),
        ('G', 'Surface ground heat flux $G$', '$G$ (W m$^{-2}$)', S.FLUX_COLORS['G'])]
fig, axes = plt.subplots(5, 1, figsize=(S.DOUBLE_COL, 7.6), sharex=True, layout='constrained')
for i, (k, title, lab, c) in enumerate(rows):
    ax = axes[i]; S.grid(ax)
    for y in range(t.min().year, t.max().year + 1):
        ax.axvspan(pd.Timestamp(y, 5, 1), pd.Timestamp(y, 9, 30), color='#4477AA', alpha=0.12, lw=0, zorder=0)
    if k == 'Precip':
        ax.bar(t, daily['Precip'], width=1.0, color=c, alpha=0.5, lw=0, zorder=2)
        ax.set_ylabel(lab, color=c); ax.tick_params(axis='y', colors=c); ax.set_ylim(0, 75)
        ax2 = ax.twinx(); ax2.plot(t, roll['Precip'], color='#111', lw=1.6, zorder=4)
        ax2.set_ylabel('30-day mean\n(mm d$^{-1}$)'); ax2.set_ylim(0, 24)
        h = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5), plt.Line2D([], [], color='#111', lw=1.6),
             plt.Rectangle((0, 0), 1, 1, color='#4477AA', alpha=0.12)]
        ax2.legend(h, ['Daily total (left)', '30-day mean (right)', 'Wet season'],
                   loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=7.2, frameon=False, columnspacing=1.2,
                   handlelength=1.4, borderaxespad=0.1)   # above the panel, so no data are hidden
    else:
        ax.plot(t, daily[k], color=c, lw=0.5, alpha=0.35, zorder=2)
        ax.plot(t, roll[k], color=c, lw=1.8, zorder=4)
        ax.set_ylabel(lab)
        if k != 'SWC':
            ax.axhline(0, color='#999', lw=0.5)
    S.panel(ax, chr(97 + i), title)
axes[-1].xaxis.set_major_locator(mdates.MonthLocator((3, 6, 9, 12)))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
axes[-1].set_xlim(t.min(), t.max()); axes[-1].set_xlabel('Date')
fig.align_ylabels(axes)
S.save(fig, 'Fig02_TimeSeries.png')
