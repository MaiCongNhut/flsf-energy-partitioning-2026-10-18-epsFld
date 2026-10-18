"""Fig. A2: hourly mean temperatures of the front- and back-panel surfaces, the air above and below the panel
and the concrete pile (data/raw/AllTemperature.xlsx, sheet 'Data')."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import style as S

S.apply()
df = pd.read_excel(os.path.join(S.paths.RAW, 'AllTemperature.xlsx'), sheet_name='Data'); h = df['Hour'].values
series = [('Front_Panel_Surface', 'Front-panel surface (1–15 Aug 2023)', '#C0392B', '-'),
          ('Back_Panel_Surface', 'Back-panel surface', '#C0392B', '--'),
          ('Air Temperature (above panel)', 'Air temperature (above panel)', '#222222', '-'),
          ('Air Temperature (below panel)', 'Air temperature (below panel)', '#222222', '--'),
          ('Concrete_Temperature', 'Concrete pile temperature', '#0072B2', '-')]
fig, ax = plt.subplots(figsize=(S.ONE_HALF_COL, 3.4), layout='constrained'); S.grid(ax)
for col, lab, c, ls in series:
    ax.plot(h, df[col].values, color=c, lw=0.9, ls=ls, label=lab)
ax.set_xlim(0, 23); ax.set_xticks(range(0, 24, 3)); ax.set_xlabel('Hour (LST)'); ax.set_ylabel('Temperature (°C)')
fig.legend(loc='outside lower center', ncol=2, fontsize=7.8)
S.save(fig, 'FigA2_PanelConcrete.png')
