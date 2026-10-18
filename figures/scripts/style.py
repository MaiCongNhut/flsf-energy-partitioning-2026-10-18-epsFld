"""Shared figure style (AMS/JAMC): fonts, colors and helpers used by every figure script."""
import os, sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'flsf'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'data'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'analysis'))
import paths  # noqa: E402

SINGLE_COL, ONE_HALF_COL, DOUBLE_COL = 3.30, 4.80, 6.50
FLUX_COLORS = {'Rn': '#E69F00', 'H': '#D55E00', 'LE': '#0072B2', 'G': '#117733',
               'Eelec': '#CC79A7', 'dQc': '#7A6800', 'Precip': '#332288', 'SWC': '#882255'}
SEASON_COLORS = {'MAM': '#1A9641', 'JJA': '#D7191C', 'SON': '#FF7F00', 'DJF': '#2C7BB6'}
SEASON_ORDER = ['MAM', 'JJA', 'SON', 'DJF']
SEASON_LONG = {'MAM': 'Spring (MAM)', 'JJA': 'Summer (JJA)', 'SON': 'Autumn (SON)', 'DJF': 'Winter (DJF)'}
GRID_COLOR = '#E6E6E6'
OBS_STYLE = dict(color='k', ls='-')          # observed: black solid line
MOD_LS = '--'                                # modeled: dashed line


def apply():
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9.5, 'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.7, 'axes.edgecolor': '#444444',
        # light, short tick marks on the bottom and left axes only; no minor ticks
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': False, 'ytick.right': False,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.minor.visible': False, 'ytick.minor.visible': False,
        'xtick.color': '#8C8C8C', 'ytick.color': '#8C8C8C', 'xtick.labelcolor': '#222222', 'ytick.labelcolor': '#222222',
        'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def grid(ax, axis='both'):
    ax.grid(True, axis=axis, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, letter, extra=''):
    ax.set_title('(%s)%s' % (letter, (' ' + extra) if extra else ''), loc='left', fontsize=10,
                 fontweight='bold', pad=3)


def save(fig, name, dpi=600):
    p = paths.fig(name)
    fig.savefig(p, dpi=dpi, facecolor='white')
    plt.close(fig)
    print('->', os.path.relpath(p, paths.ROOT))
    return p
