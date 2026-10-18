"""Repository paths. Every script imports this module, so the checkout can live anywhere."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RAW = os.path.join(ROOT, 'data', 'raw')
DERIVED = os.path.join(ROOT, 'data', 'derived')
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(ROOT, 'figures', 'output')

OBSERVED = os.path.join(RAW, 'FLSF_ObservedData.csv')               # half-hourly tower record, 1 Jan 2024 - 1 Feb 2026
PANEL_CONCRETE = os.path.join(RAW, 'panel_concrete_temperature.xlsx')  # HOBO MX2303 back-panel, cavity-air and concrete-pile temperatures (10 min)
SOUNDING = os.path.join(RAW, 'radiosonde_FLSF_2025-10-15_1435LST.txt')  # on-site radiosonde, 15 Oct 2025 (times in UTC)
DAYTIME = os.path.join(DERIVED, 'daytime_record.csv')                # 16,188 daytime records (six drivers complete)
MODELING = os.path.join(DERIVED, 'modeling_dataset.csv')             # 14,421 quality-controlled records (LSM, PCR)
CNN_SERIES = os.path.join(DERIVED, 'cnn_daytime_series.csv')         # 16,323 daytime records with a QC flag (CNN)


def out(*parts):
    """Return a path under results/, creating its directory."""
    p = os.path.join(RESULTS, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def fig(name):
    os.makedirs(FIGURES, exist_ok=True)
    return os.path.join(FIGURES, name)
