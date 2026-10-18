"""
Build the three derived datasets from the raw tower record.

  daytime_record.csv      daytime records (time stamps 06:00-18:30 LST, Fsd > 0) after the four-month
                          commissioning period, with all six model drivers present (N = 16,188).
                          Used for the descriptive analyses; the flux-characterization dataset
                          is its subset with Fsd > 50 W m-2 (N = 12,334).
  modeling_dataset.csv    the same records after the modeling screening of Section 2c (N = 14,421).
  cnn_daytime_series.csv  all daytime records (N = 16,323) with a QC_bad flag, so that the CNN
                          input windows stay continuous while flagged targets are excluded.

All three carry the storage-corrected surface ground heat flux Fg_surface (Section 2d).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths, config as C
from preprocessing import load_observed, LIMITS, _papale_spikes, _season

DRIVERS = ['Fsd', 'Fld', 'Ta', 'RH', 'Ws', 'SWC']


def raw_with_surface_g():
    r = pd.read_csv(paths.OBSERVED, skiprows=2, header=0).rename(columns={'xlDateTime': 'DateTime'})
    r.replace(-9999, np.nan, inplace=True)
    r['DateTime'] = pd.to_datetime(r['DateTime'])
    r = r.sort_values('DateTime').reset_index(drop=True)
    r['SWC'] = r[['Sws_10cma', 'Sws_10cmb']].mean(axis=1)
    r['Year'] = r.DateTime.dt.year; r['Month'] = r.DateTime.dt.month; r['Day'] = r.DateTime.dt.day
    r['Hour'] = r.DateTime.dt.hour; r['DOY'] = r.DateTime.dt.dayofyear; r['Date'] = r.DateTime.dt.date
    r['Season'] = r['Month'].map(_season)
    same = r['Date'].eq(r['Date'].shift())
    r['Cs'] = C.soil_heat_capacity(r['SWC'])
    r['dTs_dt'] = ((r['Ts'] - r['Ts'].shift()) / C.DT_SEC).where(same, np.nan)
    r['Fg_surface'] = r['Fg'] + r['Cs'] * C.SOIL_PLATE_DEPTH * r['dTs_dt']
    r['albedo'] = r['Fsu'] / r['Fsd']
    return r


def main():
    os.makedirs(paths.DERIVED, exist_ok=True)
    r = raw_with_surface_g()
    day = r[(r.Fsd > 0) & r.Hour.between(6, 18) & ~((r.Year == 2024) & r.Month.isin([1, 2, 3, 4]))].copy()

    # CNN series: all daytime records with the modeling-screening flag (same tests as load_observed)
    lim = pd.Series(False, index=day.index)
    for c, (lo, hi) in LIMITS.items():
        lim |= ((day[c] < lo) | (day[c] > hi)).fillna(False)
    spike = pd.Series(False, index=day.index)
    for c in ('Fh', 'Fe', 'Fg'):
        spike |= _papale_spikes(day, c)
    dew = (day.RH > 95) & (day.Fe < 0)
    midbad = day.Hour.between(10, 15) & ((day.Fe < 0) | (day.Fh < -50))
    day['QC_bad'] = ((lim | spike | midbad) & ~dew)
    cnn = day.drop(columns=['Day', 'DOY', 'albedo'])
    cnn.to_csv(paths.CNN_SERIES, index=False)

    d6 = day.dropna(subset=DRIVERS).drop(columns=['QC_bad'])
    d6.to_csv(paths.DAYTIME, index=False)

    m, rep = load_observed(paths.OBSERVED, G_TARGET='surface')
    m['albedo'] = (m['Fsu'] / m['Fsd']).clip(0, 1)
    m.to_csv(paths.MODELING, index=False)

    print('daytime (Fsd > 0, post-commissioning):', len(day))
    print('daytime_record (six drivers):        ', len(d6))
    print('  of which Fsd > 50 W m-2:            ', int((d6.Fsd > 50).sum()))
    print('modeling_dataset:                     ', len(m), rep)
    print('cnn_daytime_series:                   ', len(cnn), 'QC_bad =', int(cnn.QC_bad.sum()))


if __name__ == '__main__':
    main()
