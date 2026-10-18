"""
Coverage of the records and of the auxiliary temperature data, as quoted in Sections 2a and 2c, Table 4 (note f)
and the Fig. A2 caption.

  daytime_window   time stamps of the daytime record (N = 16,188), the modeling dataset (N = 14,421) and the
                   flux-characterization dataset (Fsd > 50 W m-2, N = 12,334): first and last stamp of the day
                   and the number of records at 18:00 and 18:30 LST
  soil_water       range of each 10-cm SWC probe and of their mean (SWC); the porosity phi = 0.44 is the highest
                   value recorded by either probe
  figA2_periods    period of each series in data/raw/AllTemperature.xlsx (Fig. A2), checked by rebuilding the
                   hourly composite from the raw records: the 10-min HOBO MX2303 series
                   (back panel, below-panel air, concrete) from data/raw/panel_concrete_temperature.xlsx and
                   the above-panel air from the tower air temperature (HMP155A, 2.4 m); the front-panel series
                   (operator campaign) has no raw file in the repository
  site_climate     Section 2a and the Fig. 2 caption: mean air temperature of 2024 and 2025 and its monthly range
                   (calendar-month means averaged over the two years), annual precipitation of 2024 and 2025,
                   seasonal mean monthly precipitation over the complete months of the record and the May-September
                   share of the total
  amplitudes       Section 2d: diurnal amplitude (range of the hourly composite) of each Fig. A2 series
  panel_contrast   Table 4, note f: dT = Ta,below - Ta,above on the daytime record, where Ta,below is the
                   half-hourly mean of the 10-min below-panel air record (HOBO MX2303) and
                   Ta,above the tower air temperature; N, period, hourly composite and its morning
                   (08:00-11:00 LST) peak

Output: results/observational/record_coverage.json
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths


def stamps(df):
    hm = df['DateTime'].dt.strftime('%H:%M')
    return {'N': int(len(df)), 'first_stamp': hm.min(), 'last_stamp': hm.max(),
            'n_1800': int((hm == '18:00').sum()), 'n_1830': int((hm == '18:30').sum())}


def tower():
    r = pd.read_csv(paths.OBSERVED, skiprows=2, header=0).rename(columns={'xlDateTime': 'DateTime'})
    r.replace(-9999, np.nan, inplace=True)
    r['DateTime'] = pd.to_datetime(r['DateTime'])
    return r.sort_values('DateTime').set_index('DateTime')


def main():
    out = {}
    day = pd.read_csv(paths.DAYTIME, parse_dates=['DateTime'])
    mod = pd.read_csv(paths.MODELING, parse_dates=['DateTime'])
    out['daytime_window'] = {'daytime_record': stamps(day), 'modeling_dataset': stamps(mod),
                             'flux_characterization': stamps(day[day['Fsd'] > 50])}

    r = tower()
    swc = {}
    for c in ('Sws_10cma', 'Sws_10cmb'):
        s = r[c].dropna()
        swc[c] = {'min': float(s.min()), 'max': float(s.max()), 'time_of_max': str(s.idxmax())}
    m = r[['Sws_10cma', 'Sws_10cmb']].mean(axis=1).dropna()
    swc['SWC_mean_of_probes'] = {'mean': float(m.mean()), 'min': float(m.min()), 'max': float(m.max())}
    out['soil_water'] = swc

    comp_file = pd.read_excel(os.path.join(paths.RAW, 'AllTemperature.xlsx'), sheet_name='Data')
    h = pd.read_excel(paths.PANEL_CONCRETE, sheet_name='Data')
    h['t'] = pd.to_datetime(h['Date-Time (Taiwan Standard Time)'])
    h = h.set_index('t').sort_index()
    rebuild = {'Back_Panel_Surface': h['Back_Panel_Surface'],
               'Air Temperature (below panel)': h['Air Temperature (below panel)'],
               'Concrete_Temperature': h['Concrete_Average'],
               'Air Temperature (above panel)': r['Ta'].loc[:'2025-04-30 23:59']}
    per = {}
    for col, s in rebuild.items():
        s = s.dropna()
        c = s.groupby(s.index.hour).mean().values
        per[col] = {'first': str(s.index.min()), 'last': str(s.index.max()),
                    'max_abs_diff_to_AllTemperature_degC': float(np.abs(c - comp_file[col].values).max())}
    out['figA2_periods'] = per
    out['amplitudes'] = {c: float(comp_file[c].max() - comp_file[c].min()) for c in comp_file.columns if c != 'Hour'}

    yrs = r.loc['2024':'2025']
    mon_t = yrs['Ta'].resample('MS').mean()
    clim_t = mon_t.groupby(mon_t.index.month).mean()
    mon_p = r['Precip'].resample('MS').sum()
    mon_p = mon_p[mon_p.index < '2026-02-01']                     # February 2026 has one day only
    season = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA',
              9: 'SON', 10: 'SON', 11: 'SON'}
    out['site_climate'] = {
        'Ta_mean_2024_2025_degC': float(yrs['Ta'].mean()),
        'Ta_annual_mean_degC': {y: float(r.loc[y, 'Ta'].mean()) for y in ('2024', '2025')},
        'Ta_monthly_mean_degC': {int(k): float(v) for k, v in clim_t.items()},
        'precip_annual_mm': {y: float(r.loc[y, 'Precip'].sum()) for y in ('2024', '2025')},
        'precip_seasonal_mean_monthly_mm': {k: float(v) for k, v in
                                            mon_p.groupby(mon_p.index.month.map(season)).mean().items()},
        'precip_share_May_Sep': float(mon_p[mon_p.index.month.isin([5, 6, 7, 8, 9])].sum() / mon_p.sum())}

    below = h['Air Temperature (below panel)'].resample('30min').mean()
    d = day.set_index('DateTime')
    x = pd.DataFrame({'dT': below.reindex(d.index) - d['Ta']}).dropna()
    hourly = x.groupby(x.index.hour)['dT'].mean()
    morning = hourly.loc[8:10]
    out['panel_contrast'] = {'definition': 'Ta,below - Ta,above (degC)', 'N': int(len(x)),
                             'first': str(x.index.min()), 'last': str(x.index.max()),
                             'hourly_composite': {int(k): float(v) for k, v in hourly.items()},
                             'morning_peak_08_11': float(morning.max()), 'hour_of_peak': int(morning.idxmax())}

    p = paths.out('observational', 'record_coverage.json')
    json.dump(out, open(p, 'w'), indent=2)
    print('->', p)
    print(json.dumps({k: out[k] for k in ('daytime_window', 'soil_water', 'site_climate', 'amplitudes')}, indent=1))
    print({k: (v['first'], v['last'], round(v['max_abs_diff_to_AllTemperature_degC'], 3)) for k, v in per.items()})
    pc = out['panel_contrast']
    print('note f: N = %d, %s to %s, morning peak %.2f degC at %02d LST' % (pc['N'], pc['first'], pc['last'],
                                                                            pc['morning_peak_08_11'], pc['hour_of_peak']))


if __name__ == '__main__':
    main()
