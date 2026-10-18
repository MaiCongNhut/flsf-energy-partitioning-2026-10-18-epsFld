"""
Convective boundary-layer depth H_BL of Eq. (A1) from the on-site radiosonde sounding launched inside the solar
farm on 15 October 2025 (start 06:34:40 UTC = 14:34:40 LST; data/raw/radiosonde_FLSF_2025-10-15_1435LST.txt).

Parcel method: H_BL is the lowest height (above 100 m) at which the virtual potential temperature of the
sounding exceeds its value at the launch point by a given surface excess. The virtual potential temperature
profile is close to neutral between about 1.1 and 1.3 km, so the result depends on the excess; it is reported
for excesses of 0 to 0.5 K. Heights are above the launch point.

Output: results/observational/boundary_layer_height.json
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths, config as C


def sounding():
    d = pd.read_csv(paths.SOUNDING, skiprows=3, header=None, skipinitialspace=True,
                    names=['Time', 'Height', 'P', 'T', 'RH', 'WS', 'WD', 'Ascent'])
    d = d.apply(pd.to_numeric, errors='coerce').dropna(subset=['Height', 'P', 'T', 'RH']).reset_index(drop=True)
    start = open(paths.SOUNDING).readline().split(':', 1)[1].strip()
    return d, start


def main():
    d, start = sounding()
    T = d['T'] + 273.15
    es = 6.112 * np.exp(17.67 * d['T'] / (d['T'] + 243.5))            # hPa, over water
    e = d['RH'] / 100 * es
    q = 0.622 * e / (d['P'] - 0.378 * e)                               # specific humidity
    theta_v = (T * (1000.0 / d['P']) ** 0.2857 * (1 + 0.61 * q)).values
    z = (d['Height'] - d['Height'].iloc[0]).values
    out = {'launch_start_utc': start, 'launch_start_lst': '15/10/2025 14:34:40',
           'theta_v_surface_K': float(theta_v[0]), 'H_BL_used_in_Eq_A1_m': C.H_BL, 'parcel_height_m': {}}
    for exc in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        k = np.where((theta_v > theta_v[0] + exc) & (z > 100))[0][0]
        out['parcel_height_m']['%.1f' % exc] = float(z[k])
    hs = list(out['parcel_height_m'].values())
    out['range_m'] = [min(hs), max(hs)]
    json.dump(out, open(paths.out('observational', 'boundary_layer_height.json'), 'w'), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
