"""
Sensitivity of the storage-corrected surface ground heat flux to the soil porosity phi (Section 2d).

phi = 0.44 is the highest soil water content recorded by the two 10-cm probes (18 March 2025), taken as
saturation. The surface flux G = G_plate + Cs(SWC) dz dT_soil/dt is recomputed with
Cs = (1 - phi) C_m + SWC C_w + (phi - SWC) C_a for phi = 0.40 to 0.50, and the following are reported:
  - conventional and augmented closure slopes on the FC dataset (as in analysis/observational_statistics.py);
  - mean G between 10:00 and 14:30 LST and median G/Rn (Fsd > 50 W m-2) on the modeling dataset.

Output: results/observational/porosity_sensitivity.json
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths, config as C
from observational_statistics import fc_dataset


def cs(theta, phi):
    th = np.clip(theta, 0.0, phi)
    return (1.0 - phi) * C.C_MINERAL + th * C.C_WATER + (phi - th) * C.C_AIR


def main():
    d0 = fc_dataset()
    m = pd.read_csv(paths.MODELING)
    w = m['Hour'].between(10, 14); b = m['Fsd'] > 50
    out = {'phi_used': C.POROSITY, 'max_SWC_mean_of_probes_FC': float(d0['SWC'].max()), 'by_phi': {}}
    for phi in (0.40, 0.42, 0.44, 0.46, 0.48, 0.50):
        G = d0['Fg'] + cs(d0['SWC'], phi) * C.SOIL_PLATE_DEPTH * d0['dTs_dt']
        rng = d0['Fn'] - G
        Gm = m['Fg'] + cs(m['SWC'], phi) * C.SOIL_PLATE_DEPTH * m['dTs_dt']
        out['by_phi']['%.2f' % phi] = {
            'conventional_slope': float(stats.linregress(rng, d0['HLE']).slope),
            'augmented_slope': float(stats.linregress(rng - d0['dQc'] - d0['Eelec'], d0['HLE']).slope),
            'G_mean_10_1430': float(Gm[w].mean()),
            'median_G_over_Rn_bright': float((Gm / m['Fn'])[b].median())}
    p = paths.out('observational', 'porosity_sensitivity.json')
    json.dump(out, open(p, 'w'), indent=2)
    print('->', p)
    for k, v in out['by_phi'].items():
        print('phi %s  conventional %.3f  augmented %.3f  G(10-14:30) %.1f  median G/Rn %.3f'
              % (k, v['conventional_slope'], v['augmented_slope'], v['G_mean_10_1430'], v['median_G_over_Rn_bright']))


if __name__ == '__main__':
    main()
