"""
Configuration and parameter provenance of the process-based land surface model (LSM) for the
Feng-Lin Sheng-Feng (FLSF) photovoltaic solar farm, Hualien, Taiwan (Mai et al., JAMC-D-26-0098).

This file holds every numerical constant of the model. Each parameter has a provenance class:

  Class A  re-estimated inside every cross-validation fold
  Class B  fixed once against the full record (offline fit or sweep; script named in the comment)
  Class C  physical constant, measured value or literature value, not tuned to the flux record
"""

# --- Physical constants (Class C) --------------------------------------------
KAPPA   = 0.40          # von Karman constant                    (Foken 2008)
SIGMA   = 5.67e-8       # Stefan-Boltzmann       [W m-2 K-4]
EPSILON = 0.97          # broadband surface emissivity (glass+concrete composite)
RHO_AIR = 1.225         # air density            [kg m-3]        (standard atm.)
CP_AIR  = 1005.0        # air heat capacity      [J kg-1 K-1]
LV      = 2.501e6       # latent heat of vaporization [J kg-1]
P_SURF  = 101325.0      # surface pressure       [Pa]

# --- Radiation / albedo (Class B) ---------------------------------------------
# Eq. (3), fitted by least squares to the half-hourly CNR4 albedo (Fsu/Fsd) of the modeling
#   dataset with Fsd > 50 W m-2 (calibration/fit_albedo.py).
ALBEDO_NOON  = 0.098    # minimum (solar-noon) albedo
ALBEDO_RANGE = 0.0065   # diurnal half-amplitude
ALBEDO_CAP   = 0.16

# Net radiation, Eq. (2): Rn = (1 - a)Fsd + eps*Fld - eps*sigma*Ts^4 (gray body; the surface absorbs
#   eps*Fld and reflects (1 - eps)*Fld, Kirchhoff's law).
RN_EMISSIVITY = True    # apply the emissivity to the absorbed longwave flux (Eq. 2)

# --- Aerodynamics (z0m and d prescribed, Class C; f_m Class B) -----------------
Z0M   = 0.15           # momentum roughness [m] (~10% obstacle height; Macdonald 1998)
DISP  = 0.5            # zero-plane displacement [m]              (Grimmond & Oke 1999a)
Z_REF = 4.7            # reference height of the logarithmic wind profile = height of the wind measurement [m]
U_DEFAULT = 2.5        # calm-period wind fallback [m s-1]
H_BL      = 1260.0     # convective boundary-layer depth of Eq. (A1) [m], on-site sounding (analysis/boundary_layer_height.py)
RA_MULT   = 1.5678     # array-roughness multiplier f_m (1.57 in the paper); the full-record sweep over FM_GRID is flat between 1.43 and 1.57 (H RMSE 34.52 vs 34.60 W m-2) and the hold-out training year selects 1.57; calibration/select_roughness_multiplier.py
FM_GRID   = (1.2828, 1.4253, 1.5678, 1.7104, 1.8529, 1.9954)   # values of f_m tried by the sweeps (1.28 to 2.00)
RA_RH_COEF= 0.3        # r_a humidity slope

# --- Canopy resistance (Jarvis-Stewart; rc_base is Class A) -------------------
RC_BASE_INIT = 620.0   # initial base canopy resistance [s m-1]   (Jarvis 1976; Monteith 1965)
RC_LOWER, RC_UPPER = 150.0, 1400.0   # Brent bounds for per-fold rc_base
RC_CLIP = (30.0, 800.0)
SWC_REF, SWC_RC_COEF = 0.30, 0.8
VPD_REF, VPD_RC_COEF = 0.8, 0.5
FSD_REF, FSD_RC_COEF = 400.0, 0.6
RH_REF,  RH_RC_COEF  = 0.70, 0.4
# Per-season base canopy resistance calibrated on the full record (Section 2d; results/lsm_cv/metrics.json).
# Reference only: the model calibrates rc_base itself in every run.
RC_SEASON_REF = {"MAM": 580, "JJA": 596, "SON": 591, "DJF": 649}

# --- Electricity export (Class C; metered) -----------------------------------
ETA_EFFECTIVE = 0.11   # effective conversion efficiency per unit array area (53.6 ha) from the plant's
                       # metered generation (84.7 GWh per year); includes the average effect of panel
                       # temperature, inverter conversion and soiling (Section 2a; Skoplaki & Palyvos 2009)
FSD_ELEC_MIN  = 50.0

# --- Ground heat flux (Class B) ----------------------------------------------
# Eq. (5), the hysteresis form of Santanello & Friedl (2003) and Grimmond et al. (1991),
#   G = g1*Rn + g2*(dRn/dt) + g3, fitted per season to the storage-corrected surface ground heat flux
#   (calibration/fit_ground_heat_flux.py). Rn is the measured net radiation.
G_SCHEME = "ohm"       # Eq. (5), used in the paper; "multiplicative" selects the alternative form below
G_OHM = {
  "MAM": [0.1606, -154.4, -16.17],
  "JJA": [0.1675, -187.2, -24.62],
  "SON": [0.0853, -103.5, -10.15],
  "DJF": [0.1203, -172.0, -10.09],
}

# --- Concrete heat storage (Objective Hysteresis Model; Class C) --------------
# Eq. (6): c1-c3 fitted to the continuous storage flux derived from the concrete-pile temperature
#   record (HOBO MX2303) (calibration/fit_concrete_storage.py; R2 = 0.67). Literature coefficients for fully exposed
#   concrete (0.2-0.7) are not applicable to the shaded piles.
OHM_A1 = 0.0430        # c1, storage fraction
OHM_A2 = 131.0         # c2, hysteresis timescale [s]
OHM_A3 = -4.30         # c3, offset [W m-2]
DT_SEC = 1800.0        # half-hourly step

# Alternative multiplicative ground-heat form (G_SCHEME = "multiplicative"; not used in the paper):
#   G = gamma_G(SWC)*Rn*f_season + A_G*phase + offset.
GAMMA_G_BASE = 0.110
GAMMA_G_SWC  = 0.240
G_DIURNAL_AMP= 8.6     # A_G [W m-2]
G_OFFSET     = -9.1    # [W m-2]

# --- Concrete thermal properties (for the physical storage cross-check) -------
RHO_CONCRETE = 2300.0  # [kg m-3]
CC_CONCRETE  = 880.0   # [J kg-1 K-1]
K_CONCRETE   = 1.4     # [W m-1 K-1]
D_EFF_CONCRETE = 0.04  # effective depth d_c [m] (diurnal penetration depth 0.138 m; Section 2d)

# --- Soil heat capacity for the ground-storage correction (de Vries 1963) -----
# Three-phase mixing rule, each constituent weighted by its volume fraction:
#   Cs = (1 - phi)*C_MINERAL + theta*C_WATER + (phi - theta)*C_AIR.
C_MINERAL = 1.94e6     # [J m-3 K-1] soil solids (de Vries 1963)
C_WATER   = 4.18e6     # [J m-3 K-1] liquid water
C_AIR     = RHO_AIR * CP_AIR   # [J m-3 K-1] soil air (~1.2e3; negligible but kept)
POROSITY  = 0.44       # [m3 m-3] highest value recorded by either 10-cm probe (18 March 2025)
SOIL_PLATE_DEPTH = 0.10   # heat-flux plate depth [m]

def soil_heat_capacity(theta):
    """Volumetric heat capacity of the 0-10 cm soil layer [J m-3 K-1] (de Vries 1963)."""
    import numpy as _np
    th = _np.clip(theta, 0.0, POROSITY)
    return (1.0 - POROSITY) * C_MINERAL + th * C_WATER + (POROSITY - th) * C_AIR

# --- Newton-Raphson controls (Class C) ---------------------------------------
NR_MAX_ITER = 50
NR_TOL      = 0.1      # [W m-2]
NR_PERTURB  = 0.1      # [K]

# --- Cross-validation --------------------------------------------------------
K_FOLDS = 5
SEASON_CODES = ["MAM", "JJA", "SON", "DJF"]
SEASONS_FULL = ["DJF", "MAM", "JJA", "SON"]

# Residual diurnal correction a*sin(theta) + b*cos(theta) + c (Section 2d); the intercept c makes the
#   correction unbiased. All three coefficients are Class A (fitted within each fold).
DIURNAL_CORRECTION_INTERCEPT = True

# Dew and condensation: when qs(Ts) < qa the bulk formula gives LE < 0 (downward vapor flux), which
#   occurs at dawn under RH > 95 %. Condensation bypasses the canopy resistance, so r = ra (Section 2d).
LE_ALLOW_NEGATIVE = True   # False clips LE at zero


def params_template():
    """Return a fresh parameter dict for the model kernels."""
    return {
        "albedo": ALBEDO_NOON, "use_diurnal_albedo": True, "albedo_range": ALBEDO_RANGE,
        "z0m": Z0M, "d": DISP, "z_ref": Z_REF, "u_default": U_DEFAULT,
        "ra_multiplier": RA_MULT, "ra_rh_coef": RA_RH_COEF,
        "rc_base": RC_BASE_INIT, "swc_ref": SWC_REF, "swc_rc_coef": SWC_RC_COEF,
        "vpd_ref": VPD_REF, "vpd_rc_coef": VPD_RC_COEF,
        "fsd_ref": FSD_REF, "fsd_rc_coef": FSD_RC_COEF,
        "rh_ref": RH_REF, "rh_rc_coef": RH_RC_COEF,
        "eta_effective": ETA_EFFECTIVE,
        "gamma_G_base": GAMMA_G_BASE, "gamma_G_swc_coef": GAMMA_G_SWC,
        "G_diurnal_amp": G_DIURNAL_AMP, "G_offset": G_OFFSET, "G_ohm": G_OHM,
        "H_diurnal_amp": 0.0, "H_diurnal_amp_cos": 0.0,
        "LE_diurnal_amp_sin": 0.0, "LE_diurnal_amp_cos": 0.0,
        "ohm_a1": OHM_A1, "ohm_a2": OHM_A2, "ohm_a3": OHM_A3,
        "rc_base_by_season": None,
    }
