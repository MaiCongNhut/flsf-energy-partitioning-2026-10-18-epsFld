"""
Physics kernels of the land surface model: the five-term surface energy balance of the solar farm,

    Rn = H + LE + G + E_elec + dQc,

closed by Newton-Raphson iteration on the surface temperature Ts (tolerance NR_TOL, at most NR_MAX_ITER
iterations; Section 2d).
All constants come from config.py; equation numbers refer to the paper.
"""
import numpy as np
import config as C


def month_to_season(m):
    if m in (12, 1, 2): return "DJF"
    if m in (3, 4, 5):  return "MAM"
    if m in (6, 7, 8):  return "JJA"
    return "SON"

# --- Humidity (Tetens) -------------------------------------------------------
def calc_esat(Tk):
    Tc = Tk - 273.15
    return 610.8 * np.exp(17.27 * Tc / (Tc + 237.3))          # [Pa]

def calc_q(e, P):
    return 0.622 * e / (P - 0.378 * e)                        # specific humidity

def calc_vpd(Tc, RH):
    es = 0.6108 * np.exp(17.27 * Tc / (Tc + 237.3))           # [kPa]
    return max(0.0, es * (1 - RH))

# --- Eq. (3) diurnal albedo --------------------------------------------------
def calc_albedo(hour, an, ar):
    if hour < 6 or hour > 18: return an
    h = abs(hour - 12)
    return min(an + ar * (1 - np.cos(np.pi * h / 6)), C.ALBEDO_CAP)

# --- Aerodynamic resistance (Monin-Obukhov neutral + multipliers) ------------
def calc_ra(u, z_ref, z0m, d, Ts, Ta, month, RH, p):
    z0h = 0.1 * z0m
    z = max(z_ref, d + z0m + 1)
    ra = np.log((z - d) / z0m) * np.log((z - d) / z0h) / (C.KAPPA**2 * u)
    ra = max(ra, 10.0)
    sf = 1.05 if month in (12, 1, 2) else 1.08 if month in (3, 11) else 1.02 if month in (4, 10) else 1.0
    dT = Ts - Ta
    stab = 0.95 if dT > 8 else 1.10 if dT < -5 else 1.0
    rh = max(0.9, min(1.0 + p["ra_rh_coef"] * (RH - 0.70), 1.3))
    return max(ra * sf * stab * rh * p["ra_multiplier"], 15.0)

# --- Canopy resistance (Jarvis 1976; Stewart 1988; Monteith 1965) ------------
def calc_rc(Fsd, RH, Ta_c, month, SWC, VPD, p):
    sk = month_to_season(month)
    rb = p["rc_base_by_season"][sk] if p.get("rc_base_by_season") else p["rc_base"]
    sf = 0.85 if month in (6, 7, 8) else 1.2 if month == 9 else 1.1 if month in (5, 10) \
         else 1.0 if month == 4 else 0.9 if month in (3, 11) else 1.0
    tf = 1 + 0.01 * max(0.0, Ta_c - 25)
    rh = max(0.8, min(1.0 + p["rh_rc_coef"] * (p["rh_ref"] - RH), 1.3))
    sw = max(0.6, min(1.0 - p["swc_rc_coef"] * (SWC - p["swc_ref"]), 1.4))
    vp = max(0.7, min(1.0 + p["vpd_rc_coef"] * (VPD - p["vpd_ref"]), 2.0))
    fs = max(0.8, min(1.0 + p["fsd_rc_coef"] * (Fsd - p["fsd_ref"]) / 500, 2.0))
    return max(C.RC_CLIP[0], min(rb * sf * tf * rh * sw * vp * fs, C.RC_CLIP[1]))

# --- Alternative multiplicative ground heat flux (G_SCHEME = 'multiplicative'; not used in the paper)
def calc_G(Rn, SWC, hour, month, p):
    gG = max(0.01, min(p["gamma_G_base"] + p["gamma_G_swc_coef"] * (SWC - p["swc_ref"]), 0.30))
    phase = np.sin((hour - 8) * np.pi / 12)
    sf = 1.5 if month in (6, 7, 8) else 1.0 if month in (5, 9) else 0.7 if month in (4, 10) \
         else 0.5 if month in (3, 11) else 0.3
    return gG * Rn * sf + p["G_diurnal_amp"] * (phase - 0.35) + p.get("G_offset", 0.0)

# --- Surface ground heat flux, Eq. (5) (hysteresis form; Santanello & Friedl 2003) ---
def calc_G_ohm(Rn_obs, Rn_prev, month, p):
    g1, g2, g3 = p["G_ohm"][month_to_season(month)]
    dRn = 0.0 if Rn_prev is None else (Rn_obs - Rn_prev) / C.DT_SEC
    return g1 * Rn_obs + g2 * dRn + g3

# --- Concrete heat storage (OHM; Grimmond, Cleugh & Oke 1991) ----------------
def calc_dQc(Rn_obs, Rn_prev, p):
    dRn = 0.0 if Rn_prev is None else (Rn_obs - Rn_prev) / C.DT_SEC
    return p["ohm_a1"] * Rn_obs + p["ohm_a2"] * dRn + p["ohm_a3"]

def calc_E_elec(Fsd, eta):
    return 0.0 if Fsd < C.FSD_ELEC_MIN else eta * Fsd

# --- Full energy balance residual for a single record ------------------------
def energy_balance(Ts, Fsd, Fld, Ta, RH, u, SWC, p, hour, month, Rn_obs, Rn_prev):
    alpha = calc_albedo(hour, p["albedo"], p["albedo_range"]) if p["use_diurnal_albedo"] else p["albedo"]
    # net radiation, Eq. (2)
    absorbed_lw = (C.EPSILON * Fld) if C.RN_EMISSIVITY else Fld
    Rn = (1 - alpha) * Fsd + absorbed_lw - C.EPSILON * C.SIGMA * Ts**4
    Ta_c = Ta - 273.15
    VPD = calc_vpd(Ta_c, RH)
    ra = calc_ra(u, p["z_ref"], p["z0m"], p["d"], Ts, Ta, month, RH, p)
    rc = calc_rc(Fsd, RH, Ta_c, month, SWC, VPD, p)
    qa = calc_q(RH * calc_esat(Ta), C.P_SURF)
    qs = calc_q(calc_esat(Ts), C.P_SURF)
    H  = C.RHO_AIR * C.CP_AIR * (Ts - Ta) / ra
    # dew and condensation (qs < qa): the canopy resistance is bypassed and LE < 0
    if getattr(C, "LE_ALLOW_NEGATIVE", False):
        r_le = (ra + rc) if (qs >= qa) else ra
        LE = C.RHO_AIR * C.LV * (qs - qa) / r_le
    else:
        LE = max(0.0, C.RHO_AIR * C.LV * (qs - qa) / (ra + rc))
    G  = calc_G_ohm(Rn_obs, Rn_prev, month, p) if getattr(C, "G_SCHEME", "multiplicative") == "ohm" \
         else calc_G(Rn, SWC, hour, month, p)
    E  = calc_E_elec(Fsd, p["eta_effective"])
    dQ = calc_dQc(Rn_obs, Rn_prev, p)
    return {"residual": Rn - H - LE - G - E - dQ, "Rn": Rn, "H": H, "LE": LE,
            "G": G, "E_elec": E, "dQc_dt": dQ, "Ts_c": Ts - 273.15, "ra": ra, "rc": rc}

def solve_Ts(Fsd, Fld, Ta, RH, u, SWC, p, hour, month, Rn_obs, Rn_prev):
    Ts = Ta + 3.0
    for _ in range(C.NR_MAX_ITER):
        r = energy_balance(Ts, Fsd, Fld, Ta, RH, u, SWC, p, hour, month, Rn_obs, Rn_prev)
        if abs(r["residual"]) < C.NR_TOL: break
        rp = energy_balance(Ts + C.NR_PERTURB, Fsd, Fld, Ta, RH, u, SWC, p, hour, month, Rn_obs, Rn_prev)
        dF = (rp["residual"] - r["residual"]) / C.NR_PERTURB
        if abs(dF) < 1e-10: break
        dl = max(min(-r["residual"] / dF, 5.0), -5.0)
        Ts = max(min(Ts + dl, Ta + 25), Ta - 10)
    return energy_balance(Ts, Fsd, Fld, Ta, RH, u, SWC, p, hour, month, Rn_obs, Rn_prev)

def run_model(df, p, indices=None):
    """Solve Ts and diagnose all fluxes for the rows in `indices` (default all)."""
    n = len(df)
    pm = np.zeros(n, bool); pm[:] = True if indices is None else False
    if indices is not None: pm[np.asarray(indices)] = True
    cols = ["Rn", "H", "LE", "G", "E_elec", "dQc_dt"]
    out = {c: np.full(n, np.nan) for c in cols}
    Rp = None
    dates = df["Date"].values; Fsd = df["Fsd"].values; Fld = df["Fld"].values
    Ta = df["Ta"].values + 273.15; RH = df["RH"].values / 100.0
    Ws = df["WindSpeed"].values; SWC = df["SWC"].values
    Hh = df["Hour"].values; Mo = df["Month"].values; Fn = df["Fn"].values
    for i in range(n):
        if i > 0 and dates[i] != dates[i - 1]: Rp = None
        if not pm[i]:
            Rp = Fn[i] if not np.isnan(Fn[i]) else Rp; continue
        u = Ws[i] if (not np.isnan(Ws[i]) and Ws[i] > 0.3) else p["u_default"]
        r = solve_Ts(Fsd[i], Fld[i], Ta[i], RH[i], u, SWC[i], p, int(Hh[i]), int(Mo[i]), Fn[i], Rp)
        for c in cols: out[c][i] = r[c]
        Rp = Fn[i] if not np.isnan(Fn[i]) else Rp
    return out
