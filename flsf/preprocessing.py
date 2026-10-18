"""
================================================================================
 PREPROCESSING AND QUALITY CONTROL OF THE MODELING DATASET (Sections 2c and 2d)
================================================================================
 Builds the modeling dataset from the raw tower record, including the calorimetric
 storage correction of the ground heat flux. Every step is logged in a QC report.

 Steps:
   1. Read the half-hourly tower record; -9999 -> NaN; SWC = mean of the two 10-cm probes.
   2. Ground-heat-flux STORAGE CORRECTION (de Vries 1963), three-phase Cs:
          Cs = (1-phi)*C_MINERAL + theta*C_WATER + (phi-theta)*C_AIR
          G_surface = G_plate(-10cm) + Cs*dz*dTsoil/dt,   dz = SOIL_PLATE_DEPTH
      (the raw column Ts is the measured 10-cm soil temperature, T_soil in the paper),
      computed on the continuous 24-h series before the daytime subset.
   3. Exclude the commissioning period (January-April 2024); keep daytime records
      (Fsd > 0, time stamps 06:00 to 18:30 LST).
   4. Physical-limit tests (Foken 2008; Mauder et al. 2013).
   5. Papale et al. (2006) double-difference spike detection, per season.
   6. Ts flat-line flag (driver QC).
   7. Dew flag (RH > 95 %, LE < 0): kept, because dew and condensation are physically
      real and simulated by the model (Section 2d).
   8. Remove implausible midday records (10:00-15:30 LST with LE < 0 or H < -50 W m-2).

 Returns (df_clean, qc_report). G_TARGET='surface' (used in the paper) sets the observed G
 to the storage-corrected surface flux.
================================================================================
"""
import numpy as np, pandas as pd
import config as C

LIMITS = {"Fh": (-200, 800), "Fe": (-100, 800), "Fg": (-200, 400), "Fn": (-200, 1200)}

def _season(m):
    return "DJF" if m in (12,1,2) else "MAM" if m in (3,4,5) else "JJA" if m in (6,7,8) else "SON"

def _papale_spikes(df, col, z=5.5):
    flag = pd.Series(False, index=df.index)
    for _, g in df.groupby("Season"):
        x = g[col].values; d = np.full(len(x), np.nan)
        d[1:-1] = (x[1:-1]-x[:-2]) - (x[2:]-x[1:-1])
        md = np.nanmedian(d); mad = np.nanmedian(np.abs(d - md))
        if not np.isfinite(mad) or mad == 0: continue
        flag.loc[g.index[np.abs(d - md) > z*mad/0.6745]] = True
    return flag

def load_observed(data_file, G_TARGET="surface"):
    r = pd.read_csv(data_file, skiprows=2, header=0).rename(columns={"xlDateTime": "DateTime"})
    r.replace(-9999, np.nan, inplace=True)
    r["DateTime"] = pd.to_datetime(r["DateTime"], errors="coerce")
    r = r.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    r["SWC"] = r[["Sws_10cma", "Sws_10cmb"]].mean(axis=1)
    r["Year"]=r["DateTime"].dt.year; r["Month"]=r["DateTime"].dt.month
    r["Hour"]=r["DateTime"].dt.hour; r["Date"]=r["DateTime"].dt.date; r["Season"]=r["Month"].map(_season)
    rep = {"raw_rows": int(len(r))}

    # 2. surface-G storage correction (continuous series)
    same = r["Date"].eq(r["Date"].shift())
    r["Cs"] = C.soil_heat_capacity(r["SWC"])
    r["dTs_dt"] = ((r["Ts"] - r["Ts"].shift()) / 1800.0).where(same, np.nan)
    r["Fg_surface"] = r["Fg"] + r["Cs"] * C.SOIL_PLATE_DEPTH * r["dTs_dt"]
    rep["Ts_flatline"] = int((r["Ts"].diff() == 0).sum())

    # 3. daytime, post-commissioning
    d = r[(r["Fsd"] > 0) & (r["Hour"].between(6, 18)) &
          ~((r["Year"] == 2024) & (r["Month"].isin([1, 2, 3, 4])))].copy()
    rep["daytime_post_commissioning"] = int(len(d))

    # 4. physical limits
    lim = pd.Series(False, index=d.index)
    for c, (lo, hi) in LIMITS.items():
        b = ((d[c] < lo) | (d[c] > hi)).fillna(False); lim |= b; rep[f"limit_{c}"] = int(b.sum())
    # 5. Papale spikes
    spike = pd.Series(False, index=d.index)
    for c in ("Fh", "Fe", "Fg"):
        sp = _papale_spikes(d, c); spike |= sp; rep[f"spike_{c}"] = int(sp.sum())
    # 7-8. dew (keep) & midday implausible (remove)
    dew = (d["RH"] > 95) & (d["Fe"] < 0); rep["dew_kept"] = int(dew.sum())
    midbad = (d["Hour"].between(10, 15)) & ((d["Fe"] < 0) | (d["Fh"] < -50)); rep["midday_bad"] = int(midbad.sum())

    remove = (lim | spike | midbad) & ~dew
    rep["removed"] = int(remove.sum())
    d = d[~remove].copy()
    rep["retained"] = int(len(d)); rep["retained_pct"] = round(100*len(d)/rep["daytime_post_commissioning"], 1)

    d["Gobs"] = d["Fg_surface"] if G_TARGET == "surface" else d["Fg"]
    d["Season_short"] = d["Season"]; d["WindSpeed"] = d["Ws"]
    d = d.dropna(subset=["Fsd", "Fld", "Ta", "RH", "SWC", "Fn", "Fe", "Fh"]).reset_index(drop=True)
    rep["final_rows"] = int(len(d))
    return d, rep
