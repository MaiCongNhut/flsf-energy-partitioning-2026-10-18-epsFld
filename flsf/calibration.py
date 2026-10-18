"""
Calibration and skill metrics of the land surface model (Sections 2d and 2f).

  - The per-season base canopy resistance rc_base is re-estimated inside every cross-validation fold
    (Class A) by bounded Brent minimization of the RMSE of LE on the calibration rows of the fold.
  - The diurnal residual correction a*sin(theta) + b*cos(theta) + c is fitted within each fold by
    ordinary least squares; the intercept c makes it unbiased (Class A).
  - The other coefficients are fixed (config.py). Their effect is assessed by the independent hold-out
    test (models/holdout_lsm.py).
"""
import numpy as np
from scipy.optimize import minimize_scalar
import config as C
from physics import run_model


def metrics(obs, mod):
    obs = np.asarray(obs, float); mod = np.asarray(mod, float)
    ok = np.isfinite(obs) & np.isfinite(mod); obs, mod = obs[ok], mod[ok]
    if len(obs) < 3:
        return dict(N=len(obs), R2=np.nan, RMSE=np.nan, Bias=np.nan, Slope=np.nan)
    s, i = np.polyfit(obs, mod, 1)
    r2 = 1 - ((mod - obs)**2).sum() / ((obs - obs.mean())**2).sum()
    return dict(N=int(len(obs)), R2=float(r2), RMSE=float(np.sqrt(np.mean((mod - obs)**2))),
                Bias=float(np.mean(mod - obs)), Slope=float(s))

def per_season_metrics(obs, mod, sa):
    return {s: metrics(obs[sa == s], mod[sa == s]) for s in C.SEASONS_FULL}

def fit_diurnal(y, s, c):
    """Fit y ~ a*sin + b*cos (+ c intercept if configured). Returns (a,b,c)."""
    ok = np.isfinite(y) & np.isfinite(s) & np.isfinite(c)
    if ok.sum() < 3: return 0.0, 0.0, 0.0
    cols = [s[ok], c[ok]] + ([np.ones(ok.sum())] if C.DIURNAL_CORRECTION_INTERCEPT else [])
    X = np.column_stack(cols)
    co, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    a, b = float(co[0]), float(co[1])
    cc = float(co[2]) if C.DIURNAL_CORRECTION_INTERCEPT else 0.0
    return a, b, cc

def _rmse_LE_train(rc, ti, sk, df, pb):
    rl = {s: pb["rc_base"] for s in C.SEASON_CODES}; rl[sk] = float(rc)
    p = {**pb, "rc_base_by_season": rl}
    o = run_model(df, p, indices=ti)
    obs = df.loc[ti, "Fe"].values; mod = o["LE"][ti]
    ok = np.isfinite(obs) & np.isfinite(mod)
    return 1e9 if ok.sum() == 0 else float(np.sqrt(np.mean((mod[ok] - obs[ok])**2)))

def calibrate_and_predict(df, cal, val, params_template):
    n = len(df); rcs = {}
    for s in C.SEASON_CODES:
        tr = np.array([i for i in cal if df["Season_short"].iloc[i] == s])
        if len(tr) < 30:
            rcs[s] = params_template["rc_base"]; continue
        res = minimize_scalar(_rmse_LE_train, bounds=(C.RC_LOWER, C.RC_UPPER),
                              method="bounded", options={"xatol": 5.0},
                              args=(tr, s, df, params_template))
        rcs[s] = float(res.x)
    p = {**params_template, "rc_base_by_season": rcs}
    oa = run_model(df, p, indices=np.arange(n))
    Hb, LEb, Gb = oa["H"], oa["LE"], oa["G"]
    ha = (df["Hour"].to_numpy(float) - 12.0) * np.pi / 6.0
    sh, ch = np.sin(ha), np.cos(ha)
    Fh, Fe = df["Fh"].to_numpy(), df["Fe"].to_numpy(); ca = np.asarray(cal)
    aH, bH, cH = fit_diurnal(Fh[ca] - Hb[ca], sh[ca], ch[ca])
    aL, bL, cL = fit_diurnal(Fe[ca] - LEb[ca], sh[ca], ch[ca])
    return dict(H=Hb + aH*sh + bH*ch + cH,
                LE=(LEb + aL*sh + bL*ch + cL) if getattr(C, "LE_ALLOW_NEGATIVE", False)
                   else np.maximum(0.0, LEb + aL*sh + bL*ch + cL),
                G=Gb, rc_by_season=rcs,
                autocal={"aH": aH, "bH": bH, "cH": cH, "aL": aL, "bL": bL, "cL": cL})

def make_folds(df):
    df = df.copy(); df["fold"] = -1
    for s in C.SEASON_CODES:
        idx = np.where(df["Season_short"].values == s)[0]
        br = np.floor(np.linspace(0, len(idx), C.K_FOLDS + 1)).astype(int)
        for k in range(C.K_FOLDS):
            df.loc[df.index[idx[br[k]:br[k+1]]], "fold"] = k + 1
    return df
