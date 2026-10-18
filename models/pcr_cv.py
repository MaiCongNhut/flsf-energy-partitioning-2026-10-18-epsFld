"""
Principal-component regression baseline (Section 2e; Chen et al. 2012), season-stratified five-fold
time-block cross-validation on the modeling dataset. In each fold the eight predictors are standardized
with training statistics, a PCA is fitted on the training rows, and a quadratic regression on the first
two components, [PC1, PC2, PC1*PC2, PC1^2, PC2^2, 1], is fitted separately for H and LE.
Outputs: results/pcr_cv/metrics.json, predictions.csv
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths, config as C
from calibration import metrics, per_season_metrics

FEATS = ['Rn', 'Ts', 'Ta', 'Precip', 'RH', 'Ws', 'SWC', 'albedo']   # Ts = measured 10-cm soil temperature


def quad(sc):
    return np.column_stack([sc[:, 0], sc[:, 1], sc[:, 0] * sc[:, 1], sc[:, 0] ** 2, sc[:, 1] ** 2, np.ones(len(sc))])


def folds(df):
    df = df.copy(); df['fold'] = -1
    for s in C.SEASON_CODES:
        idx = np.where(df['Season'].values == s)[0]; br = np.floor(np.linspace(0, len(idx), 6)).astype(int)
        for k in range(5):
            df.loc[df.index[idx[br[k]:br[k + 1]]], 'fold'] = k + 1
    return df


def run(df, target):
    df = folds(df.dropna(subset=FEATS + [target]).reset_index(drop=True))
    X = df[FEATS].values; y = df[target].values; oof = np.full(len(df), np.nan); var = []
    for k in range(1, 6):
        val = df['fold'].values == k; cal = ~val
        mu = X[cal].mean(0); sg = X[cal].std(0, ddof=0); sg[sg == 0] = 1; Xs = (X - mu) / sg
        _, sv, Vt = np.linalg.svd(Xs[cal], full_matrices=False); var.append((sv ** 2 / (sv ** 2).sum())[:2].tolist())
        Xq = quad(Xs @ Vt.T)
        beta, *_ = np.linalg.lstsq(Xq[cal], y[cal], rcond=None); oof[val] = Xq[val] @ beta
    m = metrics(y, oof); m['MAE'] = float(np.nanmean(np.abs(oof - y)))
    return df, oof, m, per_season_metrics(y, oof, df['Season'].values), var


def main():
    df = pd.read_csv(paths.MODELING).rename(columns={'Fn': 'Rn'})
    res = {'N': int(len(df)), 'features': FEATS}; pred = None
    for tgt, name in (('Fh', 'H'), ('Fe', 'LE')):
        d, oof, m, ms, var = run(df, tgt)
        res[name] = m; res[name + '_season'] = ms; res['explained_variance_PC1_PC2_by_fold'] = var
        print('PCR %-2s R2 = %.3f  RMSE = %.2f  MAE = %.2f  Slope = %.2f  Bias = %+.2f  N = %d'
              % (name, m['R2'], m['RMSE'], m['MAE'], m['Slope'], m['Bias'], m['N']))
        p = pd.DataFrame({'DateTime': d['DateTime'], 'Season': d['Season'], name + '_obs': d[tgt], name + '_mod': oof})
        pred = p if pred is None else pred.merge(p, on=['DateTime', 'Season'], how='outer')
    json.dump(res, open(paths.out('pcr_cv', 'metrics.json'), 'w'), indent=2, default=float)
    pred.to_csv(paths.out('pcr_cv', 'predictions.csv'), index=False)


if __name__ == '__main__':
    main()
