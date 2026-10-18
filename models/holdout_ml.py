#!/usr/bin/env python3
"""
Independent chronological hold-out test of the two machine-learning baselines (PCR, CNN).

Same split as holdout_lsm.py:
  Training period : 1 May 2024 to 30 Apr 2025   (all fitting, scaling and early stopping)
  Test period     : 1 May 2025 to  1 Feb 2026   (never seen during training)

PCR  Standardization, principal-component analysis and the quadratic least-squares
     regression on [PC1, PC2, PC1*PC2, PC1^2, PC2^2, 1] are fitted on training rows only
     (data: data/derived/modeling_dataset.csv, the quality-controlled record, N = 14,421).
CNN  The architecture and settings of Section 2e are unchanged (two 1-D convolutional layers,
     16 and 32 filters, kernel 3; global average pooling; dense 32 and 2; Adam, learning
     rate 1e-3, batch 128, at most 25 epochs, early stopping with patience 5). Input and
     target scaling use training windows only. Early stopping monitors a validation subset
     drawn from the training period only: the last 10 % (in time) of the training windows
     of each season. Input windows (24 consecutive daytime records) that straddle the split are discarded, and
     records with a failed target quality flag are excluded from training and scoring
     (data: data/derived/cnn_daytime_series.csv, as in the cross-validation). Because the result depends on the
     random seed of the weight initialization, the network is trained with five seeds (1 to 5); the mean and
     range are reported.

Usage: python holdout_ml.py pcr
       python holdout_ml.py cnn <seed>      (one seed per call; results appended)
       python holdout_ml.py summary
"""
import os, sys, json, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths
OUT = os.path.dirname(paths.out('holdout', 'x'))
SPLIT = pd.Timestamp('2025-05-01 00:00')
SEAS = ['MAM', 'JJA', 'SON', 'DJF']


def season(m):
    return 'DJF' if m in (12, 1, 2) else 'MAM' if m in (3, 4, 5) else 'JJA' if m in (6, 7, 8) else 'SON'


def metrics(o, m):
    o = np.asarray(o, float); m = np.asarray(m, float)
    ok = np.isfinite(o) & np.isfinite(m); o, m = o[ok], m[ok]
    s, i = np.polyfit(o, m, 1)
    return dict(N=int(len(o)), R2=float(1 - ((m - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()),
                RMSE=float(np.sqrt(np.mean((m - o) ** 2))), MAE=float(np.mean(np.abs(m - o))),
                Bias=float(np.mean(m - o)), Slope=float(s))


def run_pcr():
    FEATS = ['Rn', 'Ts', 'Ta', 'Precip', 'RH', 'Ws', 'SWC', 'albedo']
    df = pd.read_csv(paths.MODELING).rename(columns={'Fn': 'Rn'})
    dt = pd.to_datetime(df['DateTime']); df['Season'] = dt.dt.month.map(season)
    res = {'N_all': int(len(df))}
    for tgt, name in (('Fh', 'H'), ('Fe', 'LE')):
        d = df.dropna(subset=FEATS + [tgt]).reset_index(drop=True)
        t = pd.to_datetime(d['DateTime']); tr = (t < SPLIT).values; te = ~tr
        X = d[FEATS].values; y = d[tgt].values
        mu = X[tr].mean(0); sg = X[tr].std(0); sg[sg == 0] = 1; Xs = (X - mu) / sg
        Vt = np.linalg.svd(Xs[tr], full_matrices=False)[2]; sc = Xs @ Vt.T
        Q = np.column_stack([sc[:, 0], sc[:, 1], sc[:, 0] * sc[:, 1], sc[:, 0] ** 2, sc[:, 1] ** 2, np.ones(len(sc))])
        b, *_ = np.linalg.lstsq(Q[tr], y[tr], rcond=None); pred = Q @ b
        ev = (np.linalg.svd(Xs[tr], full_matrices=False)[1] ** 2); ev = ev / ev.sum()
        res[name] = {'N_train': int(tr.sum()), 'N_test': int(te.sum()),
                     'test': metrics(y[te], pred[te]), 'train': metrics(y[tr], pred[tr]),
                     'test_by_season': {s: metrics(y[te & (d['Season'] == s).values], pred[te & (d['Season'] == s).values]) for s in SEAS},
                     'variance_explained_PC1_PC2_training': [float(ev[0]), float(ev[1])]}
        print('PCR %s  test R2=%.3f RMSE=%.1f (N=%d) | train R2=%.3f'
              % (name, res[name]['test']['R2'], res[name]['test']['RMSE'], te.sum(), res[name]['train']['R2']))
    json.dump(res, open(os.path.join(OUT, 'holdout_pcr.json'), 'w'), indent=2)


def run_cnn(seed):
    import cnn as CNN
    CNN.rng = np.random.default_rng(seed)          # weight initialization and batch order
    WINDOW = 24; FEATS = ['Fsd', 'Fld', 'Ta', 'RH', 'Ws', 'SWC', 'albedo', 'Hour', 'DOY']
    df = pd.read_csv(paths.CNN_SERIES).rename(columns={'Fn': 'Rn'})
    dt = pd.to_datetime(df['DateTime'])
    df['Hour'] = dt.dt.hour; df['DOY'] = dt.dt.dayofyear; df['Month'] = dt.dt.month
    df['albedo'] = (df['Fsu'] / df['Fsd']).clip(0, 1); df['Season'] = df['Month'].map(season)
    df['bad'] = df['QC_bad'].astype(bool)
    df = df.dropna(subset=FEATS).reset_index(drop=True)
    period = (pd.to_datetime(df['DateTime']) >= SPLIT).astype(int).values      # 0 train, 1 test
    F = df[FEATS].to_numpy(np.float32); Y = df[['Fh', 'Fe']].to_numpy(np.float32)
    bad = df['bad'].to_numpy(); seas = df['Season'].to_numpy(); tstamp = pd.to_datetime(df['DateTime']).values
    Xw, Yw, pw, bw, sw, tw = [], [], [], [], [], []
    for i in range(WINDOW - 1, len(df)):
        seg = period[i - WINDOW + 1:i + 1]
        if np.all(seg == seg[-1]):
            Xw.append(F[i - WINDOW + 1:i + 1]); Yw.append(Y[i]); pw.append(period[i]); bw.append(bad[i])
            sw.append(seas[i]); tw.append(tstamp[i])
    Xw = np.array(Xw); Yw = np.array(Yw); pw = np.array(pw); bw = np.array(bw); sw = np.array(sw); tw = np.array(tw)
    trn = (pw == 0) & ~bw; tst = (pw == 1) & ~bw
    # chronological validation subset: last 10 % of each season's training windows
    vmask = np.zeros(len(Xw), bool)
    for s in SEAS:
        idx = np.where(trn & (sw == s))[0]; idx = idx[np.argsort(tw[idx])]
        vmask[idx[int(0.9 * len(idx)):]] = True
    fit = trn & ~vmask
    mu = Xw[trn].reshape(-1, Xw.shape[2]).mean(0); sd = Xw[trn].reshape(-1, Xw.shape[2]).std(0); sd[sd == 0] = 1
    my = Yw[trn].mean(0); sy = Yw[trn].std(0)
    Xc = (Xw - mu) / sd; Yc = (Yw - my) / sy
    model = CNN.CNN1D(n_features=Xw.shape[2], n_outputs=2)
    t0 = time.time()
    hist = CNN.train(model, Xc[fit], Yc[fit], Xc[vmask], Yc[vmask], epochs=25, batch=128, lr=1e-3, patience=5, verbose=False)
    pred = CNN.predict_in_batches(model, Xc) * sy + my
    r = {'seed': seed, 'epochs_run': len(hist['val']), 'best_epoch': int(np.argmin(hist['val']) + 1),
         'N_windows': int(len(Xw)), 'N_fit': int(fit.sum()), 'N_validation': int(vmask.sum()), 'N_test': int(tst.sum()),
         'H': metrics(Yw[tst, 0], pred[tst, 0]), 'LE': metrics(Yw[tst, 1], pred[tst, 1]),
         'H_by_season': {s: metrics(Yw[tst & (sw == s), 0], pred[tst & (sw == s), 0]) for s in SEAS},
         'LE_by_season': {s: metrics(Yw[tst & (sw == s), 1], pred[tst & (sw == s), 1]) for s in SEAS},
         'seconds': time.time() - t0}
    print('CNN seed %d  test H R2=%.3f RMSE=%.1f | LE R2=%.3f RMSE=%.1f  N=%d  epochs %d  %.0fs'
          % (seed, r['H']['R2'], r['H']['RMSE'], r['LE']['R2'], r['LE']['RMSE'], r['N_test'], r['epochs_run'], r['seconds']))
    p = os.path.join(OUT, 'holdout_cnn_seeds.json')
    allr = json.load(open(p)) if os.path.exists(p) else {}
    allr[str(seed)] = r; json.dump(allr, open(p, 'w'), indent=2)


def summary():
    allr = json.load(open(os.path.join(OUT, 'holdout_cnn_seeds.json')))
    out = {}
    for f in ('H', 'LE'):
        for k in ('R2', 'RMSE'):
            v = np.array([allr[s][f][k] for s in allr])
            out['%s_%s' % (f, k)] = {'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max()), 'n_seeds': int(len(v))}
    json.dump(out, open(os.path.join(OUT, 'holdout_cnn_summary.json'), 'w'), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else 'pcr'
    if a == 'pcr': run_pcr()
    elif a == 'cnn': run_cnn(int(sys.argv[2]))
    else: summary()
