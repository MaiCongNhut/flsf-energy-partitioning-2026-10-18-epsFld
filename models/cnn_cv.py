"""
One-dimensional CNN baseline (Section 2e), season-stratified five-fold time-block cross-validation.

Input: the continuous daytime series (data/derived/cnn_daytime_series.csv). Each sample is a window of
24 consecutive daytime records of nine standardized drivers; the network predicts H and LE at the last
record. Windows that straddle a fold boundary are discarded, and records flagged by the modeling
screening (QC_bad) are excluded as targets from training, early stopping and evaluation.
Architecture and training settings are fixed (no hyperparameter search): two Conv1D layers (16, 32
filters, kernel 3, ReLU), global average pooling, Dense(32, ReLU), Dense(2); Adam, learning rate 1e-3,
batch 128, up to 25 epochs, early stopping (patience 5) on a random 10 % of the training windows.

Usage:  python models/cnn_cv.py             all five folds in one process (as for the results in the paper)
        python models/cnn_cv.py --fold k    one fold (k = 1..5), saved to results/cnn_cv/fold_k.npz
        python models/cnn_cv.py --merge     pool the saved folds and write metrics.json
The network is implemented in NumPy (flsf/cnn.py); results vary slightly between platforms.
"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flsf'))
import paths, config as C
import cnn as CNN

WINDOW = 24
FEATS = ['Fsd', 'Fld', 'Ta', 'RH', 'Ws', 'SWC', 'albedo', 'Hour', 'DOY']


def windows():
    df = pd.read_csv(paths.CNN_SERIES); dt = pd.to_datetime(df['DateTime'])
    df['Hour'] = dt.dt.hour; df['DOY'] = dt.dt.dayofyear
    df['albedo'] = (df['Fsu'] / df['Fsd']).clip(0, 1)
    df['bad'] = df['QC_bad'].astype(bool)
    df = df.dropna(subset=FEATS).reset_index(drop=True)
    df['fold'] = -1
    for s in C.SEASON_CODES:
        idx = np.where(df['Season'].values == s)[0]; br = np.floor(np.linspace(0, len(idx), 6)).astype(int)
        for k in range(5):
            df.loc[df.index[idx[br[k]:br[k + 1]]], 'fold'] = k + 1
    F = df[FEATS].to_numpy(np.float32); Y = df[['Fh', 'Fe']].to_numpy(np.float32)
    fold = df['fold'].to_numpy(); bad = df['bad'].to_numpy()
    Xw, Yw, fw, bw, iw = [], [], [], [], []
    for i in range(WINDOW - 1, len(df)):
        seg = fold[i - WINDOW + 1:i + 1]
        if np.all(seg == seg[-1]):
            Xw.append(F[i - WINDOW + 1:i + 1]); Yw.append(Y[i]); fw.append(fold[i]); bw.append(bad[i]); iw.append(i)
    return df, np.array(Xw), np.array(Yw), np.array(fw), np.array(bw), np.array(iw)


def train_fold(Xw, Yw, fw, bw, k):
    cal = (fw != k) & ~bw
    mu = Xw[cal].reshape(-1, Xw.shape[2]).mean(0); sd = Xw[cal].reshape(-1, Xw.shape[2]).std(0); sd[sd == 0] = 1
    my = Yw[cal].mean(0); sy = Yw[cal].std(0); sy[sy == 0] = 1
    Xc = (Xw - mu) / sd; Yc = (Yw - my) / sy
    ci = np.where(cal)[0]; np.random.seed(42); np.random.shuffle(ci); nv = int(0.1 * len(ci))
    model = CNN.CNN1D(n_features=Xw.shape[2], n_outputs=2)
    CNN.train(model, Xc[ci[nv:]], Yc[ci[nv:]], Xc[ci[:nv]], Yc[ci[:nv]], epochs=25, batch=128, lr=1e-3, patience=5, verbose=False)
    return CNN.predict_in_batches(model, Xc[fw == k]) * sy + my


def full_metrics(o, m):
    ok = np.isfinite(o) & np.isfinite(m); o, m = o[ok], m[ok]; s, _ = np.polyfit(o, m, 1)
    return dict(N=int(len(o)), R2=float(1 - ((m - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()),
                RMSE=float(np.sqrt(np.mean((m - o) ** 2))), MAE=float(np.mean(np.abs(m - o))),
                Slope=float(s), Bias=float(np.mean(m - o)))


def evaluate(df, Yw, fw, bw, iw, oof):
    good = ~bw; seas = df['Season'].values[iw]
    res = {'N': int(good.sum()), 'windows': int(len(Yw)), 'bad_target_windows': int(bw.sum()),
           'H': full_metrics(Yw[good, 0], oof[good, 0]), 'LE': full_metrics(Yw[good, 1], oof[good, 1])}
    for j, name in ((0, 'H'), (1, 'LE')):
        res[name + '_season'] = {s: full_metrics(Yw[good & (seas == s), j], oof[good & (seas == s), j]) for s in C.SEASONS_FULL}
    json.dump(res, open(paths.out('cnn_cv', 'metrics.json'), 'w'), indent=2)
    pd.DataFrame({'DateTime': df['DateTime'].values[iw], 'Season': seas, 'fold': fw, 'QC_bad': bw,
                  'H_obs': Yw[:, 0], 'H_mod': oof[:, 0], 'LE_obs': Yw[:, 1], 'LE_mod': oof[:, 1]}
                 ).to_csv(paths.out('cnn_cv', 'predictions.csv'), index=False)
    print('CNN H R2 = %.3f RMSE = %.1f | LE R2 = %.3f RMSE = %.1f  (N = %d)'
          % (res['H']['R2'], res['H']['RMSE'], res['LE']['R2'], res['LE']['RMSE'], res['N']))


def main():
    df, Xw, Yw, fw, bw, iw = windows()
    oof = np.full((len(Yw), 2), np.nan)
    if '--fold' in sys.argv:
        k = int(sys.argv[sys.argv.index('--fold') + 1]); t = time.time()
        np.savez(paths.out('cnn_cv', 'fold_%d.npz' % k), pred=train_fold(Xw, Yw, fw, bw, k))
        print('fold %d done (%.0f s)' % (k, time.time() - t)); return
    if '--merge' in sys.argv:
        for k in range(1, 6):
            oof[fw == k] = np.load(paths.out('cnn_cv', 'fold_%d.npz' % k))['pred']
    else:
        for k in range(1, 6):
            oof[fw == k] = train_fold(Xw, Yw, fw, bw, k)
    evaluate(df, Yw, fw, bw, iw, oof)


if __name__ == '__main__':
    main()
