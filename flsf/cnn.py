"""
Minimal 1D-CNN implementation in pure numpy.
Architecture:
  Input (B, T=24, F=9)
  Conv1D(F -> 16, k=3) + ReLU
  Conv1D(16 -> 32, k=3) + ReLU
  GlobalAvgPool over time axis
  Dense(32 -> 32) + ReLU
  Dense(32 -> 2)   # outputs H and LE (normalized)

Trained with Adam on MSE.
"""
import numpy as np
import time

rng = np.random.default_rng(42)


# ----- Layers -----
class Conv1D:
    def __init__(self, in_ch, out_ch, k):
        self.W = rng.standard_normal((out_ch, in_ch, k)).astype(np.float32) * np.sqrt(2.0 / (in_ch * k))
        self.b = np.zeros(out_ch, dtype=np.float32)
        self.k = k

    def forward(self, x):
        # x: (B, T, Cin)
        self.x = x
        B, T, Cin = x.shape
        Cout, _, K = self.W.shape
        Tp = T - K + 1
        # Build windowed view via stride tricks
        # x_view: (B, Tp, K, Cin)
        s0, s1, s2 = x.strides
        x_view = np.lib.stride_tricks.as_strided(
            x, shape=(B, Tp, K, Cin), strides=(s0, s1, s1, s2), writeable=False
        )
        # out = einsum('btkc,ock->bto', x_view, W) + b
        self.x_view = x_view
        out = np.einsum("btkc,ock->bto", x_view, self.W).astype(np.float32) + self.b
        return out

    def backward(self, dout):
        # dout: (B, Tp, Cout)
        x_view = self.x_view
        W = self.W
        B, T, Cin = self.x.shape
        Cout, _, K = W.shape
        Tp = T - K + 1
        # dW: (Cout, Cin, K)
        self.dW = np.einsum("bto,btkc->ock", dout, x_view).astype(np.float32)
        self.db = dout.sum(axis=(0, 1)).astype(np.float32)
        # dx: propagate through conv. dx[b,t,c] = sum_{k, o} dout[b, t-k, o] * W[o, c, k] (valid indices)
        dx = np.zeros_like(self.x)
        # dout_full: (B, Tp, Cout), want to scatter back
        # For each k in [0, K), position t in dout contributes to dx at position t+k
        # Precompute d_per_k = einsum('bto,oc->btc', dout, W[:,:,k])  -> (B, Tp, Cin)
        for k in range(K):
            d_per_k = np.einsum("bto,oc->btc", dout, W[:, :, k])
            dx[:, k:k + Tp, :] += d_per_k
        return dx


class ReLU:
    def forward(self, x):
        self.mask = (x > 0).astype(x.dtype)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class GlobalAvgPool1D:
    def forward(self, x):
        # x: (B, T, C) -> (B, C)
        self.shape = x.shape
        return x.mean(axis=1)

    def backward(self, dout):
        B, T, C = self.shape
        return np.broadcast_to(dout[:, None, :] / T, self.shape).astype(dout.dtype)


class Dense:
    def __init__(self, in_f, out_f):
        self.W = rng.standard_normal((in_f, out_f)).astype(np.float32) * np.sqrt(2.0 / in_f)
        self.b = np.zeros(out_f, dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T


# ----- Model -----
class CNN1D:
    def __init__(self, n_features, n_outputs=2):
        """Two Conv1D layers (16 and 32 filters, kernel 3, ReLU), global average pooling, Dense(32, ReLU) and
        Dense(n_outputs) (Section 2e of the paper)."""
        self.c1 = Conv1D(n_features, 16, 3)
        self.r1 = ReLU()
        self.c2 = Conv1D(16, 32, 3)
        self.r2 = ReLU()
        self.pool = GlobalAvgPool1D()
        self.d1 = Dense(32, 32)
        self.r3 = ReLU()
        self.d2 = Dense(32, n_outputs)
        self.layers = [self.c1, self.r1, self.c2, self.r2,
                       self.pool, self.d1, self.r3, self.d2]

    def forward(self, x):
        for lyr in self.layers:
            x = lyr.forward(x)
        return x

    def backward(self, dout):
        for lyr in reversed(self.layers):
            dout = lyr.backward(dout)
        return dout

    def params(self):
        """Return (layer_ref, attr_name) so Adam can mutate parameters in-place."""
        out = []
        for lyr in self.layers:
            if hasattr(lyr, "W"):
                out.append((lyr, "W"))
                out.append((lyr, "b"))
        return out

    def grads(self):
        out = []
        for lyr in self.layers:
            if hasattr(lyr, "W"):
                out.append(lyr.dW)
                out.append(lyr.db)
        return out


# ----- Adam optimizer -----
class Adam:
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        params = model.params()
        self.m = [np.zeros_like(getattr(lyr, name)) for lyr, name in params]
        self.v = [np.zeros_like(getattr(lyr, name)) for lyr, name in params]

    def step(self):
        self.t += 1
        params = self.model.params()
        grads = self.model.grads()
        for i, ((lyr, name), g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p = getattr(lyr, name)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
            setattr(lyr, name, p)


def mse_loss(pred, target):
    diff = pred - target
    loss = (diff * diff).mean()
    dpred = (2.0 / diff.size) * diff
    return loss, dpred


# ----- Training loop -----
def train(model, X, Y, X_val, Y_val, epochs=30, batch=128, lr=1e-3, patience=6, verbose=True):
    opt = Adam(model, lr=lr)
    n = X.shape[0]
    history = {"train": [], "val": []}
    best_val = np.inf
    best_state = None
    patience_ct = 0
    for ep in range(epochs):
        t0 = time.time()
        idx = rng.permutation(n)
        X, Y = X[idx], Y[idx]
        losses = []
        for i in range(0, n, batch):
            xb = X[i:i + batch]
            yb = Y[i:i + batch]
            pred = model.forward(xb)
            loss, dpred = mse_loss(pred, yb)
            model.backward(dpred)
            opt.step()
            losses.append(loss)
        train_loss = float(np.mean(losses))
        # Validation
        pred_val = predict_in_batches(model, X_val, batch=512)
        val_loss = float(((pred_val - Y_val) ** 2).mean())
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if verbose:
            print(f"Epoch {ep + 1:3d} | train {train_loss:.4f} | val {val_loss:.4f} | {time.time() - t0:.1f}s")
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = _snapshot(model)
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= patience:
                if verbose:
                    print(f"Early stop at epoch {ep + 1}")
                break
    _restore(model, best_state)
    return history


def predict_in_batches(model, X, batch=512):
    outs = []
    for i in range(0, len(X), batch):
        outs.append(model.forward(X[i:i + batch]))
    return np.concatenate(outs, axis=0)


def _snapshot(model):
    out = []
    for lyr in model.layers:
        if hasattr(lyr, "W"):
            out.append((lyr.W.copy(), lyr.b.copy()))
    return out


def _restore(model, state):
    if state is None:
        return
    j = 0
    for lyr in model.layers:
        if hasattr(lyr, "W"):
            lyr.W = state[j][0]
            lyr.b = state[j][1]
            j += 1
