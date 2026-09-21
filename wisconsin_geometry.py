"""Affine geometry along data segments, before and after training.

Companion code for the section on the Breast Cancer Wisconsin experiment.
Self-contained: NumPy, SciPy-free, no deep learning framework.  The only
external dependency is scikit-learn, and only to load the dataset.

    pip install numpy matplotlib scikit-learn
    python wisconsin_geometry.py

--------------------------------------------------------------------------
MODEL.  Exactly the model of the paper, in the paper's parametrisation:

    z_1(x)   = b_1   + W_1 x                        W_1  ~ N(0, gamma_1 I_d)
    z_l(x)   = b_l   + W_l h_{l-1} / sqrt(n_{l-1})  W_l  ~ N(0, gamma_l)
    f(x)     = b_out + a   h_L     / sqrt(n_L)
    Var(b_l) = beta_l = sigma_b^2 > 0

The condition beta_l > 0 is required by the theorems.  The usual library
default of a zero bias would violate it, so we do not use a framework
initializer.  He scaling in this parametrisation is gamma_1 = 2/d and
gamma_l = 2 for l >= 2, which gives A_l = l sigma_b^2 and B_l = 2/d and keeps
Var(z_l) = l sigma_b^2 + 2 = O(1), so the network is trainable.

--------------------------------------------------------------------------
MEASUREMENT.  Along a segment x(t) = x0 + t (x1 - x0), t in [0,1], every
preactivation is affine on each cell inherited from the preceding layers.
All hidden switches are therefore located EXACTLY by solving linear
equations, layer by layer.  No grid and no discretisation of t is used.
We record the switches per layer and, separately, the number of kinks of the
scalar output across the ordered switch locations.

--------------------------------------------------------------------------
THEORY.  A segment is a chord.  With v = (x1-x0)/|x1-x0|, s0 = <x0,v>,
s1 = s0 + |x1-x0| and y = x0 - s0 v (so y is perpendicular to v), the
layer-l intercept and slope variances along the chord are
A'_l = A_l + B_l |y|^2 and B_l.  The corollary for constant variances then
gives the expected layer-l switch count on the segment in closed form:

    n_l/pi * [ arctan(s1 sqrt(B_l/A'_l)) - arctan(s0 sqrt(B_l/A'_l)) ].

No quadrature and no simulation enter the prediction.

--------------------------------------------------------------------------
SEGMENT FAMILIES.  within-malignant, within-benign, between-class, and an
off-manifold CONTROL whose endpoints are drawn from N(0, I_d).  The control
has the same per-coordinate scale as the standardised features but does not
lie on the data manifold.  It separates a genuine data-dependent
reorganisation from a global contraction of the network; without it the main
claim would not be identifiable.

--------------------------------------------------------------------------
REPRODUCIBILITY.  Base seed 42; independent networks use
default_rng([42, s]) for the tables, [42, 1000+s] for the trajectory, and
[42, 7000+s] for the robustness sweep.  Two implementation errors found
during the study and
corrected here, recorded for the reproducibility statement:
  * the trajectory reset the Adam state at every checkpoint, so it did not
    reproduce the main table at epoch 300; the optimizer state is now carried
    through the run (train() takes and returns `state`);
  * init_net() and theory_segment() did not forward sigma_b2 to the
    covariance recursion, so a robustness sweep returned identical rows for
    all values of sigma_b^2.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer

BASE_SEED = 42
SIGMA_B2 = 0.1
D_IN = 30


# ============================================================== theory =====
def AB(L, d, sigma_b2=SIGMA_B2):
    """A_l, B_l under He scaling in the paper's parametrisation."""
    beta = [sigma_b2] * L
    gamma = [2.0 / d] + [2.0] * (L - 1)
    A, B = [beta[0]], [gamma[0]]
    for l in range(1, L):
        A.append(beta[l] + gamma[l] * A[-1] / 2)
        B.append(gamma[l] * B[-1] / 2)
    return A, B, beta, gamma


def theory_segment(x0, x1, widths, d=D_IN, sigma_b2=SIGMA_B2):
    """Expected switches per layer on the segment, in closed form."""
    A, B, _, _ = AB(len(widths), d, sigma_b2)
    u = x1 - x0
    seg_len = np.linalg.norm(u)
    v = u / seg_len
    s0 = float(x0 @ v)
    s1 = s0 + seg_len
    y2 = float(x0 @ x0) - s0 ** 2            # |y|^2, y perpendicular to v
    out = []
    for l, n in enumerate(widths):
        Ap = A[l] + B[l] * y2
        c = np.sqrt(B[l] / Ap)
        out.append(n / np.pi * (np.arctan(s1 * c) - np.arctan(s0 * c)))
    return np.array(out)


# ============================================================= network =====
def init_net(widths, d=D_IN, rng=None, sigma_b2=SIGMA_B2):
    rng = rng or np.random.default_rng(BASE_SEED)
    _, _, beta, gamma = AB(len(widths), d, sigma_b2)
    params, n_prev = [], d
    for l, n in enumerate(widths):
        b = np.sqrt(beta[l]) * rng.standard_normal(n)
        W = np.sqrt(gamma[l]) * rng.standard_normal((n, n_prev))
        if l > 0:
            W = W / np.sqrt(n_prev)
        params.append([b, W])
        n_prev = n
    a = rng.standard_normal(widths[-1]) / np.sqrt(widths[-1])
    return params, [a, np.array(0.0)]


def forward(params, out, X):
    h = X
    for b, W in params:
        h = np.maximum(0.0, b[None, :] + h @ W.T)
    a, b_out = out
    return b_out + h @ a


def loss_and_grads(params, out, X, y):
    """Binary cross entropy on the logit; hand-written backward pass."""
    hs, zs, h = [X], [], X
    for b, W in params:
        z = b[None, :] + h @ W.T
        h = np.maximum(0.0, z)
        zs.append(z); hs.append(h)
    a, b_out = out
    logit = b_out + h @ a
    loss = np.mean(np.maximum(logit, 0) - logit * y
                   + np.log1p(np.exp(-np.abs(logit))))
    g = (1.0 / (1.0 + np.exp(-logit)) - y) / len(y)
    ga, gb_out = hs[-1].T @ g, g.sum()
    gh = np.outer(g, a)
    gp = [None] * len(params)
    for l in range(len(params) - 1, -1, -1):
        gz = gh * (zs[l] > 0)
        gp[l] = [gz.sum(0), gz.T @ hs[l]]
        gh = gz @ params[l][1]
    return loss, gp, [ga, gb_out]


def train(params, out, X, y, epochs=300, lr=3e-3, state=None):
    """Full-batch Adam.  Pass `state` to continue a run without resetting the
    optimizer; this is required when measuring along a training trajectory."""
    flat = [t for pair in params for t in pair] + list(out)
    if state is None:
        state = {'m': [np.zeros_like(t) for t in flat],
                 'v': [np.zeros_like(t) for t in flat], 'step': 0}
    ms, vs = state['m'], state['v']
    b1, b2, eps = 0.9, 0.999, 1e-8
    loss = None
    for _ in range(epochs):
        state['step'] += 1
        k = state['step']
        loss, gp, go = loss_and_grads(params, out, X, y)
        gflat = [t for pair in gp for t in pair] + list(go)
        for i, (t, gt) in enumerate(zip(flat, gflat)):
            ms[i] = b1 * ms[i] + (1 - b1) * gt
            vs[i] = b2 * vs[i] + (1 - b2) * gt ** 2
            t -= lr * (ms[i] / (1 - b1 ** k)) / (
                np.sqrt(vs[i] / (1 - b2 ** k)) + eps)
    return loss, state


# ========================================================= measurement =====
def measure_segment(params, out, x0, x1):
    """Exact per-layer switch counts and output kink count on x(t), t in [0,1]."""
    u = x1 - x0

    def hidden(ts, upto):
        h = x0[None, :] + ts[:, None] * u[None, :]
        for l in range(upto + 1):
            b, W = params[l]
            h = np.maximum(0.0, b[None, :] + h @ W.T)
        return h

    per_layer, sw = [], []
    b1, W1 = params[0]
    a0, a1 = b1 + W1 @ x0, W1 @ u
    with np.errstate(divide='ignore', invalid='ignore'):
        r = -a0 / a1
    new = list(r[np.isfinite(r) & (r > 0) & (r < 1)])
    per_layer.append(len(new)); sw += new

    for l in range(1, len(params)):
        grid = np.sort(np.array([0.0] + sw + [1.0]))
        H = hidden(grid, l - 1)
        b, W = params[l]
        Z = b[None, :] + H @ W.T
        sg = np.sign(Z)
        k, j = np.nonzero(sg[:-1, :] * sg[1:, :] < 0)
        z0, z1 = Z[k, j], Z[k + 1, j]
        t0, t1 = grid[k], grid[k + 1]
        new = list(t0 - z0 * (t1 - t0) / (z1 - z0))
        per_layer.append(len(new)); sw += new

    pts = np.sort(np.array([0.0] + sw + [1.0]))
    vals = forward(params, out, x0[None, :] + pts[:, None] * u[None, :])
    slopes = np.diff(vals) / np.diff(pts)
    kinks = int((~np.isclose(slopes[:-1], slopes[1:],
                             rtol=1e-8, atol=1e-12)).sum())
    return np.array(per_layer), kinks


# =============================================================== data ======
def get_data():
    data = load_breast_cancer()
    X = (data.data - data.data.mean(0)) / data.data.std(0)
    return X, data.target.astype(float)


def split(X, y, rs, frac=0.7):
    perm = rs.permutation(len(X))
    n = int(frac * len(X))
    return X[perm[:n]], y[perm[:n]], X[perm[n:]], y[perm[n:]]


def make_pairs(Xte, yte, n_pairs, rs):
    """Three data families plus the off-manifold control."""
    i0, i1 = np.where(yte == 0)[0], np.where(yte == 1)[0]
    return {
        'within-malignant': [(Xte[i0[a]], Xte[i0[b]]) for a, b in
                             rs.integers(0, len(i0), (n_pairs, 2)) if a != b],
        'within-benign':    [(Xte[i1[a]], Xte[i1[b]]) for a, b in
                             rs.integers(0, len(i1), (n_pairs, 2)) if a != b],
        'between-class':    [(Xte[i0[a]], Xte[i1[b]]) for a, b in
                             zip(rs.integers(0, len(i0), n_pairs),
                                 rs.integers(0, len(i1), n_pairs))],
        'off-manifold':     [(rs.standard_normal(D_IN),
                              rs.standard_normal(D_IN))
                             for _ in range(n_pairs)],
    }


# =============================================================== runs ======
def main_table(widths_list, n_seeds=20, n_pairs=250, epochs=300):
    """Theory vs initialization vs trained network, per layer and in total."""
    X, y = get_data()
    rs = np.random.default_rng(BASE_SEED)
    Xtr, ytr, Xte, yte = split(X, y, rs)
    P = make_pairs(Xte, yte, n_pairs, rs)
    print("=" * 80)
    print(f"MAIN TABLE  seeds={n_seeds}  segments/family={n_pairs}  "
          f"epochs={epochs}  sigma_b^2={SIGMA_B2}")
    print("=" * 80)
    for widths in widths_list:
        L = len(widths)
        theo = {k: np.mean([theory_segment(a, b, widths) for a, b in v], 0)
                for k, v in P.items()}
        I = {k: [] for k in P}; T = {k: [] for k in P}
        KI = {k: [] for k in P}; KT = {k: [] for k in P}
        accs = []
        for s in range(n_seeds):
            rng = np.random.default_rng([BASE_SEED, s])
            p, o = init_net(list(widths), rng=rng)
            for k, v in P.items():
                m = [measure_segment(p, o, a, b) for a, b in v]
                I[k].append(np.mean([q for q, _ in m], 0))
                KI[k].append(np.mean([r for _, r in m]))
            train(p, o, Xtr, ytr, epochs=epochs)
            accs.append(((forward(p, o, Xte) > 0).astype(float) == yte).mean())
            for k, v in P.items():
                m = [measure_segment(p, o, a, b) for a, b in v]
                T[k].append(np.mean([q for q, _ in m], 0))
                KT[k].append(np.mean([r for _, r in m]))
        print(f"\n--- widths {widths}   test accuracy "
              f"{np.mean(accs):.3f} +- {np.std(accs):.3f} ---")
        head = (f"{'family':<18}{'layer':>6}{'theory':>9}{'init':>14}"
                f"{'trained':>15}{'change':>10}{'rel':>8}")
        print(head); print('-' * len(head))
        for k in P:
            Ia, Ta = np.array(I[k]), np.array(T[k])
            for l in range(L):
                print(f"{k if l == 0 else '':<18}{l+1:>6}{theo[k][l]:>9.2f}"
                      f"{Ia[:,l].mean():>10.2f}+-{Ia[:,l].std():<3.2f}"
                      f"{Ta[:,l].mean():>11.2f}+-{Ta[:,l].std():<3.2f}"
                      f"{Ta[:,l].mean()-Ia[:,l].mean():>+10.2f}"
                      f"{100*(Ta[:,l].mean()/Ia[:,l].mean()-1):>+7.1f}%")
            it, tt = Ia.sum(1), Ta.sum(1)
            print(f"{'':<18}{'TOTAL':>6}{theo[k].sum():>9.2f}"
                  f"{it.mean():>10.2f}+-{it.std():<3.2f}"
                  f"{tt.mean():>11.2f}+-{tt.std():<3.2f}"
                  f"{tt.mean()-it.mean():>+10.2f}"
                  f"{100*(tt.mean()/it.mean()-1):>+7.1f}%")
            print(f"{'':<18}{'kinks':>6}{'':>9}"
                  f"{np.mean(KI[k]):>10.2f}+-{np.std(KI[k]):<3.2f}"
                  f"{np.mean(KT[k]):>11.2f}+-{np.std(KT[k]):<3.2f}")


def trajectory(widths=(64, 64), n_seeds=20, n_pairs=200,
               checkpoints=(0, 5, 10, 20, 40, 70, 100, 150, 200, 300, 450, 600),
               save='wisconsin_trajectory'):
    """Total switch count against training epoch, one continuous Adam run."""
    X, y = get_data()
    rs = np.random.default_rng(BASE_SEED)
    Xtr, ytr, Xte, yte = split(X, y, rs)
    P = make_pairs(Xte, yte, n_pairs, rs)
    fams = list(P)
    curves = np.zeros((n_seeds, len(fams), len(checkpoints)))
    for s in range(n_seeds):
        rng = np.random.default_rng([BASE_SEED, 1000 + s])
        p, o = init_net(list(widths), rng=rng)
        prev, st = 0, None
        for ci, ep in enumerate(checkpoints):
            if ep > prev:
                _, st = train(p, o, Xtr, ytr, epochs=ep - prev, state=st)
                prev = ep
            for fi, k in enumerate(fams):
                curves[s, fi, ci] = np.mean(
                    [measure_segment(p, o, a, b)[0].sum() for a, b in P[k]])
    np.save(save + '.npy', curves)
    np.save(save + '_checkpoints.npy', np.array(checkpoints))
    print("\n" + "=" * 80)
    print(f"TRAJECTORY  widths={widths}  seeds={n_seeds}  "
          f"segments/family={n_pairs}")
    print("=" * 80)
    print(f"{'epoch':>7}" + "".join(f"{k:>20}" for k in fams))
    for ci, ep in enumerate(checkpoints):
        print(f"{ep:>7}" + "".join(
            f"{curves[:,fi,ci].mean():>13.2f}+-{curves[:,fi,ci].std():<5.2f}"
            for fi in range(len(fams))))
    return curves, checkpoints, fams


def make_figure(curves, checkpoints, fams, stem='fig_wisconsin_trajectory',
                figsize=(3.30, 3.6), fs=8.0):
    """Training-trajectory figure.

    Sized for a half-width panel of a two-column page, so that the file is
    included at scale one and the tick labels keep their authored size.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = [('o', '-', 'black'), ('s', '--', 'black'),
              ('^', ':', 'black'), ('d', '-.', '0.55')]
    labels = ['within malignant', 'within benign', 'between class',
              'off manifold (control)']
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig, ax = plt.subplots(figsize=figsize)
        for fi in range(len(fams)):
            m = curves[:, fi, :].mean(0)
            e = curves[:, fi, :].std(0) / np.sqrt(curves.shape[0])
            mk, ls, col = styles[fi]
            ax.errorbar(checkpoints, m / m[0], yerr=e / m[0], marker=mk,
                        ls=ls, color=col, ms=3.2, lw=1.0, mfc='white',
                        mew=0.7, label=labels[fi])
        ax.axhline(1.0, color='0.7', lw=0.7)
        ax.set_xlabel('training epoch', fontsize=fs)
        # The plotted quantity is a ratio of averages, not an average of
        # ratios: the mean over network seeds at each checkpoint, divided by
        # the mean over network seeds at epoch zero.  The label says so.
        ax.set_ylabel(r'mean $S_N$ at epoch $t$ $/$ mean $S_N$ at epoch $0$',
                      fontsize=fs)
        ax.tick_params(labelsize=fs)
        ax.legend(frameon=False, fontsize=fs - 1.2, loc='center right',
                  bbox_to_anchor=(1.0, 0.30))
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(pad=0.4)
        fig.savefig(f'{stem}.{ext}', **kw)
        plt.close(fig)
    print(f"\nsaved {stem}.pdf and {stem}.png")


def robustness(widths=(64, 64), n_seeds=10, n_pairs=150,
               sigmas=(0.01, 0.1, 1.0), epoch_list=(100, 300, 1000)):
    """Relative change of the total count under different sigma_b^2 and
    training lengths."""
    X, y = get_data()
    rs = np.random.default_rng(BASE_SEED)
    Xtr, ytr, Xte, yte = split(X, y, rs)
    P = make_pairs(Xte, yte, n_pairs, rs)
    print("\n" + "=" * 80); print("ROBUSTNESS"); print("=" * 80)
    print(f"{'sigma_b^2':>10}{'epochs':>8}{'acc':>7}" +
          "".join(f"{k:>19}" for k in P))
    for sb2 in sigmas:
        for ep in epoch_list:
            ch = {k: [] for k in P}; accs = []
            for s in range(n_seeds):
                rng = np.random.default_rng([BASE_SEED, 7000 + s])
                p, o = init_net(list(widths), rng=rng, sigma_b2=sb2)
                before = {k: np.mean([measure_segment(p, o, a, b)[0].sum()
                                      for a, b in v]) for k, v in P.items()}
                train(p, o, Xtr, ytr, epochs=ep)
                accs.append(((forward(p, o, Xte) > 0).astype(float)
                             == yte).mean())
                for k, v in P.items():
                    after = np.mean([measure_segment(p, o, a, b)[0].sum()
                                     for a, b in v])
                    ch[k].append(100 * (after / before[k] - 1))
            print(f"{sb2:>10}{ep:>8}{np.mean(accs):>7.3f}" +
                  "".join(f"{np.mean(ch[k]):>+14.1f}%    " for k in P))


if __name__ == "__main__":
    main_table([(64, 64), (64, 64, 64)], n_seeds=20, n_pairs=250)
    main_table([(128, 128)], n_seeds=20, n_pairs=200)
    main_table([(128, 128, 128)], n_seeds=15, n_pairs=150)
    curves, cps, fams = trajectory()
    make_figure(curves, cps, fams)
    robustness()
