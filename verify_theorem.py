"""Monte Carlo verification for the one-dimensional numerics of
"Affine Geometry of Gaussian ReLU Networks via Conditional Kac-Rice Formulas"
(R. Ozkan and C. Hirsch).

Model: fully connected ReLU network, one-dimensional input, scalar output,
all weight and bias variances equal to 1, input interval I = [-1, 1].

NOTE ON NOTATION.  In the manuscript R_N(I) is the number of KINKS of the
scalar output, and the number of maximal affine intervals is 1 + R_N(I).
What this script reports is the AFFINE INTERVAL COUNT, i.e. 1 + R_N(I) in
the manuscript's notation.  The printed numbers match the "MC" columns of
the tables.

Theory (Theorem 2.1 and its constant-variance corollary):
    int_I rho_l dt = (2/pi) * arcsin(2**(-l/2))   [= 1/2, 1/3, ~0.2301 for l=1,2,3]
    1 + E R_N(I) = 1 + sum_l n_l * int_I rho_l + o(N)
    For L = 1 the formula is exact at every finite width.

Region counts are computed EXACTLY by piecewise-affine propagation:
first-layer switches are the roots -b/w; higher-layer switches are found by
solving the affine restriction of each preactivation on each inherited cell;
the reported count is one plus the number of slope changes of f_N across the
ordered switch locations.  No discretization of I is involved.

Usage:
    python verify_theorem.py          # full run (a few minutes)
    python verify_theorem.py --quick  # reduced trials, for a fast check

Requirements: numpy, matplotlib.  Reproducible via the fixed seed below.
"""
import argparse
import numpy as np

SEED = 20260720
A_END, B_END = -1.0, 1.0


def rho_integral(l):
    """int_{-1}^{1} rho_l(t) dt for unit variances (Corollary 7.1)."""
    return 2 / np.pi * np.arcsin(2.0 ** (-l / 2))


def simulate_once(widths, RNG):
    """One network sample; returns (R_N(I), S_N(I)) exactly.

    widths = [n_1, ..., n_L]
    """
    L = len(widths)
    params = []
    n_prev = None
    for l, n in enumerate(widths):
        b = RNG.standard_normal(n)
        W = RNG.standard_normal((n, n_prev)) if l > 0 else RNG.standard_normal(n)
        params.append((b, W))
        n_prev = n
    a_out = RNG.standard_normal(widths[-1])
    b_out = RNG.standard_normal()

    def hidden(t, upto):
        """Hidden activations h_{upto+1, .}(t) for a vector of inputs t."""
        h = None
        for l in range(upto + 1):
            b, W = params[l]
            if l == 0:
                z = b[None, :] + W[None, :] * t[:, None]
            else:
                z = b[None, :] + h @ W.T / np.sqrt(h.shape[1])
            h = np.maximum(0.0, z)
        return h

    # layer-1 switches: roots of b + w t
    b1, w1 = params[0]
    s = -b1 / w1
    switches = list(s[(s > A_END) & (s < B_END)])

    # layers 2..L: solve the affine restriction on each inherited cell
    for l in range(1, L):
        grid = np.sort(np.array([A_END] + switches + [B_END]))
        H_prev = hidden(grid, l - 1)
        b, W = params[l]
        Z = b[None, :] + H_prev @ W.T / np.sqrt(H_prev.shape[1])
        sgn = np.sign(Z)
        cross = sgn[:-1, :] * sgn[1:, :] < 0
        k, j = np.nonzero(cross)
        z0, z1 = Z[k, j], Z[k + 1, j]
        t0, t1 = grid[k], grid[k + 1]
        switches += list(t0 - z0 * (t1 - t0) / (z1 - z0))

    # scalar output slopes across ordered switch locations
    pts = np.sort(np.array([A_END] + switches + [B_END]))
    hL = hidden(pts, L - 1)
    vals = b_out + hL @ a_out / np.sqrt(hL.shape[1])
    slopes = np.diff(vals) / np.diff(pts)
    jumps = ~np.isclose(slopes[:-1], slopes[1:], rtol=1e-8, atol=1e-12)
    return 1 + int(jumps.sum()), len(switches)


def run_depth(L, widths, trials):
    RNG = np.random.default_rng([SEED, L])   # per-depth reproducible stream
    coeff = sum(rho_integral(l) for l in range(1, L + 1))
    limit = coeff / L
    print(f"\n=== L = {L}, equal widths, trials = {trials} ===")
    print(f"{'n':>5} {'N':>5} {'1+E R_N (MC)':>12} {'+-':>6} {'theory':>10} "
          f"{'gap':>7} {'R/N':>8} {'limit':>7}")
    rows = []
    for n in widths:
        Rs = np.array([simulate_once([n] * L, RNG)[0] for _ in range(trials)])
        N = L * n
        mR, se = Rs.mean(), Rs.std(ddof=1) / np.sqrt(trials)
        pred = 1 + coeff * n
        print(f"{n:>5} {N:>5} {mR:>12.3f} {se:>6.3f} {pred:>10.3f} "
              f"{mR - pred:>7.3f} {mR / N:>8.4f} {limit:>7.4f}")
        rows.append((n, N, mR, se, pred, mR / N, limit))
    return np.array(rows)


def make_figure(r1, r2, r3, fname="fig_mc_convergence.pdf"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for y, ls in [(0.5, "--"), (5 / 12, ":"), (r3[0, 6], (0, (3, 1, 1, 1)))]:
        ax.axhline(y, color="0.6", ls=ls, lw=1)
    ax.errorbar(r1[:, 1], r1[:, 5], yerr=r1[:, 3] / r1[:, 1], marker="o",
                ms=4.5, color="black", lw=1.2, label=r"$L=1$ (limit $1/2$)")
    ax.errorbar(r2[:, 1], r2[:, 5], yerr=r2[:, 3] / r2[:, 1], marker="s",
                ms=4.5, mfc="white", color="black", lw=1.2, ls="--",
                label=r"$L=2$ (limit $5/12$)")
    ax.errorbar(r3[:, 1], r3[:, 5], yerr=r3[:, 3] / r3[:, 1], marker="^",
                ms=5, mfc="0.6", color="black", lw=1.2, ls=":",
                label=r"$L=3$ (limit $\approx 0.3545$)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"total width $N$")
    ax.set_ylabel(r"$(1+\mathbb{E}\,R_N(I))\,/\,N$  (Monte Carlo)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname)
    print(f"\nsaved {fname}")


def run_unequal(trials=500):
    """Unequal-width check (last block of Table 1): two allocations of the
    same total width N = 192 at depth L = 2, plus an L = 3 sanity check."""
    RNG = np.random.default_rng([SEED, 99])
    print(f"\n=== unequal widths, trials = {trials} ===")
    for widths in [(128, 64), (64, 128), (32, 64, 128)]:
        pred = 1 + sum(n * rho_integral(l + 1) for l, n in enumerate(widths))
        Rs = np.array([simulate_once(list(widths), RNG)[0]
                       for _ in range(trials)])
        se = Rs.std(ddof=1) / np.sqrt(trials)
        print(f"widths {widths}: MC {Rs.mean():.3f} +- {se:.3f}   "
              f"pred {pred:.3f}   gap {Rs.mean() - pred:+.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="reduced trial counts")
    p.add_argument("--depth", type=int, default=0,
                   help="run a single depth (1, 2, or 3); 0 = all")
    args = p.parse_args()
    t1, t2, t3 = (200, 100, 60) if args.quick else (2000, 1000, 600)
    if args.depth in (0, 1):
        np.save("mc_L1.npy", run_depth(1, [8, 16, 32, 64, 128, 256], t1))
    if args.depth in (0, 2):
        np.save("mc_L2.npy", run_depth(2, [8, 16, 32, 64, 128, 256], t2))
    if args.depth in (0, 3):
        np.save("mc_L3.npy", run_depth(3, [8, 16, 32, 64, 128], t3))
    if args.depth == 0 or all(__import__('os').path.exists(f"mc_L{i}.npy") for i in (1, 2, 3)):
        make_figure(np.load("mc_L1.npy"), np.load("mc_L2.npy"), np.load("mc_L3.npy"))
    if args.depth == 0:
        run_unequal(60 if args.quick else 500)
