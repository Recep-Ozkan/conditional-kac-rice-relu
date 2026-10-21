"""Exact verification of the zero-bias identities of Theorem 2.1.

Companion to the article

    R. Ozkan and C. Hirsch, Affine Geometry of Gaussian ReLU Networks via
    Conditional Kac-Rice Formulas.

The zero-bias theorem is an identity at every finite width, not an asymptotic
statement, so a Monte Carlo check tests it sharply: the measured means should
agree with the predicted values up to Monte Carlo error alone.

Method
------
With no hidden biases the network restricted to the unit circle is, on every
cell of the inherited partition, a linear combination of cos(t) and sin(t).
Each preactivation is therefore of the form

    z(t) = alpha cos(t) + beta sin(t) = R cos(t - phi),

whose zeros are at phi +- pi/2.  Switch angles are obtained by solving for
those directly, cell by cell.  No angular grid is used anywhere, so the counts
carry no discretisation error.

What is counted
---------------
  * S_l  : fresh sign-changing switches of layer l, that is, zeros lying
           strictly inside an inherited cell.  Zeros at inherited breakpoints
           and cells on which the incoming activation vector vanishes
           identically are excluded, matching the definition in the article.
  * kinks: breakpoints at which the two adjacent cells give different affine
           coefficients for the scalar output.

Predictions
-----------
    p_n = 1 - 2^-n                  (at least one neuron active)
    q_n = 1 - (n+1) 2^-n            (at least two neurons active)
    E S_l   = 2 n_l prod_{r<l} q_{n_r}
    E kinks = 2 sum_l n_l prod_{r<l} q_{n_r} prod_{r>l} p_{n_r}

Seeds
-----
Each configuration uses default_rng([SEED, *widths]) with SEED = 20261020, so
the streams of different configurations are disjoint and the run is
reproducible.

Usage
-----
    python zero_bias_circle.py            # full run, about two minutes
    python zero_bias_circle.py --quick    # 2000 networks per configuration

Requirements: numpy.
"""
import argparse

import numpy as np

SEED = 20261020
TOL = 1e-12
TWO_PI = 2.0 * np.pi
CONFIGS = [(2, 2), (4, 4), (8, 8), (2, 8), (8, 2),
           (4, 4, 4), (2, 3, 4), (2, 2, 2)]


def _coeffs_upto(Ws, theta, upto):
    """Coefficients of the layer-`upto` activation vector on the cell holding
    `theta`.  Row i is (alpha, beta) with h_i(t) = alpha cos t + beta sin t."""
    h = np.eye(2)                                   # layer 0 is (cos t, sin t)
    u = np.array([np.cos(theta), np.sin(theta)])
    for r in range(upto):
        Z = Ws[r] @ h / np.sqrt(h.shape[0])
        h = Z * (Z @ u > 0)[:, None]
        # a fully inactive layer propagates zeros onward, which keeps the
        # shapes of the later layers correct
    return h


def _zeros_in(alpha, beta, lo, hi):
    """Zeros of alpha cos t + beta sin t strictly inside the arc (lo, hi)."""
    if alpha * alpha + beta * beta <= TOL:
        return []                                   # vanishes identically
    phi = np.arctan2(beta, alpha)
    out = []
    for z in (phi + np.pi / 2, phi - np.pi / 2):
        for k in (-1, 0, 1, 2):
            t = z + k * TWO_PI
            if lo + TOL < t < hi - TOL:
                out.append(t)
    return out


def run_once(widths, rng):
    """One network: fresh switches per layer, and scalar-output kinks."""
    L = len(widths)
    Ws, prev = [], 2
    for n in widths:
        Ws.append(rng.standard_normal((n, prev)))
        prev = n
    a_out = rng.standard_normal(widths[-1])

    breaks, fresh = [], []
    for l in range(L):
        edges = np.array([0.0] + sorted(breaks) + [TWO_PI])
        found = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            h = _coeffs_upto(Ws, 0.5 * (lo + hi), l)
            if not h.any():
                continue
            Z = Ws[l] @ h / np.sqrt(h.shape[0])
            for al, be in Z:
                found += _zeros_in(al, be, lo, hi)
        fresh.append(len(found))
        breaks += [t % TWO_PI for t in found]

    edges = np.array([0.0] + sorted(breaks) + [TWO_PI])
    coef = [a_out @ _coeffs_upto(Ws, 0.5 * (lo + hi), L) / np.sqrt(widths[-1])
            for lo, hi in zip(edges[:-1], edges[1:])]
    coef.append(coef[0])                            # the circle wraps around
    kinks = sum(not np.allclose(coef[i], coef[i + 1], atol=1e-9)
                for i in range(len(coef) - 1))
    return np.array(fresh, dtype=float), float(kinks)


def theory(widths):
    """Predicted switch counts per layer and predicted kink count."""
    p = lambda n: 1.0 - 2.0 ** (-n)
    q = lambda n: 1.0 - (n + 1) * 2.0 ** (-n)
    L = len(widths)
    S = np.array([2 * widths[l] * np.prod([q(widths[r]) for r in range(l)])
                  for l in range(L)])
    M = sum(widths[l] * np.prod([q(widths[r]) for r in range(l)])
            * np.prod([p(widths[r]) for r in range(l + 1, L)])
            for l in range(L))
    return S, 2.0 * M


def check(widths, trials):
    rng = np.random.default_rng([SEED, *widths])
    S = np.zeros((trials, len(widths)))
    K = np.zeros(trials)
    for i in range(trials):
        S[i], K[i] = run_once(list(widths), rng)
    Sth, Kth = theory(list(widths))

    print(f"\nwidths {tuple(widths)}, {trials} networks")
    print(f"  {'quantity':<10}{'measured':>20}{'predicted':>12}{'dev/se':>9}")
    worst = 0.0
    for l in range(len(widths)):
        se = S[:, l].std(ddof=1) / np.sqrt(trials)
        z = 0.0 if se == 0 else (S[:, l].mean() - Sth[l]) / se
        worst = max(worst, abs(z))
        print(f"  S_{l + 1:<8}{S[:, l].mean():>13.3f} +-{se:5.3f}"
              f"{Sth[l]:>12.3f}{z:>9.2f}")
    se = K.std(ddof=1) / np.sqrt(trials)
    z = (K.mean() - Kth) / se
    worst = max(worst, abs(z))
    print(f"  {'kinks':<10}{K.mean():>13.3f} +-{se:5.3f}{Kth:>12.3f}{z:>9.2f}")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    trials = 2000 if ap.parse_args().quick else 20000
    worst = max(check(w, trials) for w in CONFIGS)
    print(f"\nlargest standardised deviation over all entries: {worst:.2f}")
