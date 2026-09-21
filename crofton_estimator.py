"""Crofton-based Monte Carlo verification of Theorem (higher-dimensional
boundaries) for fully connected Gaussian ReLU networks in input dimension
d >= 2.

Theory
------
    E S_{N,d}(D) = sum_l n_l * int_D rho_{l,d}(x) dx + o(N)

    s_l(x)   = A_l + B_l |x|^2
    rho_{l,d}(x) = sqrt(B_l) / sqrt(2 pi s_l(x))
                   * E[ sum_{r<d} G_r^2 + (A_l/s_l(x)) G_d^2 ]^{1/2}
    A_1 = beta_1, B_1 = gamma_1,
    A_l = beta_l + gamma_l A_{l-1}/2,   B_l = gamma_l B_{l-1}/2.

Estimator
---------
Crofton / Cauchy formula.  For a (d-1)-rectifiable set S,

    H^{d-1}(S) = 1/(2 v_{d-1}) * int_{S^{d-1}} int_{v^perp} #(S cap (y+Rv)) dy sigma(dv),

with v_{d-1} the volume of the unit ball in R^{d-1}.  Taking D = ball(0,R) and
sampling v uniformly on S^{d-1} and y uniformly on the (d-1)-ball of radius R
in v^perp, the projection of D, this collapses to

    H^{d-1}(S) ~= ( |S^{d-1}| R^{d-1} / 2 ) * E[ # zeros along a random chord ].

Zeros along a chord are found EXACTLY by piecewise-affine propagation, exactly
as in verify_theorem.py: along x(t) = y + t v every preactivation is affine on
each cell inherited from the previous layers, so no discretization is used.
"""
import numpy as np
from scipy import integrate
from scipy.special import gamma as Gammafn

SEED = 20260816


# ---------------------------------------------------------------- theory ---
def AB(L, beta, gamma):
    """Covariance recursion; returns lists A[1..L], B[1..L]."""
    A, B = [beta[0]], [gamma[0]]
    for l in range(1, L):
        A.append(beta[l] + gamma[l] * A[-1] / 2.0)
        B.append(gamma[l] * B[-1] / 2.0)
    return A, B


def E_sqrt_quadratic(d, c):
    """E[ sum_{r=1}^{d-1} G_r^2 + c G_d^2 ]^{1/2} via the Laplace identity
    sqrt(u) = 1/(2 sqrt(pi)) int_0^inf (1 - e^{-u lam}) lam^{-3/2} dlam."""
    def integrand(lam):
        mgf = (1 + 2 * lam) ** (-(d - 1) / 2.0) * (1 + 2 * lam * c) ** (-0.5)
        return (1.0 - mgf) * lam ** (-1.5)
    val, _ = integrate.quad(integrand, 0, np.inf, limit=200,
                            epsabs=1e-12, epsrel=1e-12)
    return val / (2 * np.sqrt(np.pi))


def rho(l_A, l_B, r, d):
    """rho_{l,d} at radius |x| = r."""
    s = l_A + l_B * r ** 2
    return np.sqrt(l_B) / np.sqrt(2 * np.pi * s) * E_sqrt_quadratic(d, l_A / s)


def theory_total(widths, beta, gamma, d, R):
    """sum_l n_l int_{|x|<R} rho_{l,d}(x) dx."""
    L = len(widths)
    A, B = AB(L, beta, gamma)
    surf = 2 * np.pi ** (d / 2.0) / Gammafn(d / 2.0)      # |S^{d-1}|
    tot = 0.0
    for l in range(L):
        f = lambda r: rho(A[l], B[l], r, d) * surf * r ** (d - 1)
        val, _ = integrate.quad(f, 0, R, epsabs=1e-11, epsrel=1e-11)
        tot += widths[l] * val
    return tot


# ------------------------------------------------------------- simulation ---
def sample_network(widths, d, beta, gamma, rng):
    params, n_prev = [], d
    for l, n in enumerate(widths):
        b = np.sqrt(beta[l]) * rng.standard_normal(n)
        W = np.sqrt(gamma[l]) * rng.standard_normal((n, n_prev))
        if l > 0:
            W = W / np.sqrt(n_prev)          # 1/sqrt(n_{l-1}) scaling
        params.append((b, W))
        n_prev = n
    return params


def random_chord(d, R, rng):
    """Uniform direction v on S^{d-1}; y uniform on the (d-1)-ball of radius R
    inside v^perp.  Returns (y, v, T) with the chord {y + t v : |t| <= T}."""
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    Q, _ = np.linalg.qr(np.column_stack([v, rng.standard_normal((d, d - 1))]))
    basis = Q[:, 1:]                          # orthonormal basis of v^perp
    u = rng.standard_normal(d - 1)
    u /= np.linalg.norm(u)
    rad = R * rng.random() ** (1.0 / (d - 1))  # uniform in (d-1)-ball
    y = basis @ (rad * u)
    return y, v, np.sqrt(max(R ** 2 - rad ** 2, 0.0))


def count_zeros_on_chord(params, y, v, T):
    """Exact number of hidden-neuron zeros on {y+tv : |t|<T}, all layers,
    counted with multiplicity over neurons."""
    def hidden(ts, upto):
        X = y[None, :] + ts[:, None] * v[None, :]
        h = None
        for l in range(upto + 1):
            b, W = params[l]
            z = b[None, :] + (X if l == 0 else h) @ W.T
            h = np.maximum(0.0, z)
        return h

    b1, W1 = params[0]
    a0, a1 = b1 + W1 @ y, W1 @ v              # z = a0 + t a1
    with np.errstate(divide='ignore', invalid='ignore'):
        roots = -a0 / a1
    sw = list(roots[np.isfinite(roots) & (np.abs(roots) < T)])
    total = len(sw)

    for l in range(1, len(params)):
        grid = np.sort(np.array([-T] + sw + [T]))
        H = hidden(grid, l - 1)
        b, W = params[l]
        Z = b[None, :] + H @ W.T
        sgn = np.sign(Z)
        cross = sgn[:-1, :] * sgn[1:, :] < 0
        k, j = np.nonzero(cross)
        z0, z1 = Z[k, j], Z[k + 1, j]
        t0, t1 = grid[k], grid[k + 1]
        new = list(t0 - z0 * (t1 - t0) / (z1 - z0))
        sw += new
        total += len(new)
    return total


def crofton_estimate(widths, d, beta, gamma, R, n_nets, n_chords, rng):
    surf = 2 * np.pi ** (d / 2.0) / Gammafn(d / 2.0)
    const = surf * R ** (d - 1) / 2.0
    per_net = []
    for _ in range(n_nets):
        params = sample_network(widths, d, beta, gamma, rng)
        counts = [count_zeros_on_chord(params, *random_chord(d, R, rng))
                  for _ in range(n_chords)]
        per_net.append(const * np.mean(counts))
    per_net = np.array(per_net)
    return per_net.mean(), per_net.std(ddof=1) / np.sqrt(n_nets)


def sphere_calibration(R=1.0, n_chords=40000, radii=(0.4, 0.7), dims=(2, 3, 5)):
    """Calibrate the estimator on spheres, whose measure is known exactly.

    For a sphere of radius a < R centred at the origin, a chord {y + t v}
    with y perpendicular to v meets it in exactly two points when |y| < a and
    not at all otherwise, so the estimator is unbiased with the exact value
    H^{d-1} = |S^{d-1}| a^{d-1}.  This checks the Crofton normalisation and
    the chord sampling independently of any network.
    """
    rng = np.random.default_rng(SEED)
    print(f"calibration on spheres inside the unit ball, "
          f"{n_chords} chords per row\n")
    print(f"{'d':>2} {'a':>5} {'exact':>10} {'estimate':>12} {'+-':>7} {'ratio':>7}")
    for d in dims:
        surf = 2 * np.pi ** (d / 2.0) / Gammafn(d / 2.0)
        const = surf * R ** (d - 1) / 2.0
        for a in radii:
            counts = np.empty(n_chords)
            for i in range(n_chords):
                y, v, T = random_chord(d, R, rng)
                counts[i] = 2.0 if float(y @ y) < a * a else 0.0
            est = const * counts.mean()
            se = const * counts.std(ddof=1) / np.sqrt(n_chords)
            exact = surf * a ** (d - 1)
            print(f"{d:>2} {a:>5} {exact:>10.4f} {est:>12.4f}"
                  f" {se:>7.4f} {est / exact:>7.4f}")
    print()


# ------------------------------------------------------------------ main ---
if __name__ == "__main__":
    R = 1.0
    sphere_calibration(R=R)
    print(f"D = unit ball in R^d,  beta_l = gamma_l = 1,  R = {R}\n")
    print(f"{'d':>2} {'L':>2} {'widths':>14} {'theory':>10} {'Crofton MC':>12}"
          f" {'+-':>7} {'ratio':>7}")
    for d in (2, 3, 5):
        for widths in ([64], [32, 32], [24, 24, 24]):
            L = len(widths)
            beta = [1.0] * L
            gamma = [1.0] * L
            rng = np.random.default_rng([SEED, d, L])
            th = theory_total(widths, beta, gamma, d, R)
            mc, se = crofton_estimate(widths, d, beta, gamma, R,
                                      n_nets=40, n_chords=300, rng=rng)
            print(f"{d:>2} {L:>2} {str(widths):>14} {th:>10.4f} {mc:>12.4f}"
                  f" {se:>7.4f} {mc/th:>7.4f}")
