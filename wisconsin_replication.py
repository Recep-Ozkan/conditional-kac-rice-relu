"""Replication study for the Breast Cancer Wisconsin experiment.

Companion to wisconsin_geometry.py, which it imports.  That file runs the
exploratory study on one fixed train/test split; this file re-runs the principal
(64, 64) experiment across many splits and many network seeds, so that the
reported change carries a measure of its variability.

Design
------
  * SPLITS stratified 70/30 train/test splits of the same dataset.
  * SEEDS independently initialised networks per split.
  * Feature standardisation is fitted on the TRAINING observations only and then
    applied to the held-out observations.  wisconsin_geometry.get_data()
    standardises over the full dataset; that is avoided here.
  * Within a split the evaluation segments are drawn once and held fixed across
    initialisation, training, and all network seeds, so a change is measured on
    the same segments before and after training.
  * The relative change is computed separately for each (split, seed) cell.
    Variability is reported across those cells.  Segments from one network are
    NOT treated as independent replicates, and the splits are NOT described as
    independent datasets: they are re-partitions of one dataset.

Seeds
-----
  * split partitions      : StratifiedShuffleSplit(random_state=SPLIT_SEED)
  * segment sampling      : default_rng([BASE_SEED, 5000 + split_index])
  * network initialisation: default_rng([BASE_SEED, split_index, seed_index])

These streams are disjoint from the ones used in wisconsin_geometry.py, which
uses [42, s], [42, 1000 + s] and [42, 7000 + s].

Usage
-----
    python wisconsin_replication.py            # full run
    python wisconsin_replication.py --quick    # 3 splits x 2 seeds

Requirements: numpy, scikit-learn.
"""
import argparse
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit

from wisconsin_geometry import (init_net, forward, train, measure_segment,
                                theory_segment, make_pairs,
                                BASE_SEED, D_IN)

SPLIT_SEED = 42
WIDTHS = (64, 64)
FAMILIES = ['within-malignant', 'within-benign', 'between-class', 'off-manifold']
LABELS = {'within-malignant': 'within malignant',
          'within-benign': 'within benign',
          'between-class': 'between class',
          'off-manifold': 'off manifold (control)'}


def replication(n_splits=10, n_seeds=5, n_pairs=200, epochs=300,
                widths=WIDTHS, test_size=0.30):
    """Return per-cell relative changes, per-cell raw counts, and accuracies."""
    data = load_breast_cancer()
    Xraw, y = data.data, data.target.astype(float)

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                 random_state=SPLIT_SEED)

    L = len(widths)
    change = {f: np.zeros((n_splits, n_seeds)) for f in FAMILIES}
    layer = {f: np.zeros((n_splits, n_seeds, L)) for f in FAMILIES}
    init_ct = {f: np.zeros((n_splits, n_seeds)) for f in FAMILIES}
    trn_ct = {f: np.zeros((n_splits, n_seeds)) for f in FAMILIES}
    theory = {f: np.zeros(n_splits) for f in FAMILIES}
    acc = np.zeros((n_splits, n_seeds))
    # Agreement between the scalar output kink count and the total hidden
    # switch count, checked on every segment before and after training.
    kink_checked = 0
    kink_equal = 0

    for si, (tr, te) in enumerate(sss.split(Xraw, y)):
        # standardisation fitted on the training observations only
        mu, sd = Xraw[tr].mean(0), Xraw[tr].std(0)
        Xtr, Xte = (Xraw[tr] - mu) / sd, (Xraw[te] - mu) / sd
        ytr, yte = y[tr], y[te]

        # evaluation segments: drawn once per split, held fixed across seeds
        rs = np.random.default_rng([BASE_SEED, 5000 + si])
        P = make_pairs(Xte, yte, n_pairs, rs)

        # closed-form initialisation prediction depends only on the segments
        for f in FAMILIES:
            theory[f][si] = np.mean([theory_segment(a, b, list(widths), D_IN).sum()
                                     for a, b in P[f]])

        for k in range(n_seeds):
            rng = np.random.default_rng([BASE_SEED, si, k])
            p, o = init_net(list(widths), D_IN, rng)

            init_layer = {}
            for f in FAMILIES:
                meas = [measure_segment(p, o, a, b) for a, b in P[f]]
                kink_checked += len(meas)
                kink_equal += sum(int(pl.sum()) == kk for pl, kk in meas)
                init_layer[f] = np.mean([pl for pl, _ in meas], axis=0)
                init_ct[f][si, k] = init_layer[f].sum()

            train(p, o, Xtr, ytr, epochs=epochs)
            acc[si, k] = ((forward(p, o, Xte) > 0).astype(float) == yte).mean()

            for f in FAMILIES:
                meas = [measure_segment(p, o, a, b) for a, b in P[f]]
                kink_checked += len(meas)
                kink_equal += sum(int(pl.sum()) == kk for pl, kk in meas)
                tl = np.mean([pl for pl, _ in meas], axis=0)
                trn_ct[f][si, k] = tl.sum()
                change[f][si, k] = 100.0 * (trn_ct[f][si, k]
                                            / init_ct[f][si, k] - 1.0)
                layer[f][si, k] = 100.0 * (tl / init_layer[f] - 1.0)

    kinks = (kink_equal, kink_checked)
    return change, init_ct, trn_ct, theory, acc, layer, kinks


def report(change, init_ct, trn_ct, theory, acc, layer, kinks,
           widths=WIDTHS):
    n_splits, n_seeds = acc.shape
    print(f"\nReplication study: {n_splits} stratified splits "
          f"x {n_seeds} network seeds = {n_splits * n_seeds} cells")
    print(f"test accuracy {acc.mean():.3f} +- {acc.std(ddof=1):.3f}\n")

    print("Initialisation: closed-form prediction against measured count")
    print(f"  {'family':<24}{'prediction':>12}{'measured':>16}")
    for f in FAMILIES:
        a = init_ct[f].ravel()
        print(f"  {LABELS[f]:<24}{theory[f].mean():>12.2f}"
              f"{a.mean():>11.2f} +-{a.std(ddof=1):>5.2f}")

    print("\nRelative change from initialisation to epoch 300, per cell")
    print(f"  {'family':<24}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}"
          f"{'consistent':>12}")
    for f in FAMILIES:
        a = change[f].ravel()
        expected = np.sign(a.mean())
        cons = int(np.sum(np.sign(a) == expected))
        print(f"  {LABELS[f]:<24}{a.mean():>8.1f}{a.std(ddof=1):>8.1f}"
              f"{a.min():>8.1f}{a.max():>8.1f}{cons:>8d}/{len(a)}")

    L = len(widths)
    if L > 1:
        print("\nLayerwise relative change, per cell")
        print(f"  {'family':<24}" + "".join(f"{'layer '+str(l+1):>16}"
                                            for l in range(L)))
        for f in FAMILIES:
            a = layer[f].reshape(-1, L)
            row = "".join(f"{a[:, l].mean():>10.1f} +-{a[:, l].std(ddof=1):>4.1f}"
                          for l in range(L))
            print(f"  {LABELS[f]:<24}" + row)

    eq, tot = kinks
    print(f"\nScalar output kinks vs total hidden switch count: "
          f"{eq}/{tot} segment measurements agree exactly "
          f"({'all' if eq == tot else 'NOT all'})")

    print("\nSplit-level means (seeds aggregated within each split)")
    hdr = "  split " + "".join(f"{LABELS[f].split(' (')[0]:>22}" for f in FAMILIES)
    print(hdr)
    for si in range(n_splits):
        row = "".join(f"{change[f][si].mean():>16.1f}" +
                      f" +-{change[f][si].std(ddof=1):>4.1f}" for f in FAMILIES)
        print(f"  {si + 1:>5} " + row)


def make_splits_figure(npz='wisconsin_replication.npz',
                       stem='fig_wisconsin_splits',
                       figsize=(3.30, 3.6), fs=8.0):
    """Split-level figure for the replicated study.

    One row per architecture.  Open circles are the split means, each averaged
    over the network seeds of that split; the bar is the mean over all cells
    and the band is one standard deviation across them.  The two rows share a
    vertical axis so that the architectures can be compared directly.

    Sized for a half-width panel of a two-column page, so that the file is
    included at scale one and the tick labels keep their authored size.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = np.load(npz)
    tags = ['64x64', '64x64x64']
    titles = [r'$(64,64)$', r'$(64,64,64)$']
    short = ['within\nmalignant', 'within\nbenign', 'between\nclass',
             'off manifold\n(control)']
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        fig, axes = plt.subplots(2, 1, figsize=figsize,
                                 sharex=True, sharey=True)
        for ax, tag, ttl in zip(axes, tags, titles):
            for i, f in enumerate(FAMILIES):
                a = d[f'change_{tag}_{f}']
                sm = a.mean(1)
                m, s = a.ravel().mean(), a.ravel().std(ddof=1)
                ax.add_patch(plt.Rectangle((i - 0.32, m - s), 0.64, 2 * s,
                                           fc='0.86', ec='none', zorder=0))
                ax.hlines(m, i - 0.32, i + 0.32, color='black', lw=1.4,
                          zorder=3)
                jitter = np.linspace(-0.17, 0.17, len(sm))
                ax.plot(i + jitter, sm, 'o', ms=3.0, mfc='white',
                        mec='black', mew=0.7, ls='none', zorder=2)
            ax.axhline(0, color='0.6', lw=0.7, zorder=1)
            ax.set_xlim(-0.6, 3.6)
            ax.set_ylabel('relative change (%)', fontsize=fs)
            ax.tick_params(labelsize=fs)
            ax.text(0.99, 0.94, ttl, transform=ax.transAxes, ha='right',
                    va='top', fontsize=fs + 0.5)
            ax.spines[['top', 'right']].set_visible(False)
        axes[1].set_xticks(range(4))
        axes[1].set_xticklabels(short, fontsize=fs)
        fig.tight_layout(pad=0.4, h_pad=0.6)
        fig.savefig(f'{stem}.{ext}', **kw)
        plt.close(fig)
    print(f"\nsaved {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ns, nk, npair = (3, 2, 100) if a.quick else (10, 5, 200)
    store = {}
    for widths in ((64, 64), (64, 64, 64)):
        tag = "x".join(str(w) for w in widths)
        print("\n" + "=" * 72)
        print(f"architecture {widths}")
        print("=" * 72)
        out = replication(n_splits=ns, n_seeds=nk, n_pairs=npair, widths=widths)
        report(*out, widths=widths)
        ch, ic, tc, th, ac, ly, kk = out
        store.update({f"change_{tag}_{f}": ch[f] for f in FAMILIES})
        store.update({f"init_{tag}_{f}": ic[f] for f in FAMILIES})
        store.update({f"trained_{tag}_{f}": tc[f] for f in FAMILIES})
        store.update({f"theory_{tag}_{f}": th[f] for f in FAMILIES})
        store.update({f"layer_{tag}_{f}": ly[f] for f in FAMILIES})
        store[f"accuracy_{tag}"] = ac
        store[f"kinks_{tag}"] = np.array(kk)
    np.savez("wisconsin_replication.npz", **store)
    print("\nsaved wisconsin_replication.npz")
    make_splits_figure()
