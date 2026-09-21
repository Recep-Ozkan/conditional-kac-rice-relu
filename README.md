# Affine Geometry of Gaussian ReLU Networks via Conditional Kac–Rice Formulas

[![DOI](https://zenodo.org/badge/1380199463.svg)](https://doi.org/10.5281/zenodo.22881941)

Code and data reproducing every table and figure of

> R. Özkan and C. Hirsch, *Affine Geometry of Gaussian ReLU Networks via
> Conditional Kac–Rice Formulas*.

All computations are deterministic: every script fixes its random seed, and the
seeds are recorded both here and in the source files.

## Requirements

```
python >= 3.9
numpy
scipy          # crofton_estimator.py only
matplotlib     # figures only
scikit-learn   # the Wisconsin scripts only (data and stratified splits)
```

No deep-learning framework is used. The forward pass, the backward pass and the
Adam optimiser in `wisconsin_geometry.py` are implemented directly in NumPy.

## Contents

### Scripts

| File | Produces |
|---|---|
| `verify_theorem.py` | the one-dimensional Monte Carlo tables and `fig_mc_convergence.pdf` |
| `crofton_estimator.py` | the sphere calibration table and the higher-dimensional boundary-measure table |
| `wisconsin_geometry.py` | the single-split exploratory Wisconsin study, the robustness sweep, and `fig_wisconsin_trajectory.pdf` |
| `wisconsin_replication.py` | the replicated Wisconsin study behind the main-text tables, and `fig_wisconsin_splits.pdf` |

`wisconsin_replication.py` imports the network, training and switch-counting
routines from `wisconsin_geometry.py`, so both Wisconsin studies measure the
same quantity with the same code and differ only in their protocol.

### Saved outputs

| File | Contents |
|---|---|
| `mc_L1.npy`, `mc_L2.npy`, `mc_L3.npy` | output of `verify_theorem.py`, one array per depth |
| `wisconsin_trajectory.npy` | trajectory curves, shape `(20 seeds, 4 families, 12 checkpoints)` |
| `wisconsin_trajectory_checkpoints.npy` | the 12 epoch values |
| `wisconsin_replication.npz` | per-cell relative changes, layerwise changes, raw counts, closed-form predictions and test accuracies, for both replicated architectures |

Every number in the main-text Wisconsin tables can be recomputed from
`wisconsin_replication.npz` alone, without rerunning the study.

### Figures

| File | Source |
|---|---|
| `fig_mc_convergence.pdf` | `verify_theorem.py` |
| `fig_wisconsin_splits.pdf` | `wisconsin_replication.py` (`make_splits_figure`) |
| `fig_wisconsin_trajectory.pdf` | `wisconsin_geometry.py` (`make_figure`) |
| `fig_switch_kink.pdf` | compiled from `fig_switch_kink.tex` |

These are the versions used in the article. The two Wisconsin figures are
authored at the width of a half-width panel of a two-column page, so they are
included at scale one and their labels keep their authored size.

## Seeds

| Script | Base seed | Per-run streams |
|---|---|---|
| `verify_theorem.py` | `20260720` | `default_rng([SEED, L])` per depth; `[SEED, 99]` for the unequal-width block |
| `crofton_estimator.py` | `20260816` | `default_rng([SEED, d, L])` per configuration; a single `default_rng(SEED)` stream for the sphere calibration |
| `wisconsin_geometry.py` | `42` | `default_rng([42, s])` for the exploratory tables, `[42, 1000 + s]` for the trajectory, `[42, 7000 + s]` for the robustness sweep |
| `wisconsin_replication.py` | `42` | `StratifiedShuffleSplit(random_state=42)` for the split partitions, `default_rng([42, 5000 + i])` for the segments of split `i`, `default_rng([42, i, k])` for network seed `k` of split `i` |

The four streams of the two Wisconsin scripts are disjoint, so the exploratory
study and the replicated study share no random numbers.

## Reproducing

```bash
python verify_theorem.py              # full run, a few minutes
python verify_theorem.py --quick      # reduced trial counts, for a fast check
python crofton_estimator.py           # calibration table, then the network table
python wisconsin_geometry.py          # exploratory tables, trajectory, robustness sweep
python wisconsin_replication.py       # replicated main-text tables, about 40 seconds
python wisconsin_replication.py --quick   # 3 splits x 2 seeds, for a fast check
```

Each script writes its figures and its `.npy` / `.npz` outputs into the working
directory, overwriting the copies shipped here. The shipped copies are the ones
used in the article.

## Notes on the computations

**No discretisation.** Switch locations are never found by sampling a grid. In
one dimension the first-layer switches are the roots `-b/w`; in later layers the
preactivation is affine on each cell inherited from the previous layers, so its
zeros are obtained by solving a linear equation on each cell. The same
propagation is used along chords in `crofton_estimator.py` and along data
segments in the Wisconsin scripts.

**Notation.** In the article `R_N(I)` denotes the number of *kinks* of the
scalar output, and the number of maximal affine intervals is `1 + R_N(I)`. The
quantity reported by `verify_theorem.py` is the affine interval count, that is
`1 + R_N(I)` in the article's notation. This is stated again in the docstring of
that file. The Wisconsin scripts report the hidden switch count `S_N`, for the
reasons given in Section 5 of the article.

**Normalisation in the trajectory figure.** The curves are ratios of averages,
not averages of ratios: at each checkpoint the count is averaged over the
network seeds and over the segments of the family, and that average is divided
by the corresponding average at epoch zero. The axis label states this. The two
conventions differ by well under one percentage point here, but only one of them
is plotted.

**Crofton normalisation.** The estimator in `crofton_estimator.py` is unbiased:
a chord meets a sphere of radius `a < R` in exactly two points when its offset
is below `a` and not at all otherwise, so the estimator returns the exact value
`|S^{d-1}| a^{d-1}` in expectation. `sphere_calibration()` checks this against
the closed form before the estimator is applied to any network.

**Positive bias variance.** The theorems require `beta_l > 0`. The scripts
therefore never use a zero-bias initialiser; the Wisconsin runs use
`sigma_b^2 = 0.1` in the main configuration. This is an assumption of the
theory, not a tuning choice.

**Wisconsin data.** The Breast Cancer Wisconsin (diagnostic) dataset is public
and is loaded directly through `sklearn.datasets.load_breast_cancer`. No data
file is redistributed here. In every run the evaluation segments join held-out
test points. The two scripts differ in protocol, deliberately:

* `wisconsin_geometry.py` — the exploratory study of the article's appendix.
  One fixed 70/30 split with seed 42; features standardised coordinatewise over
  the full dataset; four architectures.
* `wisconsin_replication.py` — the principal study of the main text. Ten
  stratified 70/30 splits with five network seeds each, giving 50 split–seed
  combinations per architecture; the standardisation is fitted on the training
  observations of each split only. Within a split the evaluation segments are
  drawn once and held fixed across initialisation, training and all network
  seeds. Segments from one network are not treated as independent replicates,
  and the splits are re-partitions of one dataset, not independent datasets.

**Corrections made during the study.** Two implementation errors were found and
fixed before the reported results were recorded, and both are described in the
article's reproducibility notes. First, an early trajectory implementation reset
the Adam state at each checkpoint; the optimiser state is now carried through
the run. The trajectory remains a separate run with its own seed stream and its
own segment sample on the exploratory split, so its levels are not directly
comparable with the split-averaged figures of the main text. Second, an
initialisation routine did not forward `sigma_b^2` into the covariance recursion
in the robustness sweep, which made three rows identical; this is fixed.

## Licence

See `LICENSE`.
