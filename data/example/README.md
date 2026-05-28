# `data/example/`

A small representative subset of the computational outputs the notebooks
produce, included so that readers can exercise the plotting and analysis
pipelines without having to first run the full (hour-scale, GB-sized) sweeps.

The full `Results/` and `Plots/` trees are `.gitignore`d; they can be
regenerated end-to-end from the three notebooks at the repository root. See
the top-level [`README.md`](../../README.md) for the workflow overview.

## Contents

```
Q_results/
  Q_0e65b8.npy              # on-shell self-energy Q(eta, k) -- complex128, (11, 7500)
  Q_0e65b8_meta.json        # full Params snapshot, eta grid, Picard q-grid
  Q_0e65b8_E_ext_grid.npy   # external energy grid used during the sweep
  Q_0e65b8_E_k_prime.npy    # solved on-shell energies E_k' per (eta, k)
  Q_0e65b8_iters.npy        # Picard iteration counts per (eta, k)
  # NOTE: the matching Q_0e65b8_Sigma.npy (2.1 GB) is omitted -- regenerate
  # via `Polariton Disorder.ipynb` if you need the full off-shell Sigma.

interaction_strengths/
  g_374408.npy              # effective interaction g(eta, k) -- float64, (2, 11, 240)
  g_374408_meta.json        # Params snapshot + sweep configuration

Plots/
  nongaussian_mprime0p15_manifest.json
  # Manifest produced by Visualisations.ipynb; documents which Q / g / K
  # results were combined to build the m'=0.15 non-Gaussian figure group.
  g_k0_vs_temperature_nongaussian_disorderfree_cavity_data.npz
  # Reduced data behind the g(k=0) vs temperature panel of the paper.
```

## What is *not* included here

- **Integrand meshes** (`Results/integrand_meshes/K_*.npy`, ~450 MB each).
  Regenerate via `Polariton Disorder.ipynb`.
- **Full off-shell self-energies** (`*_Sigma.npy`, ~2 GB each).
- The remaining `interaction_strengths/` and `Q_results/` hashed entries
  swept over the paper's parameter scan.

## Using these files

Each `.npy` payload is paired with a `_meta.json` sidecar containing the
exact `polaritons.parameters.Params` snapshot under which it was generated,
plus the relevant grids. The loading helpers in
[`polaritons/io.py`](../../polaritons/io.py) read both. To make the notebooks
pick up these inputs without running the heavy sweeps, copy or symlink the
subdirectories into `Results/`:

```bash
mkdir -p Results
ln -s ../data/example/Q_results              Results/Q_results
ln -s ../data/example/interaction_strengths  Results/interaction_strengths
```
