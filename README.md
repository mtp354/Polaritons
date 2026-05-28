# Polaritons

A Python framework for **self-consistent self-energy** and **many-body**
calculations of disordered exciton-polaritons in microcavities. The package
implements the Coherent Potential Approximation (CPA) self-energy via Picard
iteration, real-space disorder localisation, finite-temperature many-body
quantities (effective mass, chemical potential, Hopfield coefficients,
Matsubara bubble), and reproduction of the figures of the accompanying paper.

> **Citation** — If you use this code, please cite both the paper and the
> archived software release. See [`CITATION.cff`](CITATION.cff).

---

## Repository layout

```
polaritons/            # the Python package
  parameters.py        # Params dataclass + natural-unit conversions
  units.py             # SI <-> natural unit helpers
  grid.py              # 1-D momentum grids and trapezoid weights
  dispersion.py        # exciton / photon / LP / UP dispersions, Q splines
  kernel.py            # Gaussian and non-Gaussian disorder kernels
  solver.py            # Picard fixed-point iteration
  sigma.py             # CPA self-energy sweep + on-shell Q reconstruction
  many_body.py         # effective mass, chemical potential, g(k)
  real_space.py        # 2-D real-space disorder + eigenmode localisation
  io.py                # hashed result storage (.npy + .json sidecars)
  plotting.py          # matplotlib helpers (label/colour conventions)

tests/                 # pytest suite (run with `pytest`)
data/example/          # curated, small inputs that the notebooks can load
Polariton Disorder.ipynb   # CPA self-energy sweep -> Results/Q_results/
Spatial Disorder.ipynb     # real-space disorder analysis
Visualisations.ipynb       # paper-figure plotting from saved results
pyproject.toml         # build + dependency metadata (PEP 621)
requirements.txt       # convenience list for notebook users
LICENSE                # MIT
CITATION.cff           # how to cite this software
```

## Installation

The package targets Python 3.10+.

```bash
# Editable install (recommended while exploring)
pip install -e .

# With plotting / notebook extras
pip install -e .[notebook]

# Or simply install the dependencies needed for the notebooks
pip install -r requirements.txt
```

## Quick start

```python
import numpy as np
from polaritons import Params, grid, kernel, solver, sigma

p = Params.default().to_natural()
q, w = grid.uniform_grid_and_weights(N=400, q_max=20.0)
K    = kernel.make_kernel_gaussian(p)(q, q)

E_ext = np.linspace(-2.0, 2.0, 201)
eta   = np.array([0.0, 0.5, 1.0])
Sigma, iters = sigma.sweep_sigma(p, K, q, w, E_ext, eta, verbose=True)
```

See [`Polariton Disorder.ipynb`](Polariton%20Disorder.ipynb) for a full
parameter sweep and [`Visualisations.ipynb`](Visualisations.ipynb) for the
plotting pipelines that produce the figures of the paper.

## Reproducing the paper figures

The three notebooks form the reproduction pipeline:

1. **`Polariton Disorder.ipynb`** — runs the Picard CPA self-energy sweep and
   writes hashed `.npy` arrays (with `.json` sidecars carrying the full
   `Params` snapshot) to `Results/Q_results/` and `Results/integrand_meshes/`.
2. **`Spatial Disorder.ipynb`** — performs the 2-D real-space disorder and
   localisation analysis.
3. **`Visualisations.ipynb`** — loads saved results, builds plot manifests
   under `Plots/`, and exports the figures used in the paper.

`Results/` and `Plots/` are `.gitignored` because the full integrand meshes
(~450 MB each) and saved self-energy arrays (~2 GB each) are too large to
distribute via Git. A representative subset sufficient to exercise the
plotting code is included under [`data/example/`](data/example/); see its
[README](data/example/README.md) for usage. Anything else can be regenerated
from the notebooks.

## Tests

The suite is `pytest`-driven; the default invocation skips tests marked
`slow`:

```bash
pytest                 # fast suite
pytest -m slow         # slow tests only
pytest -m ""           # everything
```

## License

Released under the [MIT License](LICENSE).
