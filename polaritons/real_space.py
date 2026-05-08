"""
Real-space disorder physics.

Two related calculations live in this module:

1. **Localisation pipeline** — generate a 2-D Gaussian-correlated disorder
   potential, build the lower-polariton (or exciton) Hamiltonian on a finite
   grid, find the lowest eigenmodes, and compute IPR / effective interaction
   strength of the localised ground mode.

2. **Disorder-averaged self-energy** — extract a real-space analogue of the
   CPA self-energy ``Q(k, E)`` from the momentum-diagonal disorder-averaged
   Green function

       G_avg(k, E) = < <k| (E + i ε - H)^(-1) |k> >_disorder
       Q(k, E)     = 1 / G_avg(k, E) - (E + i ε - ε_k).

Both pieces share the same Gaussian-correlated disorder generator and the
same finite-difference Laplacian (with optional periodic boundaries).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh, splu


# ---------------------------------------------------------------------------
# Disorder potential (shared)
# ---------------------------------------------------------------------------

def gaussian_correlated_disorder(
	N    : int,
	L    : float,
	sigma: float,
	xi   : float,
	seed : int | None = None,
) -> np.ndarray:
	"""
	Generate a 2-D Gaussian-correlated disorder potential V(x,y) on an N×N grid.

	The potential is zero-mean with standard deviation ``sigma`` (when
	``sigma > 0``).  The spatial correlation function is

	    <V(r)V(r')> ∝ exp(-|r-r'|^2 / (2 xi^2)),

	consistent with the FT convention used by ``polaritons.kernel``
	(filter ``exp(-0.5 * xi^2 * p^2)``).

	Parameters
	----------
	N     : number of grid points per side
	L     : system size (same units as xi)
	sigma : disorder amplitude (energy units)
	xi    : correlation length (same units as L)
	seed  : random seed for reproducibility
	"""
	rng = np.random.default_rng(seed)
	dx  = L / N

	noise_k = np.fft.fftn(rng.normal(size=(N, N)))

	kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
	KX, KY = np.meshgrid(kx, kx, indexing="ij")
	filt = np.exp(-0.25 * xi**2 * (KX**2 + KY**2))

	V = np.fft.ifftn(noise_k * filt).real
	V -= V.mean()

	std = V.std()
	if std > 0:
		V *= sigma / std
	return V


# ---------------------------------------------------------------------------
# Hamiltonian construction (shared)
# ---------------------------------------------------------------------------

def laplacian_2d(N: int, dx: float, periodic: bool = False):
	"""
	Sparse 2-D 5-point finite-difference Laplacian on an N×N grid (CSR).

	Parameters
	----------
	N        : grid points per side
	dx       : grid spacing
	periodic : if True, wrap the 1-D stencils so the operator is periodic
	           (used by the Q-extraction pipeline so plane waves are exact
	           eigenvectors of the kinetic operator).  Default False
	           reproduces the open-boundary behaviour used by the
	           localisation pipeline.
	"""
	e = np.ones(N)
	if periodic:
		T = diags([e, -2 * e, e], [-1, 0, 1], shape=(N, N), format="lil")
		T[0, -1] = 1.0
		T[-1, 0] = 1.0
		T = (T / dx**2).tocsr()
	else:
		T = diags([e, -2 * e, e], [-1, 0, 1], shape=(N, N), format="csr") / dx**2
	I = eye(N, format="csr")
	return kron(I, T, format="csr") + kron(T, I, format="csr")


def lp_hamiltonian(
	N        : int,
	L        : float,
	M_lp     : float,
	V        : np.ndarray,
	hbar     : float,
	periodic : bool = False,
):
	"""
	Sparse Hamiltonian H = -(ħ² / 2 M) ∇² + V(x,y) on an N×N grid.

	Used as the lower-polariton Hamiltonian (default open boundaries) or, with
	``periodic=True``, as the bare exciton Hamiltonian for the Q-extraction
	pipeline.
	"""
	dx        = L / N
	kinetic   = -(hbar**2 / (2.0 * M_lp)) * laplacian_2d(N, dx, periodic=periodic)
	potential = diags(V.ravel(), 0, format="csr")
	return kinetic + potential


# ---------------------------------------------------------------------------
# Eigenmode solver (localisation pipeline)
# ---------------------------------------------------------------------------

def solve_low_modes(H, n_modes: int = 6) -> tuple[np.ndarray, np.ndarray]:
	"""Find the ``n_modes`` lowest eigenmodes of sparse Hermitian ``H``."""
	vals, vecs = eigsh(H, k=n_modes, which="SA")
	order = np.argsort(vals)
	return vals[order], vecs[:, order]


# ---------------------------------------------------------------------------
# Mode analysis
# ---------------------------------------------------------------------------

def normalize_mode(psi: np.ndarray, dx: float) -> np.ndarray:
	"""L²-normalise a 2-D wavefunction on a uniform grid with spacing dx."""
	norm = np.sqrt(np.sum(np.abs(psi)**2) * dx**2)
	return psi / norm


def ipr(psi: np.ndarray, dx: float) -> float:
	"""Inverse participation ratio  IPR = ∫|ψ|⁴ d²r."""
	return float(np.sum(np.abs(psi)**4) * dx**2)


def mode_area(psi: np.ndarray, dx: float) -> float:
	"""Effective mode area  A_eff = 1 / IPR."""
	return 1.0 / ipr(psi, dx)


def g_eff_lowest_mode(
	N    : int,
	L    : float,
	M_lp : float,
	sigma: float,
	xi   : float,
	hbar : float,
	g_lp : float = 1.0,
	seed : int | None = None,
) -> float:
	"""Single-realisation pipeline → ``g_eff = g_lp × IPR`` of the ground mode."""
	dx     = L / N
	V      = gaussian_correlated_disorder(N, L, sigma, xi, seed=seed)
	H      = lp_hamiltonian(N, L, M_lp, V, hbar)
	_, psi = solve_low_modes(H, n_modes=1)
	psi0   = normalize_mode(psi[:, 0].reshape(N, N), dx)
	return g_lp * ipr(psi0, dx)


# ===========================================================================
# Disorder-averaged self-energy Q(k, E) via real-space Green function
# ===========================================================================

@dataclass(frozen=True)
class RealSpaceConfig:
	"""Bundle of parameters for the Q-extraction pipeline."""
	N      : int
	L      : float
	M      : float
	hbar   : float = 6.582119569e-16
	epsilon: float = 1e-9

	@property
	def dx(self) -> float:
		return self.L / self.N


def exciton_hamiltonian(config: RealSpaceConfig, V: np.ndarray):
	"""Periodic-boundary exciton Hamiltonian -ħ²/2M ∇² + V on the box."""
	return lp_hamiltonian(
		config.N, config.L, config.M, V, config.hbar, periodic=True,
	)


def plane_wave(config: RealSpaceConfig, nx: int, ny: int) -> np.ndarray:
	"""Box-normalised plane wave with integer momentum indices ``(nx, ny)``."""
	j = np.arange(config.N)
	X, Y = np.meshgrid(j, j, indexing="ij")
	phi = np.exp(2j * np.pi * (nx * X + ny * Y) / config.N) / config.L
	return phi.ravel()


def lattice_kinetic_energy(config: RealSpaceConfig, nx: int, ny: int) -> float:
	"""Eigenvalue of the periodic FD kinetic operator at momentum ``(nx, ny)``."""
	dx = config.dx
	lam = (
		4
		- 2 * np.cos(2 * np.pi * nx / config.N)
		- 2 * np.cos(2 * np.pi * ny / config.N)
	) / dx**2
	return config.hbar**2 * lam / (2 * config.M)


def continuum_k(config: RealSpaceConfig, nx: int, ny: int) -> float:
	"""Continuum momentum magnitude |k| = (2π/L)·√(nx²+ny²)."""
	return (2 * np.pi / config.L) * np.sqrt(nx**2 + ny**2)


def integer_shells(
	n_shells    : int,
	nmax        : int | None = None,
	include_zero: bool = True,
) -> list[list[tuple[int, int]]]:
	"""
	Group integer momentum indices by ``nx² + ny²`` and return the first
	``n_shells`` groups in ascending order of |n|².
	"""
	if nmax is None:
		nmax = max(3, int(np.ceil(np.sqrt(2 * n_shells))) + 4)

	groups: dict[int, list[tuple[int, int]]] = {}
	for nx in range(-nmax, nmax + 1):
		for ny in range(-nmax, nmax + 1):
			n2 = nx * nx + ny * ny
			if n2 == 0 and not include_zero:
				continue
			groups.setdefault(n2, []).append((nx, ny))

	return [groups[n2] for n2 in sorted(groups)[:n_shells]]


def green_for_realization(
	config        : RealSpaceConfig,
	V             : np.ndarray,
	E             : float,
	shell_vectors : Iterable[tuple[int, int]],
) -> np.ndarray:
	"""
	Diagonal momentum-space matrix elements of (E + iε - H)^(-1) for one
	disorder realisation, evaluated at the requested integer momenta.
	"""
	H  = exciton_hamiltonian(config, V)
	z  = E + 1j * config.epsilon
	lu = splu(csc_matrix(z * eye(config.N**2, format="csr") - H))

	out = []
	for nx, ny in shell_vectors:
		phi = plane_wave(config, nx, ny)
		x   = lu.solve(phi)
		out.append(config.dx**2 * np.vdot(phi, x))
	return np.asarray(out)


def averaged_green_and_Q(
	config         : RealSpaceConfig,
	E              : float,
	sigma          : float,
	xi             : float,
	shells         : list[list[tuple[int, int]]],
	n_realizations : int = 20,
	seed0          : int = 0,
):
	"""
	Average G(k, E) over disorder realisations and over each ``|n|²`` shell,
	then extract Q(k, E) = 1/G_avg - (E + iε - ε_k).
	"""
	G_sum   = np.zeros(len(shells), dtype=np.complex128)
	G_count = np.zeros(len(shells), dtype=int)

	for r in range(n_realizations):
		V = gaussian_correlated_disorder(
			config.N, config.L, sigma=sigma, xi=xi, seed=seed0 + r,
		)
		all_vectors = [v for shell in shells for v in shell]
		G_all = green_for_realization(config, V, E, all_vectors)

		start = 0
		for s, shell in enumerate(shells):
			stop = start + len(shell)
			G_sum[s]   += G_all[start:stop].sum()
			G_count[s] += len(shell)
			start = stop

	# G_count already accumulates len(shell) once per realisation, so it
	# is the total number of samples contributing to G_sum for each shell.
	G_avg = G_sum / G_count

	k_vals = np.array([
		np.mean([continuum_k(config, nx, ny) for nx, ny in shell])
		for shell in shells
	])
	eps_vals = np.array([
		np.mean([lattice_kinetic_energy(config, nx, ny) for nx, ny in shell])
		for shell in shells
	])

	Q = 1.0 / G_avg - (E + 1j * config.epsilon - eps_vals)
	return k_vals, G_avg, Q


def sweep_disorder_self_energy(
	config         : RealSpaceConfig,
	E              : float,
	sigmas         : Iterable[float],
	xi             : float,
	shells         : list[list[tuple[int, int]]],
	n_realizations : int = 20,
	seed0          : int = 0,
):
	"""Run ``averaged_green_and_Q`` for each σ in ``sigmas``."""
	sigmas    = np.asarray(list(sigmas), dtype=float)
	k_ref     = None
	Q_results = np.zeros((len(sigmas), len(shells)), dtype=np.complex128)
	G_results = np.zeros_like(Q_results)

	for i, sigma in enumerate(sigmas):
		k_vals, G_avg, Q = averaged_green_and_Q(
			config         = config,
			E              = E,
			sigma          = sigma,
			xi             = xi,
			shells         = shells,
			n_realizations = n_realizations,
			seed0          = seed0 + 10_000 * i,
		)
		if k_ref is None:
			k_ref = k_vals
		Q_results[i] = Q
		G_results[i] = G_avg

	return k_ref, sigmas, G_results, Q_results
