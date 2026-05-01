"""
Real-space localisation physics: 2-D Gaussian-correlated disorder potential,
LP Hamiltonian, eigenmode solver, IPR, and effective interaction strength.

This module is intentionally self-contained (separate imports) so it can be
used independently of the k-space polariton calculation.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# Disorder potential
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

	The potential is zero-mean with standard deviation `sigma`.
	The spatial correlation function is  <V(r)V(r')> ∝ exp(-|r-r'|^2 / xi^2).

	Parameters
	----------
	N     : number of grid points per side
	L     : system size (same units as xi)
	sigma : disorder amplitude (energy units)
	xi    : correlation length (same units as L)
	seed  : random seed for reproducibility

	Returns
	-------
	V : (N, N) float array
	"""
	rng  = np.random.default_rng(seed)
	dx   = L / N
	noise_k = np.fft.fftn(rng.normal(size=(N, N)))

	kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
	KX, KY = np.meshgrid(kx, kx, indexing="ij")
	filt = np.exp(-0.25 * xi**2 * (KX**2 + KY**2))

	V  = np.fft.ifftn(noise_k * filt).real
	V -= V.mean()
	V *= sigma / V.std()
	return V


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------

def laplacian_2d(N: int, dx: float):
	"""Sparse 2-D finite-difference Laplacian on an N×N grid (CSR format)."""
	e = np.ones(N)
	T = diags([e, -2*e, e], [-1, 0, 1], shape=(N, N), format="csr") / dx**2
	I = eye(N, format="csr")
	return kron(I, T) + kron(T, I)


def lp_hamiltonian(N: int, L: float, M_lp: float, V: np.ndarray, hbar: float):
	"""
	Sparse lower-polariton Hamiltonian H = -(hbar^2 / 2M) ∇^2 + V(x,y).

	Parameters
	----------
	N    : grid points per side
	L    : system size (m)
	M_lp : LP effective mass (eV·s²/m²)
	V    : (N, N) disorder potential array (eV)
	hbar : reduced Planck constant (eV·s)

	Returns
	-------
	H : sparse (N^2, N^2) real matrix
	"""
	dx      = L / N
	kinetic = -(hbar**2 / (2.0 * M_lp)) * laplacian_2d(N, dx)
	potential = diags(V.ravel(), 0, format="csr")
	return kinetic + potential


# ---------------------------------------------------------------------------
# Eigenmode solver
# ---------------------------------------------------------------------------

def solve_low_modes(
	H       : "scipy.sparse.spmatrix",
	n_modes : int = 6,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Find the `n_modes` lowest eigenmodes of sparse Hermitian matrix H.

	Returns
	-------
	eigenvalues  : 1-D array, sorted ascending
	eigenvectors : (N^2, n_modes) array, columns are eigenvectors
	"""
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
	"""
	Inverse participation ratio  IPR = ∫|ψ|⁴ d²r.

	A delocalised state ∝ 1/√A has IPR = 1/A;
	a perfectly localised state has IPR = 1.
	"""
	return float(np.sum(np.abs(psi)**4) * dx**2)


# ---------------------------------------------------------------------------
# Composite pipeline
# ---------------------------------------------------------------------------

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
	"""
	Full pipeline for a single disorder realisation:
		disorder → Hamiltonian → ground mode → g_eff = g_lp × IPR

	Parameters
	----------
	N, L, M_lp, sigma, xi, hbar : as in `lp_hamiltonian` / `gaussian_correlated_disorder`
	g_lp : bare LP interaction strength (same units as g_eff output)
	seed : random seed

	Returns
	-------
	g_eff : effective interaction strength for the ground mode
	"""
	dx    = L / N
	V     = gaussian_correlated_disorder(N, L, sigma, xi, seed=seed)
	H     = lp_hamiltonian(N, L, M_lp, V, hbar)
	_, psi = solve_low_modes(H, n_modes=1)
	psi0  = normalize_mode(psi[:, 0].reshape(N, N), dx)
	return g_lp * ipr(psi0, dx)
