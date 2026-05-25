"""
Unit tests for polaritons.sigma (sweep_sigma, find_E_k_prime, assemble_Q).
"""
import numpy as np
import pytest

from polaritons.parameters import Params
from polaritons.grid       import uniform_grid_and_weights
from polaritons.sigma      import (
	sweep_sigma, find_E_k_prime, assemble_Q,
	resume_unconverged, rerun_with_seed,
	refine_E_ext_grid, refine_sigma,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def p_nat():
	return Params().to_natural()


def _zero_kernel_factory(_p):
	"""Propagator that always returns zeros, regardless of arguments."""
	def F(q, Q, E_ext, eta=1.0):
		return np.zeros_like(np.asarray(q, dtype=float), dtype=complex)
	return F


# ---------------------------------------------------------------------------
# sweep_sigma
# ---------------------------------------------------------------------------

class TestSweepSigma:
	def test_output_shape(self, p_nat):
		q, w = uniform_grid_and_weights(5.0, 16)
		K    = np.zeros((16, 16))
		E    = np.linspace(-1.0, 1.0, 5)
		eta  = np.array([0.0, 0.5, 1.0])
		Sigma, iters = sweep_sigma(
			p_nat, K, q, w, E, eta,
			F_factory=_zero_kernel_factory, verbose=False,
		)
		assert Sigma.shape == (3, 5, 16)
		assert iters.shape == (3, 5)

	def test_zero_propagator_gives_zero_sigma(self, p_nat):
		q, w = uniform_grid_and_weights(5.0, 16)
		K    = np.eye(16)
		E    = np.linspace(-1.0, 1.0, 5)
		eta  = np.array([0.0, 0.5])
		Sigma, _ = sweep_sigma(
			p_nat, K, q, w, E, eta,
			F_factory=_zero_kernel_factory, verbose=False,
			tol=1e-12, max_iter=20,
		)
		np.testing.assert_allclose(Sigma, 0.0, atol=1e-12)

	def test_eta_zero_row_is_zero(self, p_nat):
		"""With eta=0 the picard fixed point is Q=0 so Sigma=0 too."""
		q, w = uniform_grid_and_weights(5.0, 16)
		K    = np.eye(16) * 0.01
		E    = np.linspace(-1.0, 1.0, 5)
		eta  = np.array([0.0, 0.3])
		Sigma, _ = sweep_sigma(p_nat, K, q, w, E, eta, verbose=False, max_iter=200)
		np.testing.assert_allclose(Sigma[0], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# find_E_k_prime
# ---------------------------------------------------------------------------

class TestFindEkPrime:
	def test_zero_sigma_gives_bare_band(self, p_nat):
		"""With Sigma=0, the root of E_ext - bare(k) is exactly bare(k)."""
		N    = 8
		n_E  = 21
		n_eta = 2
		q, _ = uniform_grid_and_weights(5.0, N)
		bare = (p_nat.hbar**2 * q**2) / (2.0 * p_nat.M)
		E_ext = np.linspace(bare.min() - 1.0, bare.max() + 1.0, n_E)
		Sigma = np.zeros((n_eta, n_E, N), dtype=complex)
		Ek = find_E_k_prime(Sigma, q, E_ext, p_nat)
		assert Ek.shape == (n_eta, N)
		for ei in range(n_eta):
			np.testing.assert_allclose(Ek[ei], bare, rtol=1e-12, atol=1e-12)

	def test_constant_real_sigma_shifts_root(self, p_nat):
		"""f = E - bare - Re[Sigma]; with Sigma = c (real), root = bare + c."""
		N    = 6
		n_E  = 41
		c    = 0.3
		q, _ = uniform_grid_and_weights(2.0, N)
		bare = (p_nat.hbar**2 * q**2) / (2.0 * p_nat.M)
		E_ext = np.linspace(-1.0, 2.0 + bare.max(), n_E)
		Sigma = np.full((1, n_E, N), c + 0j, dtype=complex)
		Ek = find_E_k_prime(Sigma, q, E_ext, p_nat)
		np.testing.assert_allclose(Ek[0], bare + c, atol=1e-10)

	def test_no_sign_change_returns_nan(self, p_nat):
		"""Sigma values pushing f always-positive give NaN."""
		N    = 4
		n_E  = 5
		q, _ = uniform_grid_and_weights(1.0, N)
		# Sigma = -100 makes f = E - bare + 100 > 0 on the chosen window.
		E_ext = np.linspace(-1.0, 1.0, n_E)
		Sigma = np.full((1, n_E, N), -100.0 + 0j, dtype=complex)
		Ek = find_E_k_prime(Sigma, q, E_ext, p_nat)
		assert np.all(np.isnan(Ek))


# ---------------------------------------------------------------------------
# assemble_Q
# ---------------------------------------------------------------------------

class TestAssembleQ:
	def test_q_equals_minus_sigma_at_grid_point(self, p_nat):
		"""If E_k' lands exactly on a grid point, Q = -Sigma at that point."""
		N    = 4
		n_E  = 7
		E_ext = np.linspace(-1.0, 1.0, n_E)
		Sigma = np.zeros((1, n_E, N), dtype=complex)
		# put a distinct value at the middle E_ext index
		mid = n_E // 2
		Sigma[0, mid] = np.array([1+1j, 2+2j, 3+3j, 4+4j])
		Ek = np.full((1, N), E_ext[mid], dtype=float)
		Q  = assemble_Q(Sigma, Ek, E_ext)
		np.testing.assert_allclose(Q[0], -Sigma[0, mid])

	def test_nan_propagates(self, p_nat):
		Sigma = np.zeros((1, 3, 2), dtype=complex)
		Ek = np.full((1, 2), np.nan, dtype=float)
		Q  = assemble_Q(Sigma, Ek, np.linspace(-1, 1, 3))
		assert np.all(np.isnan(Q.real)) and np.all(np.isnan(Q.imag))

	def test_round_trip_with_find_E_k_prime(self, p_nat):
		"""find -> assemble: Q = -Sigma evaluated at the on-shell E_k'."""
		N    = 5
		n_E  = 41
		c    = 0.2
		q, _ = uniform_grid_and_weights(2.0, N)
		bare = (p_nat.hbar**2 * q**2) / (2.0 * p_nat.M)
		E_ext = np.linspace(-1.0, 2.0 + bare.max(), n_E)
		Sigma = np.full((1, n_E, N), c + 0j, dtype=complex)
		Ek = find_E_k_prime(Sigma, q, E_ext, p_nat)
		Q  = assemble_Q(Sigma, Ek, E_ext)
		np.testing.assert_allclose(Q[0], np.full(N, -c, dtype=complex), atol=1e-10)

	def test_cubic_interp_improves_curved_sigma(self, p_nat):
		"""
		Synthetic test for the ``interp="cubic"`` reconstruction path.

		Build ``Sigma(E_ext, k)`` with a known smooth (non-linear) energy
		dependence, sample it on a *coarse* E_ext grid, and then recover
		``Q(k)`` two ways: linear (default) and cubic.  The exact on-shell
		``Q`` is ``-Sigma(E_k', k)`` with ``E_k'`` solved analytically.
		Cubic recovery must be at least 5x more accurate than linear and
		monotonic in ``k`` (the linear path produces an artificial bump
		identical to the gaussian-disorder anomaly being fixed).
		"""
		N      = 9
		n_E    = 21
		q, _   = uniform_grid_and_weights(1.5, N)
		bare   = (p_nat.hbar**2 * q**2) / (2.0 * p_nat.M)
		E_ext  = np.linspace(-0.5, 0.5 + bare.max(), n_E)

		# Smooth, non-linear Sigma(E) curvature on a sub-grid-spacing scale.
		# Small amplitude relative to grid spacing -> linear interp errs.
		dE  = E_ext[1] - E_ext[0]
		amp = 0.03 * dE
		# Sigma_re = amp * (E_ext - 0.1*bare[k])**2  -> curved in E, smooth in k
		Sigma = np.empty((1, n_E, N), dtype=complex)
		for k_idx in range(N):
			off = 0.1 * bare[k_idx]
			Sigma[0, :, k_idx] = amp * (E_ext - off)**2

		# True on-shell root for this synthetic model (closed form): solve
		# E - bare - amp*(E - 0.1*bare)**2 = 0 for the branch close to bare.
		def true_Q(k_idx):
			b = float(bare[k_idx])
			# amp*E**2 - (2*amp*0.1*b + 1)*E + (amp*(0.1*b)**2 + b) = 0
			A = amp
			B = -(2.0 * amp * 0.1 * b + 1.0)
			C = amp * (0.1 * b)**2 + b
			disc = B*B - 4*A*C
			# Two roots; pick the one closer to b (on-shell pole).
			r1 = (-B + np.sqrt(disc)) / (2 * A)
			r2 = (-B - np.sqrt(disc)) / (2 * A)
			Eprime = r1 if abs(r1 - b) < abs(r2 - b) else r2
			return -amp * (Eprime - 0.1 * b)**2   # Q = -Sigma(E')

		exact = np.array([true_Q(k) for k in range(N)], dtype=complex)

		Ek_lin = find_E_k_prime(Sigma, q, E_ext, p_nat, interp="linear")
		Q_lin  = assemble_Q(Sigma, Ek_lin, E_ext, interp="linear")[0]

		Ek_cub = find_E_k_prime(Sigma, q, E_ext, p_nat, interp="cubic")
		Q_cub  = assemble_Q(Sigma, Ek_cub, E_ext, interp="cubic")[0]

		err_lin = np.max(np.abs(Q_lin - exact))
		err_cub = np.max(np.abs(Q_cub - exact))

		assert err_cub < err_lin / 3.0, (err_lin, err_cub)

	def test_bad_interp_raises(self, p_nat):
		Sigma = np.zeros((1, 5, 3), dtype=complex)
		E     = np.linspace(-1.0, 1.0, 5)
		Ek    = np.zeros((1, 3))
		with pytest.raises(ValueError, match="interp"):
			assemble_Q(Sigma, Ek, E, interp="quadratic")
		with pytest.raises(ValueError, match="interp"):
			find_E_k_prime(
				Sigma, np.array([0.0, 0.1, 0.2]), E, p_nat, interp="quadratic"
			)


# ---------------------------------------------------------------------------
# Phase-2 reuse helpers
# ---------------------------------------------------------------------------

def _const_kernel_factory(value: complex):
	"""Q-independent propagator returning ``value`` everywhere."""
	def factory(_p):
		def F(q, Q, E_ext, eta=1.0):
			return np.full_like(np.asarray(q, dtype=float), value, dtype=complex)
		return F
	return factory


class TestSolveMaskAndSeed:
	def test_solve_mask_skips_cells(self, p_nat):
		"""Masked cells are copied straight from Sigma_seed and report iters=0."""
		N    = 8
		q, w = uniform_grid_and_weights(3.0, N)
		K    = np.eye(N)
		E    = np.linspace(-1.0, 1.0, 4)
		eta  = np.array([0.0, 0.5, 1.0])
		sentinel = (999.0 + 999.0j)
		Sigma_seed = np.full((3, 4, N), sentinel, dtype=complex)
		# Mask out the (eta=0.5, E_ext index 1) cell only.
		solve_mask = np.ones((3, 4), dtype=bool)
		solve_mask[1, 1] = False

		Sigma, iters = sweep_sigma(
			p_nat, K, q, w, E, eta,
			F_factory=_zero_kernel_factory,
			Sigma_seed=Sigma_seed,
			solve_mask=solve_mask,
			tol=1e-12, max_iter=20,
		)
		# Masked cell preserves sentinel exactly; iters==0 there.
		np.testing.assert_array_equal(Sigma[1, 1], np.full(N, sentinel))
		assert iters[1, 1] == 0
		# Non-masked, non-eta-zero cells solved against zero kernel ⇒ Sigma=0.
		np.testing.assert_allclose(Sigma[1, 0], 0.0, atol=1e-12)
		np.testing.assert_allclose(Sigma[2, :],  0.0, atol=1e-12)

	def test_sigma_seed_warmstarts(self, p_nat):
		"""Seeding with the converged Sigma should finish in ≤ 2 iterations."""
		N    = 6
		q, w = uniform_grid_and_weights(2.0, N)
		K    = 0.1 * np.eye(N)
		E    = np.linspace(-0.5, 0.5, 3)
		eta  = np.array([0.5, 1.0])
		F_factory = _const_kernel_factory(0.25 + 0.0j)

		Sigma_ref, _ = sweep_sigma(
			p_nat, K, q, w, E, eta,
			F_factory=F_factory,
			tol=1e-12, max_iter=200, w=1.0,
		)
		Sigma_warm, iters_warm = sweep_sigma(
			p_nat, K, q, w, E, eta,
			F_factory=F_factory,
			Sigma_seed=Sigma_ref,
			Q_init=-Sigma_ref,
			tol=1e-12, max_iter=5, w=1.0,
		)
		np.testing.assert_allclose(Sigma_warm, Sigma_ref, atol=1e-12)
		# eta!=0 cells must converge in ≤ 2 picard iterations from the seed.
		assert np.all(iters_warm[1:] <= 2)


class TestResumeUnconverged:
	def test_resume_unconverged_targets_iters_eq_max(self, p_nat):
		"""Only cells with iters_prev == prev_max_iter are re-solved."""
		N    = 6
		q, w = uniform_grid_and_weights(2.0, N)
		K    = np.eye(N)
		E    = np.linspace(-1.0, 1.0, 4)
		eta  = np.array([0.0, 0.5, 1.0])

		sentinel   = (7.0 + 3.0j)
		Sigma_prev = np.full((3, 4, N), sentinel, dtype=complex)
		iters_prev = np.full((3, 4), 10, dtype=int)
		prev_max_iter = 50
		# Flag exactly one non-eta-zero cell as unconverged.
		iters_prev[1, 2] = prev_max_iter
		# Also flag an eta=0 cell — must be ignored (eta=0 short-circuit).
		iters_prev[0, 0] = prev_max_iter

		Sigma_new, iters_new = resume_unconverged(
			p_nat, K, q, w, E, eta,
			Sigma_prev, iters_prev, prev_max_iter,
			F_factory=_zero_kernel_factory,
			w=1.0, max_iter=20, tol=1e-12,
		)
		# Only (eta=1, E=2) was re-solved → its Sigma now matches the zero
		# kernel fixed point (Sigma=0).
		np.testing.assert_allclose(Sigma_new[1, 2], 0.0, atol=1e-12)
		# Every other cell is identical to the input.
		mask = np.ones_like(iters_prev, dtype=bool)
		mask[1, 2] = False
		assert np.all(Sigma_new[mask] == Sigma_prev[mask])
		# Iters preserved for skipped cells; recorded for the resolved one.
		assert np.array_equal(iters_new[mask], iters_prev[mask])
		assert iters_new[1, 2] >= 1


class TestRefineEExtGrid:
	def test_refine_E_ext_grid_merges_uniquely(self, p_nat):
		E_old = np.linspace(-0.5, 0.5, 6)
		E_new, new_mask, old_to_new = refine_E_ext_grid(
			E_old, p_nat, t_local=5.0, n_add=7, band_pad_omega=0.5,
		)
		# Strictly increasing and contains the old grid.
		assert np.all(np.diff(E_new) > 0)
		np.testing.assert_allclose(E_new[old_to_new], E_old)
		# new_mask is False precisely at old grid positions.
		assert not new_mask[old_to_new].any()
		assert int(new_mask.sum()) == E_new.size - E_old.size
		# All inserted points fall in the requested band.
		bare_max = p_nat.hbar**2 * 5.0**2 / (2.0 * p_nat.M)
		pad      = 0.5 * p_nat.Omega
		inserted = E_new[new_mask]
		assert inserted.min() >= -pad - 1e-12
		assert inserted.max() <= bare_max + pad + 1e-12


class TestRefineSigma:
	def test_refine_sigma_preserves_old_columns(self, p_nat):
		"""Sigma at old E_ext indices must equal the input bit-for-bit."""
		N    = 5
		q, w = uniform_grid_and_weights(2.0, N)
		K    = np.eye(N)
		E_old = np.linspace(-0.5, 0.5, 5)
		eta   = np.array([0.0, 0.5])
		rng = np.random.default_rng(0)
		Sigma_prev = (
			rng.standard_normal((2, 5, N)) + 1j * rng.standard_normal((2, 5, N))
		)
		# eta=0 row must be exactly zero (sweep_sigma enforces it).
		Sigma_prev[0] = 0.0

		Sigma_new, iters_new, E_new = refine_sigma(
			p_nat, K, q, w, E_old, eta, Sigma_prev, t_local=3.0,
			n_add=6, band_pad_omega=0.5,
			F_factory=_zero_kernel_factory,
			w=1.0, max_iter=20, tol=1e-12,
		)
		_, new_mask, old_to_new = refine_E_ext_grid(
			E_old, p_nat, t_local=3.0, n_add=6, band_pad_omega=0.5,
		)
		# Old columns are preserved exactly.
		np.testing.assert_array_equal(
			Sigma_new[:, old_to_new, :], Sigma_prev,
		)
		# Iters at copied (non-mask) columns is zero.
		assert np.all(iters_new[:, old_to_new] == 0)
		# Refined grid round-trips.
		np.testing.assert_array_equal(E_new[old_to_new], E_old)
		# Inserted columns at eta!=0 with zero kernel converge to Sigma=0.
		if new_mask.any():
			np.testing.assert_allclose(
				Sigma_new[1][new_mask], 0.0, atol=1e-12,
			)
