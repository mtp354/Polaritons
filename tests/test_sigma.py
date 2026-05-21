"""
Unit tests for polaritons.sigma (sweep_sigma, find_E_k_prime, assemble_Q).
"""
import numpy as np
import pytest

from polaritons.parameters import Params
from polaritons.grid       import uniform_grid_and_weights
from polaritons.sigma      import sweep_sigma, find_E_k_prime, assemble_Q, refine_sigma_per_k


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


# ---------------------------------------------------------------------------
# refine_sigma_per_k
# ---------------------------------------------------------------------------

class TestRefineSigmaPerK:
	def test_zero_propagator_recovers_bare(self, p_nat):
		"""Zero kernel => Sigma=0, so refined E_k' = bare(k) exactly."""
		N = 6
		q, w = uniform_grid_and_weights(2.0, N)
		K    = np.eye(N) * 0.01
		bare = (p_nat.hbar**2 * q**2) / (2.0 * p_nat.M)
		# Seed close to bare so the tight window brackets the root.
		E_seed = bare + 0.05
		Ek, Q, Sig_col, E_per_k, iters = refine_sigma_per_k(
			p_nat, K, q, w, eta=0.5, E_seed=E_seed,
			half_width=0.5, n_E_pk=11,
			F_factory=_zero_kernel_factory, tol=1e-12, max_iter=20, verbose=False,
		)
		assert Ek.shape == (N,)
		assert Q.shape == (N,)
		assert Sig_col.shape == (11, N)
		assert E_per_k.shape == (11, N)
		assert iters.shape == (11, N)
		np.testing.assert_allclose(Ek, bare, atol=1e-10)
		np.testing.assert_allclose(Q, 0.0, atol=1e-10)

	def test_nan_seed_falls_back_to_bare(self, p_nat):
		"""Non-finite seed values are replaced by the bare exciton energy."""
		N = 4
		q, w = uniform_grid_and_weights(1.5, N)
		K    = np.eye(N) * 0.01
		bare = (p_nat.hbar**2 * q**2) / (2.0 * p_nat.M)
		E_seed = bare.copy()
		E_seed[0] = np.nan
		E_seed[2] = np.inf
		Ek, _, _, _, _ = refine_sigma_per_k(
			p_nat, K, q, w, eta=0.5, E_seed=E_seed,
			half_width=0.3, n_E_pk=9,
			F_factory=_zero_kernel_factory, tol=1e-12, max_iter=20, verbose=False,
		)
		# No NaN should propagate; bare-band fallback brackets E_k' = bare(k).
		assert np.all(np.isfinite(Ek))
		np.testing.assert_allclose(Ek, bare, atol=1e-10)

	def test_eta_zero_gives_bare(self, p_nat):
		"""eta=0 collapses Picard to Q=0, so E_k' = bare(k)."""
		N = 5
		q, w = uniform_grid_and_weights(2.0, N)
		K    = np.eye(N)
		bare = (p_nat.hbar**2 * q**2) / (2.0 * p_nat.M)
		E_seed = bare + 0.1
		Ek, Q, _, _, _ = refine_sigma_per_k(
			p_nat, K, q, w, eta=0.0, E_seed=E_seed,
			half_width=0.4, n_E_pk=7, verbose=False,
		)
		np.testing.assert_allclose(Ek, bare, atol=1e-12)
		np.testing.assert_allclose(Q, 0.0, atol=1e-12)
