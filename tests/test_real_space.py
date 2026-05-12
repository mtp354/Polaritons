"""
Unit tests for polaritons.real_space.
"""
import numpy as np
import pytest

from polaritons.real_space import (
	gaussian_correlated_disorder,
	laplacian_2d,
	lp_hamiltonian,
	solve_low_modes,
	normalize_mode,
	ipr,
	mode_area,
	g_eff_lowest_mode,
	RealSpaceConfig,
	exciton_hamiltonian,
	plane_wave,
	lattice_kinetic_energy,
	continuum_k,
	integer_shells,
	green_for_realization,
	averaged_green_and_Q,
	sweep_disorder_self_energy,
)

from polaritons.parameters import Params

# ---------------------------------------------------------------------------
# Shared small-system parameters
# ---------------------------------------------------------------------------

N_SMALL = 16    # small enough for dense eigensolver comparison
L_SMALL = 1.0   # arbitrary box size (meters or natural units — self-consistent)


@pytest.fixture(scope="module")
def dx_small():
	return L_SMALL / N_SMALL


@pytest.fixture(scope="module")
def flat_disorder():
	"""Zero disorder potential on a small grid."""
	return np.zeros((N_SMALL, N_SMALL))


@pytest.fixture(scope="module")
def p_si():
	return Params()


# ---------------------------------------------------------------------------
# gaussian_correlated_disorder
# ---------------------------------------------------------------------------

class TestGaussianCorrelatedDisorder:
	def test_output_shape(self):
		V = gaussian_correlated_disorder(N=8, L=1.0, sigma=0.01, xi=0.1)
		assert V.shape == (8, 8)

	def test_zero_mean(self):
		V = gaussian_correlated_disorder(N=32, L=1.0, sigma=0.05, xi=0.1, seed=42)
		assert V.mean() == pytest.approx(0.0, abs=1e-14)

	def test_correct_std(self):
		sigma = 0.05
		V = gaussian_correlated_disorder(N=64, L=1.0, sigma=sigma, xi=0.05, seed=0)
		assert V.std() == pytest.approx(sigma, rel=1e-6)

	def test_reproducible_with_seed(self):
		V1 = gaussian_correlated_disorder(N=16, L=1.0, sigma=0.01, xi=0.1, seed=99)
		V2 = gaussian_correlated_disorder(N=16, L=1.0, sigma=0.01, xi=0.1, seed=99)
		np.testing.assert_array_equal(V1, V2)

	def test_different_seeds_differ(self):
		V1 = gaussian_correlated_disorder(N=16, L=1.0, sigma=0.01, xi=0.1, seed=1)
		V2 = gaussian_correlated_disorder(N=16, L=1.0, sigma=0.01, xi=0.1, seed=2)
		assert not np.allclose(V1, V2)

	def test_no_seed_returns_array(self):
		V = gaussian_correlated_disorder(N=8, L=1.0, sigma=0.01, xi=0.1)
		assert isinstance(V, np.ndarray)
		assert V.shape == (8, 8)

	def test_sigma_scales_amplitude(self):
		"""Doubling sigma should double the std."""
		V1 = gaussian_correlated_disorder(N=32, L=1.0, sigma=0.01, xi=0.1, seed=5)
		V2 = gaussian_correlated_disorder(N=32, L=1.0, sigma=0.02, xi=0.1, seed=5)
		assert V2.std() == pytest.approx(2.0 * V1.std(), rel=1e-6)


# ---------------------------------------------------------------------------
# laplacian_2d
# ---------------------------------------------------------------------------

class TestLaplacian2d:
	def test_shape(self, dx_small):
		L = laplacian_2d(N_SMALL, dx_small)
		assert L.shape == (N_SMALL**2, N_SMALL**2)

	def test_interior_row_sum_near_zero(self, dx_small):
		"""
		Periodic-like FD Laplacian: every row sums to 0.
		The implementation uses Dirichlet-like boundary (open ends),
		so interior rows sum to 0.
		"""
		L  = laplacian_2d(N_SMALL, dx_small)
		L_dense = L.toarray()
		# Check a generic interior row (not on the boundary of the 2-D grid)
		interior_row = N_SMALL + 1   # safely in the interior
		row_sum = L_dense[interior_row].sum()
		assert row_sum == pytest.approx(0.0, abs=1e-10)

	def test_symmetric(self, dx_small):
		L = laplacian_2d(N_SMALL, dx_small)
		diff = (L - L.T).toarray()
		np.testing.assert_allclose(diff, 0.0, atol=1e-12)

	def test_negative_diagonal(self, dx_small):
		"""Diagonal elements of the FD Laplacian are all negative."""
		L = laplacian_2d(N_SMALL, dx_small)
		diag = L.diagonal()
		assert np.all(diag < 0)


# ---------------------------------------------------------------------------
# lp_hamiltonian
# ---------------------------------------------------------------------------

class TestLpHamiltonian:
	def test_shape(self, flat_disorder, p_si, M_lp_si):
		H = lp_hamiltonian(N_SMALL, L_SMALL, M_lp=M_lp_si, V=flat_disorder, hbar=p_si.hbar)
		assert H.shape == (N_SMALL**2, N_SMALL**2)

	def test_symmetric_for_real_potential(self, flat_disorder, p_si, M_lp_si):
		H = lp_hamiltonian(N_SMALL, L_SMALL, M_lp=M_lp_si, V=flat_disorder, hbar=p_si.hbar)
		diff = (H - H.T).toarray()
		np.testing.assert_allclose(diff, 0.0, atol=1e-10)

	def test_constant_potential_shifts_energies(self, p_si, M_lp_si):
		"""Adding a constant V to the potential shifts all eigenvalues by that amount."""
		V_flat = np.zeros((N_SMALL, N_SMALL))
		V_shift = np.ones((N_SMALL, N_SMALL)) * 0.5

		H0 = lp_hamiltonian(N_SMALL, L_SMALL, M_lp_si, V_flat,  p_si.hbar)
		H1 = lp_hamiltonian(N_SMALL, L_SMALL, M_lp_si, V_shift, p_si.hbar)

		vals0, _ = solve_low_modes(H0, n_modes=3)
		vals1, _ = solve_low_modes(H1, n_modes=3)
		np.testing.assert_allclose(vals1, vals0 + 0.5, rtol=1e-5)


# ---------------------------------------------------------------------------
# solve_low_modes
# ---------------------------------------------------------------------------

class TestSolveLowModes:
	def test_eigenvalues_sorted(self, flat_disorder, p_si, M_lp_si):
		H    = lp_hamiltonian(N_SMALL, L_SMALL, M_lp_si, flat_disorder, p_si.hbar)
		vals, _ = solve_low_modes(H, n_modes=4)
		assert np.all(np.diff(vals) >= 0)

	def test_eigenvector_shape(self, flat_disorder, p_si, M_lp_si):
		H    = lp_hamiltonian(N_SMALL, L_SMALL, M_lp_si, flat_disorder, p_si.hbar)
		_, vecs = solve_low_modes(H, n_modes=3)
		assert vecs.shape == (N_SMALL**2, 3)

	def test_eigenvalues_real(self, flat_disorder, p_si, M_lp_si):
		"""Symmetric Hamiltonian must have real eigenvalues."""
		H    = lp_hamiltonian(N_SMALL, L_SMALL, M_lp_si, flat_disorder, p_si.hbar)
		vals, _ = solve_low_modes(H, n_modes=3)
		np.testing.assert_allclose(vals.imag, 0.0, atol=1e-10)

	def test_eigenvectors_orthonormal(self, flat_disorder, p_si, M_lp_si):
		"""Eigenvectors returned by eigsh should be orthonormal."""
		H    = lp_hamiltonian(N_SMALL, L_SMALL, M_lp_si, flat_disorder, p_si.hbar)
		_, vecs = solve_low_modes(H, n_modes=4)
		gram = vecs.T @ vecs
		np.testing.assert_allclose(gram, np.eye(4), atol=1e-10)


# ---------------------------------------------------------------------------
# normalize_mode
# ---------------------------------------------------------------------------

class TestNormalizeMode:
	def test_norm_is_one(self, dx_small):
		rng = np.random.default_rng(3)
		psi = rng.normal(size=(N_SMALL, N_SMALL))
		psi_norm = normalize_mode(psi, dx_small)
		norm = np.sqrt(np.sum(np.abs(psi_norm)**2) * dx_small**2)
		assert norm == pytest.approx(1.0, rel=1e-12)

	def test_shape_preserved(self, dx_small):
		psi = np.ones((N_SMALL, N_SMALL))
		psi_norm = normalize_mode(psi, dx_small)
		assert psi_norm.shape == psi.shape

	def test_uniform_wavefunction(self, dx_small):
		"""Uniform psi = 1/(L) on a grid should normalise to 1/L."""
		psi  = np.ones((N_SMALL, N_SMALL))
		psi_n = normalize_mode(psi, dx_small)
		expected = 1.0 / L_SMALL
		np.testing.assert_allclose(psi_n, expected, rtol=1e-10)

	def test_complex_wavefunction(self, dx_small):
		rng = np.random.default_rng(11)
		psi = rng.normal(size=(N_SMALL, N_SMALL)) + 1j * rng.normal(size=(N_SMALL, N_SMALL))
		psi_n = normalize_mode(psi, dx_small)
		norm  = np.sqrt(np.sum(np.abs(psi_n)**2) * dx_small**2)
		assert norm == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# ipr
# ---------------------------------------------------------------------------

class TestIPR:
	def test_positive(self, dx_small):
		rng = np.random.default_rng(7)
		psi = rng.normal(size=(N_SMALL, N_SMALL))
		psi_n = normalize_mode(psi, dx_small)
		assert ipr(psi_n, dx_small) > 0

	def test_upper_bound_uniform(self, dx_small):
		"""
		A uniform normalised state on a grid of area L^2 has IPR = 1/L^2.

		Proof: if ψ = 1/L on the domain, then |ψ|^4 = 1/L^4 everywhere,
		and IPR = ∫|ψ|^4 d²r = (1/L^4) * L^2 = 1/L^2.
		"""
		psi   = np.ones((N_SMALL, N_SMALL))
		psi_n = normalize_mode(psi, dx_small)
		i     = ipr(psi_n, dx_small)
		expected = 1.0 / L_SMALL**2
		assert i == pytest.approx(expected, rel=1e-8)

	def test_localised_gt_delocalised(self, dx_small):
		"""A localised state should have higher IPR than a delocalised one."""
		# Delocalised (uniform)
		psi_del = normalize_mode(np.ones((N_SMALL, N_SMALL)), dx_small)
		# Localised (spike at one point)
		psi_loc = np.zeros((N_SMALL, N_SMALL))
		psi_loc[N_SMALL // 2, N_SMALL // 2] = 1.0
		psi_loc = normalize_mode(psi_loc, dx_small)

		assert ipr(psi_loc, dx_small) > ipr(psi_del, dx_small)

	def test_returns_float(self, dx_small):
		psi_n = normalize_mode(np.ones((N_SMALL, N_SMALL)), dx_small)
		assert isinstance(ipr(psi_n, dx_small), float)


# ---------------------------------------------------------------------------
# mode_area
# ---------------------------------------------------------------------------

class TestModeArea:
	def test_uniform_state_equals_box_area(self, dx_small):
		"""For a uniform normalised state IPR = 1/L^2 so mode_area = L^2."""
		psi_n = normalize_mode(np.ones((N_SMALL, N_SMALL)), dx_small)
		assert mode_area(psi_n, dx_small) == pytest.approx(L_SMALL**2, rel=1e-8)

	def test_inverse_of_ipr(self, dx_small):
		rng = np.random.default_rng(13)
		psi_n = normalize_mode(rng.normal(size=(N_SMALL, N_SMALL)), dx_small)
		assert mode_area(psi_n, dx_small) == pytest.approx(1.0 / ipr(psi_n, dx_small), rel=1e-12)


# ---------------------------------------------------------------------------
# Gaussian disorder correlation length (empirical fit of autocorrelation)
# ---------------------------------------------------------------------------

class TestCorrelationLength:
	def test_radial_autocorr_matches_documented_convention(self):
		"""
		Documented convention: <V(r)V(r')> ∝ exp(-|r-r'|^2 / (2 xi^2)).

		At r = xi the normalised autocorr should be exp(-1/2) ≈ 0.6065.
		Tests a single large realisation; tolerance accounts for finite-N noise.
		"""
		N, L, xi = 256, 1.0, 0.05
		dx = L / N
		V  = gaussian_correlated_disorder(N=N, L=L, sigma=1.0, xi=xi, seed=0)
		# Wiener-Khinchin: autocorrelation = IFFT(|FFT(V)|^2) / N^2
		C = np.fft.ifftn(np.abs(np.fft.fftn(V))**2).real
		C /= C[0, 0]  # normalise so C(0)=1

		# Look along the +x axis up to half the box.
		half = N // 2
		r    = np.arange(half) * dx
		Cx   = C[:half, 0]

		idx = int(round(xi / dx))
		assert Cx[idx] == pytest.approx(np.exp(-0.5), abs=0.05)


# ---------------------------------------------------------------------------
# g_eff_lowest_mode
# ---------------------------------------------------------------------------

class TestGEffLowestMode:
	def test_returns_positive_float(self, p_si, M_lp_si):
		result = g_eff_lowest_mode(
			N=N_SMALL, L=L_SMALL,
			M_lp=M_lp_si,
			sigma=0.01, xi=0.1,
			hbar=p_si.hbar,
			g_lp=1.0,
			seed=0,
		)
		assert isinstance(result, float)
		assert result > 0

	def test_reproducible_with_seed(self, p_si, M_lp_si):
		kw = dict(N=N_SMALL, L=L_SMALL, M_lp=M_lp_si,
				  sigma=0.01, xi=0.1, hbar=p_si.hbar, g_lp=1.0)
		r1 = g_eff_lowest_mode(**kw, seed=42)
		r2 = g_eff_lowest_mode(**kw, seed=42)
		assert r1 == pytest.approx(r2)

	def test_scales_with_g_lp(self, p_si, M_lp_si):
		"""g_eff ∝ g_lp (IPR factor is the same)."""
		kw = dict(N=N_SMALL, L=L_SMALL, M_lp=M_lp_si,
				  sigma=0.01, xi=0.1, hbar=p_si.hbar, seed=7)
		r1 = g_eff_lowest_mode(**kw, g_lp=1.0)
		r2 = g_eff_lowest_mode(**kw, g_lp=2.0)
		assert r2 == pytest.approx(2.0 * r1, rel=1e-10)

	def test_larger_sigma_increases_ipr(self, p_si, M_lp_si):
		"""
		Stronger disorder concentrates the ground mode, increasing IPR
		and therefore g_eff.  Test with large σ vs near-zero σ.
		"""
		kw = dict(N=N_SMALL, L=L_SMALL, M_lp=M_lp_si,
				  xi=0.1, hbar=p_si.hbar, g_lp=1.0, seed=3)
		g_weak   = g_eff_lowest_mode(**kw, sigma=1e-6)
		g_strong = g_eff_lowest_mode(**kw, sigma=1.0)
		assert g_strong > g_weak


# ===========================================================================
# Q-extraction pipeline (RealSpaceConfig + Green-function utilities)
# ===========================================================================

@pytest.fixture(scope="module")
def qcfg():
	# Small periodic box: N=8 keeps the dense N^2 x N^2 = 64x64 inversion fast
	# while still resolving a few k-shells.
	return RealSpaceConfig(N=8, L=1.0, M=1.0, hbar=1.0, epsilon=1e-3)


# ---------------------------------------------------------------------------
# laplacian_2d periodic mode
# ---------------------------------------------------------------------------

class TestPeriodicLaplacian:
	def test_periodic_row_sums_zero(self):
		L = laplacian_2d(8, 0.1, periodic=True).toarray()
		# Every row of a periodic FD Laplacian sums to zero.
		np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-10)

	def test_periodic_symmetric(self):
		L = laplacian_2d(8, 0.1, periodic=True)
		np.testing.assert_allclose((L - L.T).toarray(), 0.0, atol=1e-12)

	def test_periodic_differs_from_open(self):
		Lp = laplacian_2d(8, 0.1, periodic=True).toarray()
		Lo = laplacian_2d(8, 0.1, periodic=False).toarray()
		assert not np.allclose(Lp, Lo)


# ---------------------------------------------------------------------------
# RealSpaceConfig
# ---------------------------------------------------------------------------

class TestRealSpaceConfig:
	def test_dx_property(self):
		c = RealSpaceConfig(N=10, L=2.0, M=1.0)
		assert c.dx == pytest.approx(0.2)

	def test_frozen(self):
		c = RealSpaceConfig(N=10, L=2.0, M=1.0)
		with pytest.raises(Exception):
			c.N = 20  # type: ignore[misc]


# ---------------------------------------------------------------------------
# plane_wave
# ---------------------------------------------------------------------------

class TestPlaneWave:
	def test_normalised(self, qcfg):
		phi = plane_wave(qcfg, 1, 0)
		# Discrete L^2 norm with measure dx^2 should equal 1.
		norm2 = qcfg.dx**2 * np.vdot(phi, phi).real
		assert norm2 == pytest.approx(1.0, rel=1e-12)

	def test_orthogonal_distinct_modes(self, qcfg):
		p1 = plane_wave(qcfg, 1, 0)
		p2 = plane_wave(qcfg, 0, 1)
		overlap = qcfg.dx**2 * np.vdot(p1, p2)
		assert abs(overlap) == pytest.approx(0.0, abs=1e-12)

	def test_zero_mode_uniform(self, qcfg):
		phi = plane_wave(qcfg, 0, 0)
		expected = 1.0 / qcfg.L
		np.testing.assert_allclose(phi, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Lattice/continuum kinetic energies
# ---------------------------------------------------------------------------

class TestKineticEnergies:
	def test_lattice_zero_at_zero_momentum(self, qcfg):
		assert lattice_kinetic_energy(qcfg, 0, 0) == pytest.approx(0.0, abs=1e-12)

	def test_continuum_zero_at_zero_momentum(self, qcfg):
		assert continuum_k(qcfg, 0, 0) == pytest.approx(0.0, abs=1e-12)

	def test_continuum_isotropic(self, qcfg):
		assert continuum_k(qcfg, 1, 0) == pytest.approx(continuum_k(qcfg, 0, 1))

	def test_lattice_matches_continuum_small_n(self, qcfg):
		"""For small (nx, ny) the lattice dispersion → ħ² k² / 2M."""
		nx, ny = 1, 0
		k = continuum_k(qcfg, nx, ny)
		eps_cont = qcfg.hbar**2 * k**2 / (2 * qcfg.M)
		eps_lat  = lattice_kinetic_energy(qcfg, nx, ny)
		assert eps_lat == pytest.approx(eps_cont, rel=0.1)


# ---------------------------------------------------------------------------
# integer_shells
# ---------------------------------------------------------------------------

class TestIntegerShells:
	def test_first_shell_is_zero(self):
		shells = integer_shells(n_shells=4)
		assert shells[0] == [(0, 0)]

	def test_exclude_zero(self):
		shells = integer_shells(n_shells=3, include_zero=False)
		# (0,0) must not appear in any shell.
		for shell in shells:
			assert (0, 0) not in shell

	def test_shells_grouped_by_n2(self):
		shells = integer_shells(n_shells=5)
		for shell in shells:
			n2_values = {nx*nx + ny*ny for nx, ny in shell}
			assert len(n2_values) == 1

	def test_shells_strictly_increasing(self):
		shells = integer_shells(n_shells=6)
		n2s = [shell[0][0]**2 + shell[0][1]**2 for shell in shells]
		assert n2s == sorted(set(n2s))


# ---------------------------------------------------------------------------
# exciton_hamiltonian + plane-wave eigenequation
# ---------------------------------------------------------------------------

class TestExcitonHamiltonian:
	def test_zero_potential_plane_wave_eigenstate(self, qcfg):
		"""
		With V=0 and periodic boundaries, plane waves are exact eigenvectors
		of H with eigenvalue equal to lattice_kinetic_energy.
		"""
		V = np.zeros((qcfg.N, qcfg.N))
		H = exciton_hamiltonian(qcfg, V)
		nx, ny = 1, 0
		phi = plane_wave(qcfg, nx, ny)
		Hphi = H @ phi
		eps = lattice_kinetic_energy(qcfg, nx, ny)
		np.testing.assert_allclose(Hphi, eps * phi, atol=1e-10)


# ---------------------------------------------------------------------------
# green_for_realization
# ---------------------------------------------------------------------------

class TestGreenForRealization:
	def test_zero_disorder_matches_bare_resolvent(self, qcfg):
		"""
		With V=0,  G(k, E) = 1 / (E + iε - ε_k)  exactly.
		"""
		V = np.zeros((qcfg.N, qcfg.N))
		E = 5.0
		shell = [(1, 0), (0, 1), (1, 1)]
		G = green_for_realization(qcfg, V, E, shell)
		expected = np.array([
			1.0 / (E + 1j * qcfg.epsilon - lattice_kinetic_energy(qcfg, nx, ny))
			for nx, ny in shell
		])
		np.testing.assert_allclose(G, expected, rtol=1e-8, atol=1e-10)

	def test_output_length(self, qcfg):
		V = gaussian_correlated_disorder(qcfg.N, qcfg.L, sigma=0.01, xi=0.1, seed=0)
		G = green_for_realization(qcfg, V, E=1.0, shell_vectors=[(1, 0), (0, 1)])
		assert G.shape == (2,)


# ---------------------------------------------------------------------------
# averaged_green_and_Q
# ---------------------------------------------------------------------------

class TestAveragedGreenAndQ:
	def test_zero_disorder_Q_is_zero(self, qcfg):
		"""With σ=0 the averaged G is the bare resolvent, so Q = 0 identically."""
		shells = integer_shells(n_shells=4)
		k_vals, G_avg, Q = averaged_green_and_Q(
			qcfg, E=5.0, sigma=0.0, xi=0.1, shells=shells, n_realizations=2,
		)
		assert k_vals.shape == (4,)
		assert G_avg.shape == (4,)
		assert Q.shape == (4,)
		np.testing.assert_allclose(Q, 0.0, atol=1e-8)

	def test_nonzero_disorder_Q_imaginary_nonnegative(self, qcfg):
		"""
		With the retarded convention G = 1/(E + iε - H), disorder broadening
		makes Im(1/G_avg) ≥ ε, so the extracted Im Q = Im(1/G_avg) - ε ≥ 0.
		"""
		shells = integer_shells(n_shells=3, include_zero=False)
		_, _, Q = averaged_green_and_Q(
			qcfg, E=5.0, sigma=0.5, xi=0.1, shells=shells,
			n_realizations=4, seed0=0,
		)
		assert np.all(Q.imag >= -1e-10)


# ---------------------------------------------------------------------------
# sweep_disorder_self_energy
# ---------------------------------------------------------------------------

class TestSweepDisorderSelfEnergy:
	def test_output_shapes(self, qcfg):
		shells = integer_shells(n_shells=3)
		sigmas = [0.0, 0.1, 0.2]
		k_vals, sig_out, G_res, Q_res = sweep_disorder_self_energy(
			qcfg, E=5.0, sigmas=sigmas, xi=0.1,
			shells=shells, n_realizations=2,
		)
		assert k_vals.shape == (3,)
		assert sig_out.shape == (3,)
		assert G_res.shape == (3, 3)
		assert Q_res.shape == (3, 3)

	def test_first_row_zero_disorder_zero_Q(self, qcfg):
		shells = integer_shells(n_shells=3)
		_, _, _, Q_res = sweep_disorder_self_energy(
			qcfg, E=5.0, sigmas=[0.0, 0.1], xi=0.1,
			shells=shells, n_realizations=2,
		)
		np.testing.assert_allclose(Q_res[0], 0.0, atol=1e-8)
