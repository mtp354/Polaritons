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
    g_eff_lowest_mode,
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
    def test_shape(self, flat_disorder, p_si):
        H = lp_hamiltonian(N_SMALL, L_SMALL, M_lp=p_si.M_eff, V=flat_disorder, hbar=p_si.hbar)
        assert H.shape == (N_SMALL**2, N_SMALL**2)

    def test_symmetric_for_real_potential(self, flat_disorder, p_si):
        H = lp_hamiltonian(N_SMALL, L_SMALL, M_lp=p_si.M_eff, V=flat_disorder, hbar=p_si.hbar)
        diff = (H - H.T).toarray()
        np.testing.assert_allclose(diff, 0.0, atol=1e-10)

    def test_constant_potential_shifts_energies(self, p_si):
        """Adding a constant V to the potential shifts all eigenvalues by that amount."""
        V_flat = np.zeros((N_SMALL, N_SMALL))
        V_shift = np.ones((N_SMALL, N_SMALL)) * 0.5

        H0 = lp_hamiltonian(N_SMALL, L_SMALL, p_si.M_eff, V_flat,  p_si.hbar)
        H1 = lp_hamiltonian(N_SMALL, L_SMALL, p_si.M_eff, V_shift, p_si.hbar)

        vals0, _ = solve_low_modes(H0, n_modes=3)
        vals1, _ = solve_low_modes(H1, n_modes=3)
        np.testing.assert_allclose(vals1, vals0 + 0.5, rtol=1e-5)


# ---------------------------------------------------------------------------
# solve_low_modes
# ---------------------------------------------------------------------------

class TestSolveLowModes:
    def test_eigenvalues_sorted(self, flat_disorder, p_si):
        H    = lp_hamiltonian(N_SMALL, L_SMALL, p_si.M_eff, flat_disorder, p_si.hbar)
        vals, _ = solve_low_modes(H, n_modes=4)
        assert np.all(np.diff(vals) >= 0)

    def test_eigenvector_shape(self, flat_disorder, p_si):
        H    = lp_hamiltonian(N_SMALL, L_SMALL, p_si.M_eff, flat_disorder, p_si.hbar)
        _, vecs = solve_low_modes(H, n_modes=3)
        assert vecs.shape == (N_SMALL**2, 3)

    def test_eigenvalues_real(self, flat_disorder, p_si):
        """Symmetric Hamiltonian must have real eigenvalues."""
        H    = lp_hamiltonian(N_SMALL, L_SMALL, p_si.M_eff, flat_disorder, p_si.hbar)
        vals, _ = solve_low_modes(H, n_modes=3)
        np.testing.assert_allclose(vals.imag, 0.0, atol=1e-10)

    def test_eigenvectors_orthonormal(self, flat_disorder, p_si):
        """Eigenvectors returned by eigsh should be orthonormal."""
        H    = lp_hamiltonian(N_SMALL, L_SMALL, p_si.M_eff, flat_disorder, p_si.hbar)
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
# g_eff_lowest_mode
# ---------------------------------------------------------------------------

class TestGEffLowestMode:
    def test_returns_positive_float(self, p_si):
        result = g_eff_lowest_mode(
            N=N_SMALL, L=L_SMALL,
            M_lp=p_si.M_eff,
            sigma=0.01, xi=0.1,
            hbar=p_si.hbar,
            g_lp=1.0,
            seed=0,
        )
        assert isinstance(result, float)
        assert result > 0

    def test_reproducible_with_seed(self, p_si):
        kw = dict(N=N_SMALL, L=L_SMALL, M_lp=p_si.M_eff,
                  sigma=0.01, xi=0.1, hbar=p_si.hbar, g_lp=1.0)
        r1 = g_eff_lowest_mode(**kw, seed=42)
        r2 = g_eff_lowest_mode(**kw, seed=42)
        assert r1 == pytest.approx(r2)

    def test_scales_with_g_lp(self, p_si):
        """g_eff ∝ g_lp (IPR factor is the same)."""
        kw = dict(N=N_SMALL, L=L_SMALL, M_lp=p_si.M_eff,
                  sigma=0.01, xi=0.1, hbar=p_si.hbar, seed=7)
        r1 = g_eff_lowest_mode(**kw, g_lp=1.0)
        r2 = g_eff_lowest_mode(**kw, g_lp=2.0)
        assert r2 == pytest.approx(2.0 * r1, rel=1e-10)

    def test_larger_sigma_increases_ipr(self, p_si):
        """
        Stronger disorder concentrates the ground mode, increasing IPR
        and therefore g_eff.  Test with large σ vs near-zero σ.
        """
        kw = dict(N=N_SMALL, L=L_SMALL, M_lp=p_si.M_eff,
                  xi=0.1, hbar=p_si.hbar, g_lp=1.0, seed=3)
        g_weak   = g_eff_lowest_mode(**kw, sigma=1e-6)
        g_strong = g_eff_lowest_mode(**kw, sigma=1.0)
        assert g_strong > g_weak
