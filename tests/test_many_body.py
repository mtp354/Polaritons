"""
Unit tests for polaritons.many_body.

Expensive functions (chemical_potential, Pi0, polariton_interaction_strength
with screening) are marked @pytest.mark.slow and skipped by default.
Run them with:  pytest -m slow
"""
import numpy as np
import pytest

from polaritons.many_body import (
	hopfield_coefficients,
	effective_mass,
	_fugacity_series_coefficients,
	polariton_interaction_strength,
)


# ---------------------------------------------------------------------------
# Hopfield coefficients
# ---------------------------------------------------------------------------

class TestHopfieldCoefficients:
	def test_sum_rule(self, simple_model, q_grid_nat, eta_grid):
		"""
		|X_LP|² + |C_LP|² = 1 for every k and eta.

		This is a pure algebraic identity and must hold exactly.
		"""
		k_arr   = simple_model.q_grid_nat[1:8]
		for eta in eta_grid:
			E_lp = simple_model.E_LP(k_arr, eta)
			X, C = hopfield_coefficients(simple_model, eta, k_arr, E_lp)
			norm_sq = np.abs(X)**2 + np.abs(C)**2
			np.testing.assert_allclose(norm_sq, 1.0, atol=1e-10)

	def test_resonance_equal_mixing(self, simple_model, eta_grid):
		"""
		At resonance (k=0, E_LP = E_0 - Omega) the exciton fraction
		satisfies |X_LP|² = 1/2.
		"""
		for eta in eta_grid:
			k_arr = simple_model.q_grid_nat[:1]   # k=0
			E_lp  = simple_model.E_LP(k_arr, eta)
			X, C  = hopfield_coefficients(simple_model, eta, k_arr, E_lp)
			assert np.abs(X[0])**2 == pytest.approx(0.5, abs=1e-6)
			assert np.abs(C[0])**2 == pytest.approx(0.5, abs=1e-6)

	def test_output_shapes(self, simple_model, eta_grid):
		k_arr = simple_model.q_grid_nat[1:7]
		E_lp  = simple_model.E_LP(k_arr, 0.0)
		X, C  = hopfield_coefficients(simple_model, 0.0, k_arr, E_lp)
		assert X.shape == k_arr.shape
		assert C.shape == k_arr.shape


# ---------------------------------------------------------------------------
# Fugacity series coefficients
# ---------------------------------------------------------------------------

class TestFugacitySeriesCoefficients:
	def test_output_shape(self):
		L_terms = 5
		ks      = np.linspace(0.0, 1.0, 10)
		dE      = np.zeros(10)
		a = _fugacity_series_coefficients(dE, ks, beta=1.0, L_terms=L_terms)
		assert a.shape == (L_terms,)

	def test_flat_dispersion_equal_coefficients(self):
		"""
		With dE=0, exp(-l*beta*dE)=1 for all l, so every a_l is the
		same number: integral(k dk, 0 to k_max) / (2*pi).
		"""
		L_terms = 4
		ks      = np.linspace(0.0, 1.0, 500)
		dE      = np.zeros(500)
		a       = _fugacity_series_coefficients(dE, ks, beta=1.0, L_terms=L_terms)
		# All coefficients should be equal
		np.testing.assert_allclose(a, a[0], rtol=1e-6)
		# Value: integral k dk from 0 to 1 = 0.5, divided by 2*pi
		expected = 0.5 / (2.0 * np.pi)
		assert a[0] == pytest.approx(expected, rel=1e-3)

	def test_coefficients_decreasing_for_positive_dE(self):
		"""
		For dE > 0, exp(-l*beta*dE) shrinks as l increases, so a_l
		should be a decreasing sequence.
		"""
		L_terms = 6
		ks      = np.linspace(0.0, 1.0, 50)
		dE      = np.ones(50) * 0.1   # positive uniform gap
		a       = _fugacity_series_coefficients(dE, ks, beta=1.0, L_terms=L_terms)
		assert np.all(np.diff(a) < 0)

	def test_non_negative(self):
		ks  = np.linspace(0.0, 2.0, 20)
		dE  = np.linspace(0.0, 0.5, 20)
		a   = _fugacity_series_coefficients(dE, ks, beta=1.0, L_terms=3)
		assert np.all(a >= 0.0)


# ---------------------------------------------------------------------------
# Effective mass
# ---------------------------------------------------------------------------

class TestEffectiveMass:
	def test_positive(self, simple_model, eta_grid):
		"""LP effective mass at k=0 must be positive."""
		k_grid = simple_model.q_grid_nat[:10]
		for eta in eta_grid:
			m_eff = effective_mass(simple_model, eta, k_grid)
			assert np.real(m_eff) > 0, f"Negative effective mass for eta={eta}"

	def test_returns_scalar(self, simple_model):
		k_grid = simple_model.q_grid_nat[:10]
		m_eff  = effective_mass(simple_model, 0.0, k_grid)
		assert np.ndim(m_eff) == 0

	def test_stable_under_grid_refinement(self, simple_model):
		"""The local fit should not depend strongly on the sampled near-zero grid."""
		coarse = np.linspace(0.0, 1.0, 50)
		dense  = np.linspace(0.0, 1.0, 200)
		m_coarse = effective_mass(simple_model, 0.0, coarse)
		m_dense  = effective_mass(simple_model, 0.0, dense)
		assert m_dense == pytest.approx(m_coarse, rel=0.01)


# ---------------------------------------------------------------------------
# Bare polariton interaction strength
# ---------------------------------------------------------------------------

class TestPolaritonInteractionStrengthBare:
	def test_positive(self, simple_model, eta_grid):
		"""Bare interaction g = (g_ex/N_qw)*|X_LP|^4 is always positive."""
		k_arr = simple_model.q_grid_nat[1:6]
		for eta in eta_grid:
			g, _ = polariton_interaction_strength(simple_model, eta, k_arr, bubble=0.0)
			assert np.all(g > 0), f"Negative bare interaction for eta={eta}"

	def test_bounded_by_g_ex(self, simple_model):
		"""g_bare <= g_ex / N_qw since |X_LP|^4 <= 1."""
		k_arr    = simple_model.q_grid_nat[1:6]
		g_upper  = simple_model.p.g_ex / simple_model.p.N_qw
		g, _ = polariton_interaction_strength(simple_model, 0.0, k_arr, bubble=0.0)
		assert np.all(g <= g_upper + 1e-30)

	def test_output_shape(self, simple_model):
		k_arr = simple_model.q_grid_nat[1:6]
		g_bare, g_screened = polariton_interaction_strength(simple_model, 0.0, k_arr, bubble=0.0)
		assert g_bare.shape == k_arr.shape
		assert g_screened.shape == k_arr.shape

	def test_resonance_value(self, simple_model):
		"""
		At k=0 (resonance), |X_LP|^4 = (1/√2)^4 = 1/4.
		So g_bare(k=0) = g_ex / (4 * N_qw).
		"""
		k_arr    = simple_model.q_grid_nat[:1]
		g, _ = polariton_interaction_strength(simple_model, 0.0, k_arr, bubble=0.0)
		expected = simple_model.p.g_ex / (4.0 * simple_model.p.N_qw)
		assert g[0] == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Slow tests (chemical_potential, Pi0, screened interaction)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestChemicalPotential:
	def test_negative(self, simple_model):
		"""Chemical potential must be below the LP band minimum (mu < E_LP_min)."""
		from polaritons.many_body import chemical_potential
		mu = chemical_potential(
			simple_model, eta=0.0,
			L_terms=100, k_upper=1.0, n_k=1_000,
		)
		assert mu < float(np.real(simple_model.E_LP(np.array([0.0]), 0.0)[0]))

	def test_returns_float(self, simple_model):
		from polaritons.many_body import chemical_potential
		mu = chemical_potential(
			simple_model, eta=0.0,
			L_terms=100, k_upper=1.0, n_k=1_000,
		)
		assert isinstance(mu, float)


@pytest.mark.slow
class TestScreenedInteraction:
	def test_screened_less_than_bare(self, simple_model, params_nat):
		"""Screening reduces interaction: g_screened < g_bare."""
		from polaritons.many_body import polariton_interaction_strength
		k_arr = simple_model.q_grid_nat[1:4]
		g_bare, g_screened = polariton_interaction_strength(
			simple_model, 0.0, k_arr,
			L_terms=100, k_upper=1.0, n_k=1_000,
			k_upper_mass=1e-3, n_k_mass=16,
		)
		assert np.all(g_screened < g_bare)


# ---------------------------------------------------------------------------
# Chemical potential smoke test (fast: small grid, few terms)
# ---------------------------------------------------------------------------

class TestChemicalPotentialSmoke:
	def test_mu_negative_for_bose_gas(self, simple_model):
		"""For a non-degenerate Bose gas, the chemical potential is negative."""
		from polaritons.many_body import chemical_potential
		# Reduce concentration so few series terms suffice on this small grid.
		small_p = simple_model.p
		# build a fresh model with the same p but smaller target density via
		# explicitly passing override args; chemical_potential picks p.concentration,
		# so monkey-patch a copy of the dataclass.
		from dataclasses import replace
		p_low = replace(small_p, concentration=1e-6)
		# DispersionModel re-uses p as-is; create a tiny shim by replacing p attr.
		simple_model.p = p_low
		try:
			mu = chemical_potential(
				simple_model, eta=0.0, L_terms=20, k_upper=0.5, n_k=200,
			)
		finally:
			simple_model.p = small_p
		assert mu < 0.0
