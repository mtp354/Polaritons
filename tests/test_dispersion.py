"""
Unit tests for polaritons.dispersion (DispersionModel).

All tests use the ``simple_model`` fixture (Q=0 everywhere) defined in
conftest.py.  Public model inputs and outputs are SI quantities; the
saved Picard grid and Q arrays are natural-unit quantities.
"""
import numpy as np
import pytest

from polaritons.dispersion import DispersionModel
from polaritons.parameters import Params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _e0(model, eta):
	"""Real part of E_ex at k=0 for given eta."""
	return float(np.real(model.E_ex(0.0, eta)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQInterpolation:
	def test_Q_zero_everywhere(self, simple_model, eta_grid):
		"""With Q_results=0 the self-energy should be zero everywhere."""
		for eta in eta_grid:
			q_val = simple_model.q_grid_si[5]   # arbitrary interior point
			result = simple_model.Q(q_val, eta)
			assert abs(result) == pytest.approx(0.0, abs=1e-20)

	def test_Q_output_scalar_for_scalar_input(self, simple_model):
		result = simple_model.Q(0.0, 0.0)
		assert np.ndim(result) == 0

	def test_Q_output_1d_for_array_input(self, simple_model):
		k_arr = simple_model.q_grid_si[:5]
		result = simple_model.Q(k_arr, 0.0)
		assert result.shape == (5,)

	def test_Q_energy_scaled_by_E_bind(self, params_si, q_grid_nat, eta_grid):
		Q_nat = np.ones((len(eta_grid), len(q_grid_nat)), dtype=complex) * (2.0 + 3.0j)
		model = DispersionModel(params_si, q_grid_nat, eta_grid, Q_nat)
		result = model.Q(0.0, eta_grid[2])
		assert result == pytest.approx(params_si.E_bind * (2.0 + 3.0j))

	def test_Q_momentum_scaled_by_bohr_radius(self, params_si, q_grid_nat, eta_grid):
		Q_nat = np.broadcast_to(q_grid_nat, (len(eta_grid), len(q_grid_nat))).astype(complex)
		model = DispersionModel(params_si, q_grid_nat, eta_grid, Q_nat)
		idx = 5
		k_si = q_grid_nat[idx] / params_si.a
		result = model.Q(k_si, eta_grid[1])
		assert result == pytest.approx(params_si.E_bind * q_grid_nat[idx])


class TestEEx:
	def test_bare_exciton_at_k0(self, simple_model, params_si, eta_grid):
		"""With Q=0, E_ex(0, eta) = E_gap - E_bind (bare exciton energy)."""
		expected = params_si.E_gap - params_si.E_bind
		for eta in eta_grid:
			val = float(np.real(simple_model.E_ex(0.0, eta)))
			assert val == pytest.approx(expected, rel=1e-6)

	def test_kinetic_energy_increases_with_k(self, simple_model):
		"""E_ex(k) should be larger for larger k."""
		k_vals  = simple_model.q_grid_si[1:6]
		E_vals  = np.real(simple_model.E_ex(k_vals, 0.0))
		assert np.all(np.diff(E_vals) > 0)

	def test_output_shape(self, simple_model):
		k_arr = simple_model.q_grid_si[:8]
		result = simple_model.E_ex(k_arr, 0.0)
		assert result.shape == (8,)


class TestEPh:
	def test_tuned_cavity_at_k0(self, simple_model, eta_grid):
		"""E_ph(0, eta) must equal E_ex(0, eta) by construction."""
		for eta in eta_grid:
			E_ex_0 = _e0(simple_model, eta)
			E_ph_0 = float(simple_model.E_ph(0.0, eta))
			assert E_ph_0 == pytest.approx(E_ex_0, rel=1e-6)

	def test_E_ph_increases_with_k(self, simple_model):
		"""Photon dispersion is dispersive (increases with |k|)."""
		k_vals  = simple_model.q_grid_si[1:6]
		E_vals  = simple_model.E_ph(k_vals, 0.0)
		assert np.all(np.diff(E_vals) > 0)

	def test_E_ph_untuned_at_k0(self, simple_model):
		"""E_ph_untuned(0) = E_ex(0, eta=0) regardless of eta."""
		E_ex_0_eta0 = _e0(simple_model, 0.0)
		E_ph_untuned_0 = float(simple_model.E_ph_untuned(0.0))
		assert E_ph_untuned_0 == pytest.approx(E_ex_0_eta0, rel=1e-6)


class TestELP:
	def test_LP_below_both_branches_at_resonance(self, simple_model, eta_grid):
		"""
		At k=0 the cavity is tuned to resonance (E_ex == E_ph = E_0).
		The lower polariton eigenvalue is E_0 - Omega.
		"""
		p = simple_model.p
		for eta in eta_grid:
			E_0   = _e0(simple_model, eta)
			E_lp  = float(np.real(simple_model.E_LP(np.array([0.0]), eta)[0]))
			# LP must be strictly below both branches
			assert E_lp < E_0

	def test_rabi_splitting_at_k0(self, simple_model, params_si, eta_grid):
		"""
		At resonance, E_LP(0) = E_0 - Omega (2×2 eigenproblem result).
		"""
		for eta in eta_grid:
			E_0  = _e0(simple_model, eta)
			E_lp = float(np.real(simple_model.E_LP(np.array([0.0]), eta)[0]))
			assert E_lp == pytest.approx(E_0 - params_si.Omega, rel=1e-5)

	def test_disorder_tuned_vs_untuned_differ(self, params_si, q_grid_nat, eta_grid):
		"""
		With a non-zero self-energy the disorder-tuned and untuned dispersions
		must differ, because E_ex(0, eta) != E_ex(0, 0) when Q != 0.
		"""
		# Build a model where Q has a small non-zero imaginary part
		# so that E_ex(0, eta>0) != E_ex(0, 0).
		Q_nonzero = np.zeros((len(eta_grid), len(q_grid_nat)), dtype=complex)
		# Shift the self-energy linearly with eta index so the cavity
		# reference energy varies between disorder-tuned and untuned.
		# Use a shift of 10.0 per unit of eta so that after the E_bind
		# re-scaling inside DispersionModel.Q() the energy difference
		# between tuned and untuned is clearly larger than allclose's tolerance.
		for i, eta in enumerate(eta_grid):
			Q_nonzero[i] = eta * 10.0
		model = DispersionModel(params_si, q_grid_nat, eta_grid, Q_nonzero)

		k_arr = model.q_grid_si[:5]
		eta   = eta_grid[-1]   # largest disorder value
		E_tuned   = model.E_LP(k_arr, eta, disorder_tuned=True)
		E_untuned = model.E_LP(k_arr, eta, disorder_tuned=False)
		assert not np.allclose(E_tuned, E_untuned)

	def test_E_LP_output_shape(self, simple_model):
		k_arr = simple_model.q_grid_si[:8]
		result = simple_model.E_LP(k_arr, 0.0)
		assert result.shape == (8,)

	def test_E_LP_scalar_input(self, simple_model):
		result = simple_model.E_LP(np.float64(0.0), 0.0)
		assert np.ndim(result) == 0

	def test_q_grid_si_shape(self, simple_model, q_grid_nat):
		assert len(simple_model.q_grid_si) == len(q_grid_nat)


class TestDispersionModelConstruction:
	def test_eta_grid_stored(self, simple_model, eta_grid):
		np.testing.assert_array_equal(simple_model.eta_grid, eta_grid)

	def test_q_grid_si_scaling(self, simple_model, params_si, q_grid_nat):
		"""q_grid_si = q_grid_nat / a."""
		expected = q_grid_nat / params_si.a
		np.testing.assert_allclose(simple_model.q_grid_si, expected)

	def test_mismatched_Q_raises(self, params_si, q_grid_nat, eta_grid):
		"""Providing Q_results with wrong shape should raise inside spline."""
		bad_Q = np.zeros((len(eta_grid), len(q_grid_nat) + 1), dtype=complex)
		with pytest.raises(Exception):
			DispersionModel(params_si, q_grid_nat, eta_grid, bad_Q)

	def test_natural_params_raise(self, params_nat, q_grid_nat, eta_grid, Q_results_zero):
		with pytest.raises(ValueError, match="SI units"):
			DispersionModel(params_nat, q_grid_nat, eta_grid, Q_results_zero)
