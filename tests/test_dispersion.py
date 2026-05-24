"""
Unit tests for polaritons.dispersion (DispersionModel).

All tests use the ``simple_model`` fixture (Q=0 everywhere) defined in
conftest.py.  Public model inputs and outputs are natural-unit quantities;
plotting code handles conversion to SI-derived units.
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
			q_val = simple_model.q_grid_nat[5]   # arbitrary interior point
			result = simple_model.Q(q_val, eta)
			assert abs(result) == pytest.approx(0.0, abs=1e-20)

	def test_Q_output_scalar_for_scalar_input(self, simple_model):
		result = simple_model.Q(0.0, 0.0)
		assert np.ndim(result) == 0

	def test_Q_output_1d_for_array_input(self, simple_model):
		k_arr = simple_model.q_grid_nat[:5]
		result = simple_model.Q(k_arr, 0.0)
		assert result.shape == (5,)

	def test_Q_output_1d_for_singleton_array_input(self, simple_model):
		k_arr = simple_model.q_grid_nat[:1]
		result = simple_model.Q(k_arr, 0.0)
		assert result.shape == (1,)

	def test_Q_returns_natural_energy(self, params_nat, q_grid_nat, eta_grid):
		Q_nat = np.ones((len(eta_grid), len(q_grid_nat)), dtype=complex) * (2.0 + 3.0j)
		model = DispersionModel(params_nat, q_grid_nat, eta_grid, Q_nat)
		result = model.Q(0.0, eta_grid[2])
		assert result == pytest.approx(2.0 + 3.0j)

	def test_Q_uses_natural_momentum_directly(self, params_nat, q_grid_nat, eta_grid):
		Q_nat = np.broadcast_to(q_grid_nat, (len(eta_grid), len(q_grid_nat))).astype(complex)
		model = DispersionModel(params_nat, q_grid_nat, eta_grid, Q_nat)
		idx = 5
		result = model.Q(q_grid_nat[idx], eta_grid[1])
		assert result == pytest.approx(q_grid_nat[idx])

	def test_Q_selects_fixed_eta_slice(self, params_nat, q_grid_nat, eta_grid):
		Q_nat = np.empty((len(eta_grid), len(q_grid_nat)), dtype=complex)
		for i, eta in enumerate(eta_grid):
			Q_nat[i] = eta + q_grid_nat
		model = DispersionModel(params_nat, q_grid_nat, eta_grid, Q_nat)

		result = model.Q(q_grid_nat[4], eta_grid[2])
		assert result == pytest.approx(eta_grid[2] + q_grid_nat[4])

	def test_Q_rejects_unsolved_eta(self, params_nat, q_grid_nat, eta_grid):
		Q_nat = np.zeros((len(eta_grid), len(q_grid_nat)), dtype=complex)
		model = DispersionModel(params_nat, q_grid_nat, eta_grid, Q_nat)

		with pytest.raises(ValueError, match="only interpolated over k"):
			model.Q(q_grid_nat[4], 0.25)


class TestEEx:
	def test_bare_exciton_at_k0(self, simple_model, params_nat, eta_grid):
		"""With Q=0 and the band-bottom convention, E_ex(0, eta) = 0."""
		for eta in eta_grid:
			val = float(np.real(simple_model.E_ex(0.0, eta)))
			assert val == pytest.approx(0.0, abs=1e-12)

	def test_kinetic_energy_increases_with_k(self, simple_model):
		"""E_ex(k) should be larger for larger k."""
		k_vals  = simple_model.q_grid_nat[1:6]
		E_vals  = np.real(simple_model.E_ex(k_vals, 0.0))
		assert np.all(np.diff(E_vals) > 0)

	def test_output_shape(self, simple_model):
		k_arr = simple_model.q_grid_nat[:8]
		result = simple_model.E_ex(k_arr, 0.0)
		assert result.shape == (8,)


class TestEPh:
	def test_tuned_cavity_at_k0(self, simple_model, eta_grid):
		"""E_ph(0, eta) must equal E_ex(0, eta) by construction (complex)."""
		for eta in eta_grid:
			E_ex_0 = complex(simple_model.E_ex(0.0, eta))
			E_ph_0 = complex(simple_model.E_ph(0.0, eta))
			assert E_ph_0.real == pytest.approx(E_ex_0.real, rel=1e-6, abs=1e-12)
			assert E_ph_0.imag == pytest.approx(E_ex_0.imag, rel=1e-6, abs=1e-12)

	def test_E_ph_increases_with_k(self, simple_model):
		"""Photon dispersion is dispersive (Re part increases with |k|)."""
		k_vals  = simple_model.q_grid_nat[1:6]
		E_vals  = np.real(simple_model.E_ph(k_vals, 0.0))
		assert np.all(np.diff(E_vals) > 0)

	def test_E_ph_untuned_at_k0(self, simple_model):
		"""E_ph_untuned(0) = E_ex(0, eta=0) regardless of eta."""
		E_ex_0_eta0 = _e0(simple_model, 0.0)
		E_ph_untuned_0 = float(np.real(simple_model.E_ph_untuned(0.0)))
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

	def test_rabi_splitting_at_k0(self, simple_model, params_nat, eta_grid):
		"""
		At resonance the LP-UP gap equals the Rabi splitting Omega, so the
		off-diagonal coupling in the 2x2 block is Omega/2 and
		E_LP(0) = E_0 - Omega/2.
		"""
		for eta in eta_grid:
			E_0  = _e0(simple_model, eta)
			E_lp = float(np.real(simple_model.E_LP(np.array([0.0]), eta)[0]))
			assert E_lp == pytest.approx(E_0 - 0.5 * params_nat.Omega, rel=1e-5)

	def test_disorder_tuned_vs_untuned_differ(self, params_nat, q_grid_nat, eta_grid):
		"""
		With a non-zero self-energy the disorder-tuned and untuned dispersions
		must differ, because E_ex(0, eta) != E_ex(0, 0) when Q != 0.
		"""
		# Build a model where Q has a small non-zero imaginary part
		# so that E_ex(0, eta>0) != E_ex(0, 0).
		Q_nonzero = np.zeros((len(eta_grid), len(q_grid_nat)), dtype=complex)
		# Shift the self-energy linearly with eta index so the cavity
		# reference energy varies between disorder-tuned and untuned.
		# Use a shift of 10.0 per unit of eta so the energy difference between
		# tuned and untuned is clearly larger than allclose's tolerance.
		for i, eta in enumerate(eta_grid):
			Q_nonzero[i] = eta * 10.0
		model = DispersionModel(params_nat, q_grid_nat, eta_grid, Q_nonzero)

		k_arr = model.q_grid_nat[:5]
		eta   = eta_grid[-1]   # largest disorder value
		E_tuned   = model.E_LP(k_arr, eta, disorder_tuned=True)
		E_untuned = model.E_LP(k_arr, eta, disorder_tuned=False)
		assert not np.allclose(E_tuned, E_untuned)

	def test_E_LP_output_shape(self, simple_model):
		k_arr = simple_model.q_grid_nat[:8]
		result = simple_model.E_LP(k_arr, 0.0)
		assert result.shape == (8,)

	def test_E_LP_scalar_input(self, simple_model):
		result = simple_model.E_LP(np.float64(0.0), 0.0)
		assert np.ndim(result) == 0

	def test_q_grid_shape(self, simple_model, q_grid_nat):
		assert len(simple_model.q_grid_nat) == len(q_grid_nat)


class TestDispersionModelConstruction:
	def test_eta_grid_stored(self, simple_model, eta_grid):
		np.testing.assert_array_equal(simple_model.eta_grid, eta_grid)

	def test_q_grid_stored_in_natural_units(self, simple_model, q_grid_nat):
		np.testing.assert_allclose(simple_model.q_grid_nat, q_grid_nat)

	def test_mismatched_Q_raises(self, params_nat, q_grid_nat, eta_grid):
		"""Providing Q_results with wrong shape should raise at construction."""
		bad_Q = np.zeros((len(eta_grid), len(q_grid_nat) + 1), dtype=complex)
		with pytest.raises(ValueError, match="Q_results must have shape"):
			DispersionModel(params_nat, q_grid_nat, eta_grid, bad_Q)

	def test_si_params_raise(self, params_si, q_grid_nat, eta_grid, Q_results_zero):
		with pytest.raises(ValueError, match="natural units"):
			DispersionModel(params_si, q_grid_nat, eta_grid, Q_results_zero)


class TestELPAnalyticAgreesWithEigvals:
	"""
	Regression test: closed-form LP root must match np.linalg.eigvals on the
	2x2 [[Eex, Omega/2],[Omega/2, Eph]] block element-wise.
	"""

	def test_matches_eigvals_real_Q(self, params_nat, q_grid_nat, eta_grid):
		# Non-zero, real Q so that eigenvalues are real.
		rng = np.random.default_rng(0)
		Q_real = rng.normal(size=(len(eta_grid), len(q_grid_nat)))
		model = DispersionModel(params_nat, q_grid_nat, eta_grid, Q_real.astype(complex))

		k_arr = model.q_grid_nat[1:]
		for eta in eta_grid:
			Eex = np.asarray(model.E_ex(k_arr, eta), dtype=complex)
			Eph = np.asarray(model.E_ph(k_arr, eta), dtype=complex)
			g = 0.5 * model.p.Omega
			expected = np.empty_like(Eex)
			for i in range(len(k_arr)):
				vals = np.linalg.eigvals(
					np.array([[Eex[i], g], [g, Eph[i]]], dtype=complex)
				)
				expected[i] = vals[np.argmin(vals.real)]
			got = model.E_LP(k_arr, eta)
			np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

	def test_matches_eigvals_complex_Q(self, params_nat, q_grid_nat, eta_grid):
		# Complex Q -> complex Eex; LP is the eigenvalue with the lower real
		# part (under-damped) or the more-negative imaginary part (over-damped).
		rng = np.random.default_rng(1)
		Q = (rng.normal(size=(len(eta_grid), len(q_grid_nat)))
		     + 1j * rng.normal(size=(len(eta_grid), len(q_grid_nat))))
		model = DispersionModel(params_nat, q_grid_nat, eta_grid, Q.astype(complex))

		k_arr = model.q_grid_nat[1:]
		for eta in eta_grid:
			Eex = np.asarray(model.E_ex(k_arr, eta), dtype=complex)
			Eph = np.asarray(model.E_ph(k_arr, eta), dtype=complex)
			g = 0.5 * model.p.Omega
			got = model.E_LP(k_arr, eta)
			for i in range(len(k_arr)):
				vals = np.linalg.eigvals(
					np.array([[Eex[i], g], [g, Eph[i]]], dtype=complex)
				)
				# E_LP must match one of the two eigenvalues to high precision.
				diffs = np.abs(vals - got[i])
				assert diffs.min() < 1e-10, (vals, got[i])
				# And it must be the lower-real-part root (with a small
				# imaginary tie-break: more-negative imag for the LP).
				other = vals[1 - int(np.argmin(diffs))]
				if abs(other.real - got[i].real) > 1e-10:
					assert got[i].real <= other.real + 1e-12
				else:
					assert got[i].imag <= other.imag + 1e-12

	def test_complex_Q_branch_continuous_through_zero(self, params_nat, eta_grid):
		# Regression: at k=0 with complex Eex but real Eph (tuned cavity),
		# the discriminant^2 sits on the sqrt branch cut. Floating-point noise
		# previously flipped the LP root onto the UP branch at k=0 only.
		q_grid = np.linspace(0.0, 1.0, 32)
		# Q with a large imaginary part (over-damped regime).
		Q = np.zeros((len(eta_grid), len(q_grid)), dtype=complex)
		for ei, eta in enumerate(eta_grid):
			if eta == 0.0:
				continue
			Q[ei, :] = (1.0 + 0.0 * q_grid) + 1j * (10.0 + 0.0 * q_grid)
		model = DispersionModel(params_nat, q_grid, eta_grid, Q)
		ks = np.array([0.0, 1e-12, 1e-10, 1e-8])
		for eta in eta_grid:
			if eta == 0.0:
				continue
			Elp = model.E_LP(ks, eta, disorder_tuned=True)
			# LP must be continuous across k=0: every point close to k=0 value.
			np.testing.assert_allclose(Elp.real, Elp[0].real, atol=1e-6)
			np.testing.assert_allclose(Elp.imag, Elp[0].imag, atol=1e-6)
