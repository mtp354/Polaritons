"""
Unit tests for polaritons.units.
"""
import numpy as np
import pytest

from polaritons.parameters import Params
from polaritons.units import (
	k_nat_to_cm, k_cm_to_nat,
	energy_nat_to_eV, energy_nat_to_meV, energy_nat_to_microeV, energy_nat_to_neV,
	interaction_nat_to_microev_um2,
)


@pytest.fixture(scope="module")
def p_nat():
	return Params().to_natural()


class TestMomentumConversion:
	def test_round_trip_array(self, p_nat):
		k_nat = np.linspace(0.0, 50.0, 11)
		assert np.allclose(k_cm_to_nat(k_nat_to_cm(k_nat, p_nat), p_nat), k_nat)

	def test_round_trip_scalar(self, p_nat):
		assert k_cm_to_nat(k_nat_to_cm(3.14, p_nat), p_nat) == pytest.approx(3.14)

	def test_scalar_returns_float(self, p_nat):
		assert isinstance(k_nat_to_cm(1.0, p_nat), float)
		assert isinstance(k_cm_to_nat(1.0, p_nat), float)

	def test_array_returns_array(self, p_nat):
		out = k_nat_to_cm(np.array([1.0, 2.0]), p_nat)
		assert isinstance(out, np.ndarray)
		assert out.shape == (2,)

	def test_zero_maps_to_zero(self, p_nat):
		assert k_nat_to_cm(0.0, p_nat) == 0.0
		assert k_cm_to_nat(0.0, p_nat) == 0.0


class TestEnergyConversion:
	def test_eV_uses_E_unit(self, p_nat):
		# 1 in natural energy units = E_bind in eV.
		assert float(energy_nat_to_eV(1.0, p_nat)) == pytest.approx(p_nat.E_unit)

	def test_meV_is_1000x_eV(self, p_nat):
		x = np.array([0.5, 1.0, 2.0])
		assert np.allclose(energy_nat_to_meV(x, p_nat), 1e3 * energy_nat_to_eV(x, p_nat))

	def test_microeV_is_1e6x_eV(self, p_nat):
		x = 1.5
		assert energy_nat_to_microeV(x, p_nat) == pytest.approx(1e6 * energy_nat_to_eV(x, p_nat))

	def test_neV_is_1e9x_eV(self, p_nat):
		x = np.array([0.1])
		assert np.allclose(energy_nat_to_neV(x, p_nat), 1e9 * energy_nat_to_eV(x, p_nat))


class TestInteractionConversion:
	def test_units_factor(self, p_nat):
		# 1 natural g = E_unit eV * L_unit^2 m^2  -> *1e18 -> microeV * um^2
		got = float(interaction_nat_to_microev_um2(1.0, p_nat))
		assert got == pytest.approx(1e18 * p_nat.E_unit * p_nat.L_unit**2)

	def test_array_shape_preserved(self, p_nat):
		x = np.zeros(7)
		assert interaction_nat_to_microev_um2(x, p_nat).shape == (7,)
