"""
Shared pytest fixtures for the polaritons test suite.
"""
import numpy as np
import pytest

from polaritons.parameters import Params, DEFAULT_PARAMS
from polaritons.dispersion import DispersionModel


@pytest.fixture(scope="session")
def params_si():
	"""Default Params in SI units."""
	return Params()


@pytest.fixture(scope="session")
def M_lp_si(params_si):
	"""A representative LP effective mass in SI units (eV*s^2/m^2).

	Used by real-space tests that need *some* mass to build a Hamiltonian;
	the value (7.8e-5 * m_rest, PRB 77 155317) is illustrative only.
	"""
	return 7.8e-5 * params_si.m_rest


@pytest.fixture(scope="session")
def params_nat(params_si):
	"""Default Params converted to natural units (E_bind=1, a=1)."""
	return params_si.to_natural()


@pytest.fixture(scope="session")
def eta_grid():
	"""Five disorder values used to build DispersionModel fixture.

	DispersionModel builds one momentum spline per eta value.
	"""
	return np.array([0.0, 0.5, 1.0, 1.5, 2.0])


@pytest.fixture(scope="session")
def q_grid_nat():
	"""Twenty-point natural-unit momentum grid [0, 50]."""
	return np.linspace(0.0, 50.0, 20)


@pytest.fixture(scope="session")
def Q_results_zero(eta_grid, q_grid_nat):
	"""Q_results with no self-energy (all zeros)."""
	return np.zeros((len(eta_grid), len(q_grid_nat)), dtype=complex)


@pytest.fixture(scope="session")
def simple_model(params_nat, q_grid_nat, eta_grid, Q_results_zero):
	"""
	Minimal DispersionModel with zero self-energy.

	Uses natural-unit Params, natural Picard momentum, and natural Q values.

	With Q=0 everywhere:
	  E_ex(k, eta) = p.hbar^2 * k^2 / (2*p.M)   (band-bottom-relative)
	  E_ph(0, eta) == E_ex(0, eta)  by tuned-cavity construction
	"""
	return DispersionModel(params_nat, q_grid_nat, eta_grid, Q_results_zero)
