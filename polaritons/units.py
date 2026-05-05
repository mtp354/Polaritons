"""
Unit conversions between natural units (E_bind, a) and SI-derived units.

Pass a Params object that has been converted via Params.to_natural() so that
``p_nat.E_unit`` (eV per natural energy) and ``p_nat.L_unit`` (m per natural
length) carry the correct scale factors.

All functions accept either a scalar or array input and return the same
shape; scalar inputs come back as Python floats.
"""

from __future__ import annotations
import numpy as np
from .parameters import Params


CM_INV_PER_M_INV = 1e-2   # k[cm^-1] = 1e-2 * k[m^-1]


def _as_float_or_array(values: np.ndarray):
	values = np.asarray(values)
	return float(values) if values.ndim == 0 else values


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def k_nat_to_cm(k_nat, p_nat: Params):
	"""Momentum: natural units (1/a) -> cm^-1."""
	return _as_float_or_array(
		CM_INV_PER_M_INV * np.asarray(k_nat, dtype=float) / p_nat.L_unit
	)


def k_cm_to_nat(k_cm, p_nat: Params):
	"""Momentum: cm^-1 -> natural units (1/a)."""
	return _as_float_or_array(
		np.asarray(k_cm, dtype=float) * p_nat.L_unit / CM_INV_PER_M_INV
	)


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def energy_nat_to_eV(E_nat, p_nat: Params):
	"""Energy: natural units (E_bind) -> eV."""
	return np.asarray(E_nat) * p_nat.E_unit


def energy_nat_to_meV(E_nat, p_nat: Params):
	return 1e3 * energy_nat_to_eV(E_nat, p_nat)


def energy_nat_to_microeV(E_nat, p_nat: Params):
	return 1e6 * energy_nat_to_eV(E_nat, p_nat)


def energy_nat_to_neV(E_nat, p_nat: Params):
	return 1e9 * energy_nat_to_eV(E_nat, p_nat)


# ---------------------------------------------------------------------------
# Interaction strength (energy x area)
# ---------------------------------------------------------------------------

def interaction_nat_to_microev_um2(g_nat, p_nat: Params):
	"""Interaction g: natural units (E_bind * a^2) -> micro-eV * um^2."""
	return 1e18 * np.asarray(g_nat) * p_nat.E_unit * p_nat.L_unit**2
