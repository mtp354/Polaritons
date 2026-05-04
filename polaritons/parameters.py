"""
Physical parameters and natural-unit conversion for GaAs polariton calculations.

Natural units
-------------
Energy : E_bind  (exciton binding energy, ~4.2 meV)
Length : a       (exciton Bohr radius, derived from m_e, m_h, E_bind)

All functions and modules in this package expect quantities in these natural
units unless explicitly stated.  Use Params.to_natural() after construction.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np


# ---------------------------------------------------------------------------
# Fundamental constants (SI / eV-based)
# ---------------------------------------------------------------------------

_C_SI        = 3e8          # speed of light            m/s
_HBAR_SI     = 6.58e-16     # reduced Planck constant   eV·s
_KB_SI       = 8.617333262e-5  # Boltzmann constant     eV/K

# Masses from PhysRevB.44 (eV·s²/m²)
_M_REST_SI   = 5.68e-12
_M_E_SI      = 3.81e-13     # electron effective mass
_M_H_SI      = 2.56e-12     # hole effective mass
_M_EFF_SI    = 7.8e-5 * _M_REST_SI  # effective polariton mass (PRB 77 155317)


@dataclass
class Params:
	"""
	All physical parameters for a single calculation run.

	SI / eV values are stored here; call `to_natural()` to get a new Params
	where every quantity has been rescaled to natural units.

	Parameters
	----------
	Material
	~~~~~~~~
	E_bind          : exciton binding energy                     (eV)
	E_gap_bare      : bare band gap (without binding correction) (eV)
	m_e, m_h        : electron/hole effective masses             (eV·s²/m²)
	m_rest          : free-electron rest mass                    (eV·s²/m²)

	Polariton / cavity
	~~~~~~~~~~~~~~~~~~
	Omega           : Rabi splitting                             (eV)
	m_prime         : band-structure constant (dimensionless)
	n_refr          : refractive index of the cavity medium
	N_qw            : number of quantum wells
	M_eff           : effective polariton mass used by real-space routines

	Disorder
	~~~~~~~~
	D_0             : disorder strength                          (eV²·m²)
	xi              : disorder correlation length                (m)

	Thermodynamic
	~~~~~~~~~~~~~
	T               : temperature                                (K)
	concentration   : exciton areal density                     (m⁻²)
	g_ex            : bare exciton-exciton interaction          (eV·m²)

	Derived (set automatically by to_natural())
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	in_natural_units : flag so code can assert the right unit system
	E_unit, L_unit   : conversion factors stored for reference
	"""

	# ---- material ----------------------------------------------------------
	E_bind      : float = 4.2e-3           # eV
	E_gap_bare  : float = 1.6              # eV  (E_gap = E_gap_bare + E_bind)
	m_e         : float = _M_E_SI
	m_h         : float = _M_H_SI
	m_rest      : float = _M_REST_SI

	# ---- polariton / cavity ------------------------------------------------
	Omega       : float = 1.4e-2           # eV  Rabi splitting
	m_prime     : float = 1.0             # dimensionless band-structure constant
	n_refr      : float = 3.0             # cavity refractive index
	N_qw        : int   = 1               # number of quantum wells

	# ---- disorder ----------------------------------------------------------
	D_0         : float = 2.26e-20         # eV²·m²
	xi          : float = 20e-9            # m

	# ---- thermodynamic -----------------------------------------------------
	T           : float = 20.0             # K
	concentration: float = 0.3e12          # m⁻²
	g_ex        : float = 12e-18           # eV·m²

	# ---- unit system flags (populated by to_natural) -----------------------
	in_natural_units: bool = False
	E_unit      : float = field(default=1.0, repr=False)
	L_unit      : float = field(default=1.0, repr=False)

	# ---- convenience constants --------------------------------------------
	c     : float = field(default=_C_SI,    repr=False)
	hbar  : float = field(default=_HBAR_SI, repr=False)
	k_B   : float = field(default=_KB_SI,   repr=False)
	M_eff : float = field(default=_M_EFF_SI,repr=False)

	# ------------------------------------------------------------------
	# Derived quantities
	# ------------------------------------------------------------------

	@property
	def E_gap(self) -> float:
		"""Full band gap = bare gap + binding energy."""
		return self.E_gap_bare + self.E_bind

	@property
	def M(self) -> float:
		"""Total exciton mass m_e + m_h."""
		return self.m_e + self.m_h

	@property
	def mu_mass(self) -> float:
		"""Reduced mass m_e*m_h / (m_e+m_h)."""
		return self.m_e * self.m_h / (self.m_e + self.m_h)

	@property
	def a(self) -> float:
		"""Exciton Bohr radius  ħ / sqrt(2 * mu * E_bind)."""
		return self.hbar / np.sqrt(2.0 * self.mu_mass * self.E_bind)

	@property
	def beta(self) -> float:
		"""Inverse thermal energy  1 / (k_B * T)."""
		return 1.0 / (self.k_B * self.T)

	@property
	def alpha_e(self) -> float:
		return self.m_prime * self.m_rest / self.m_e

	@property
	def alpha_h(self) -> float:
		return self.m_prime * self.m_rest / self.m_h

	# ------------------------------------------------------------------
	# Unit conversion
	# ------------------------------------------------------------------

	def to_natural(self) -> "Params":
		"""
		Return a new Params with all quantities rescaled to natural units.

		Energy unit  : E_bind  → 1
		Length unit  : a       → 1
		"""
		if self.in_natural_units:
			return self  # already converted

		E_u = self.E_bind          # energy unit (eV)
		L_u = self.a               # length unit (m)

		p = Params(
			# material
			E_bind      = self.E_bind  / E_u,          # = 1
			E_gap_bare  = self.E_gap_bare / E_u,
			m_e         = self.m_e   * L_u**2 / E_u,
			m_h         = self.m_h   * L_u**2 / E_u,
			m_rest      = self.m_rest* L_u**2 / E_u,
			# polariton / cavity
			Omega       = self.Omega  / E_u,
			m_prime     = self.m_prime,
			n_refr      = self.n_refr,
			N_qw        = self.N_qw,
			# disorder
			D_0         = self.D_0 / (E_u**2 * L_u**2),
			xi          = self.xi   / L_u,
			# thermodynamic
			T           = self.T,
			concentration = self.concentration * L_u**2,
			g_ex        = self.g_ex / (E_u * L_u**2),
			# flags
			in_natural_units = True,
			E_unit      = E_u,
			L_unit      = L_u,
			# fundamental constants (rescaled)
			c     = self.c    / L_u,          # now in natural length/time
			hbar  = self.hbar / E_u,
			k_B   = self.k_B  / E_u,
			M_eff = self.M_eff * L_u**2 / E_u,
		)
		return p

	def to_dict(self) -> dict:
		"""Serialisable dictionary (for JSON metadata)."""
		d = asdict(self)
		# include derived scalar quantities
		d["E_gap"]  = float(self.E_gap)
		d["M"]      = float(self.M)
		d["a"]      = float(self.a)
		d["beta"]   = float(self.beta)
		return d


# ---------------------------------------------------------------------------
# Default parameter set  (SI)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = Params()
