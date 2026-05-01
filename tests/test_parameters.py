"""
Unit tests for polaritons.parameters.
"""
import numpy as np
import pytest

from polaritons.parameters import Params, DEFAULT_PARAMS


class TestParamsDefaults:
    def test_construction_default(self):
        p = Params()
        assert p.E_bind == pytest.approx(4.2e-3)
        assert p.E_gap_bare == pytest.approx(1.6)
        assert not p.in_natural_units

    def test_default_params_singleton(self):
        assert isinstance(DEFAULT_PARAMS, Params)
        assert DEFAULT_PARAMS.E_bind == pytest.approx(4.2e-3)

    def test_custom_construction(self):
        p = Params(E_bind=5e-3, T=10.0)
        assert p.E_bind == pytest.approx(5e-3)
        assert p.T == pytest.approx(10.0)
        # Other fields should keep defaults
        assert p.E_gap_bare == pytest.approx(1.6)


class TestDerivedProperties:
    def test_E_gap(self):
        p = Params()
        assert p.E_gap == pytest.approx(p.E_gap_bare + p.E_bind)

    def test_M(self):
        p = Params()
        assert p.M == pytest.approx(p.m_e + p.m_h)

    def test_mu_mass(self):
        p = Params()
        expected = p.m_e * p.m_h / (p.m_e + p.m_h)
        assert p.mu_mass == pytest.approx(expected)

    def test_a_positive(self):
        p = Params()
        assert p.a > 0

    def test_a_formula(self):
        """a = hbar / sqrt(2 * mu * E_bind)."""
        p = Params()
        expected = p.hbar / np.sqrt(2.0 * p.mu_mass * p.E_bind)
        assert p.a == pytest.approx(expected)

    def test_beta_positive(self):
        p = Params()
        assert p.beta > 0

    def test_beta_formula(self):
        p = Params()
        assert p.beta == pytest.approx(1.0 / (p.k_B * p.T))

    def test_alpha_e(self):
        p = Params()
        assert p.alpha_e == pytest.approx(p.m_prime * p.m_rest / p.m_e)

    def test_alpha_h(self):
        p = Params()
        assert p.alpha_h == pytest.approx(p.m_prime * p.m_rest / p.m_h)


class TestToNatural:
    def test_returns_new_params(self):
        p_si = Params()
        p_nat = p_si.to_natural()
        assert p_nat is not p_si

    def test_in_natural_units_flag(self):
        p_nat = Params().to_natural()
        assert p_nat.in_natural_units

    def test_E_bind_is_one(self):
        p_nat = Params().to_natural()
        assert p_nat.E_bind == pytest.approx(1.0)

    def test_a_is_one(self):
        """In natural units the Bohr radius is 1."""
        p_nat = Params().to_natural()
        assert p_nat.a == pytest.approx(1.0, rel=1e-6)

    def test_E_unit_stored(self):
        p_si = Params()
        p_nat = p_si.to_natural()
        assert p_nat.E_unit == pytest.approx(p_si.E_bind)

    def test_L_unit_stored(self):
        p_si = Params()
        p_nat = p_si.to_natural()
        assert p_nat.L_unit == pytest.approx(p_si.a)

    def test_idempotent(self):
        """Calling to_natural() on already-natural Params returns same object."""
        p_nat = Params().to_natural()
        assert p_nat.to_natural() is p_nat

    def test_dimensionless_ratios_preserved(self):
        """Dimensionless quantities are unchanged by unit conversion."""
        p_si  = Params()
        p_nat = p_si.to_natural()
        assert p_nat.n_refr == pytest.approx(p_si.n_refr)
        assert p_nat.m_prime == pytest.approx(p_si.m_prime)
        assert p_nat.N_qw   == p_si.N_qw
        assert p_nat.T      == pytest.approx(p_si.T)

    def test_energy_ratio_preserved(self):
        """E_gap/E_bind ratio is preserved by the rescaling."""
        p_si  = Params()
        p_nat = p_si.to_natural()
        ratio_si  = p_si.E_gap  / p_si.E_bind
        ratio_nat = p_nat.E_gap / p_nat.E_bind
        assert ratio_nat == pytest.approx(ratio_si)


class TestToDict:
    def test_returns_dict(self):
        p = Params()
        d = p.to_dict()
        assert isinstance(d, dict)

    def test_contains_direct_fields(self):
        p = Params()
        d = p.to_dict()
        for key in ("E_bind", "E_gap_bare", "m_e", "m_h", "Omega", "T"):
            assert key in d, f"Missing key: {key}"

    def test_values_match(self):
        p = Params()
        d = p.to_dict()
        assert d["E_bind"]     == pytest.approx(p.E_bind)
        assert d["E_gap_bare"] == pytest.approx(p.E_gap_bare)
        assert d["T"]          == pytest.approx(p.T)
