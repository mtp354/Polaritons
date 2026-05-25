"""
Unit tests for polaritons.plotting (formatters, labels, critical-momentum).

Matplotlib-only helpers (tick_formatter, format_plot_ticks, ...) are
exercised lightly behind a guard so the suite still runs in environments
without matplotlib.
"""
import numpy as np
import pytest

from polaritons.parameters import Params
from polaritons.plotting import (
	format_number, format_power_value, format_power_value_fixed,
	clean_number_slug, superscript_int,
	cavity_label, cavity_slug, display_kernel_type,
	build_result_slug, build_result_label,
	eta_indices_descending, eta_values_descending, eta_line_style,
	calculate_critical_momentum, q_energy_values,
)


@pytest.fixture(scope="module")
def p_nat():
	return Params().to_natural()


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class TestFormatters:
	def test_format_number_integer(self):
		assert format_number(42) == "42"

	def test_format_number_thousands(self):
		assert format_number(1234) == "1,234"

	def test_format_number_decimal_strip(self):
		assert format_number(1.500) == "1.5"

	def test_format_number_zero_tolerance(self):
		assert format_number(1e-15) == "0"

	def test_format_number_nonfinite(self):
		assert format_number(float("nan")) == ""

	def test_format_power_value_zero(self):
		assert format_power_value(0.0) == "0"

	def test_format_power_value_positive(self):
		# 1.5e3 -> "1.5×10³"
		out = format_power_value(1500.0)
		assert "10" in out and "\u00b3" in out  # superscript 3

	def test_format_power_value_negative_uses_minus_sign(self):
		out = format_power_value(-1.5e3)
		assert out.startswith("\u2212")

	def test_format_power_value_fixed_decimals(self):
		out = format_power_value_fixed(1.234e3, decimals=2)
		assert "1.23" in out

	def test_clean_number_slug_handles_signs_and_dots(self):
		assert clean_number_slug(-1.5) == "m1p5"
		assert clean_number_slug(2.0) == "2"

	def test_superscript_int(self):
		assert superscript_int(-12) == "\u207b\u00b9\u00b2"


# ---------------------------------------------------------------------------
# Labels / slugs
# ---------------------------------------------------------------------------

class TestLabels:
	def test_cavity_helpers(self):
		assert cavity_label(True).startswith("disorder-tuned")
		assert cavity_slug(False) == "disorder_free"

	def test_kernel_aliases(self):
		assert display_kernel_type("ng") == "Non-Gaussian"
		assert display_kernel_type("gaussian") == "Gaussian"
		assert display_kernel_type("foo") == "foo"  # passthrough

	def test_result_slug_xi_independent(self):
		assert build_result_slug("nongaussian", None, 1.0).startswith("nongaussian_")

	def test_result_slug_with_xi(self):
		slug = build_result_slug("gaussian", 20e-9, 1.0)
		assert slug.startswith("xi") and slug.endswith("nm_mprime1")

	def test_result_label_with_xi(self):
		label = build_result_label("gaussian", 20e-9, 1.5)
		assert "nm" in label and "1.5" in label

	def test_result_label_without_m_prime(self):
		label = build_result_label("gaussian", 20e-9, 1.5, show_m_prime=False)
		assert "m'" not in label


# ---------------------------------------------------------------------------
# Eta helpers
# ---------------------------------------------------------------------------

class TestEtaHelpers:
	def test_indices_descending(self):
		eta = np.array([0.0, 1.0, 0.5])
		# values 1.0, 0.5, 0.0 -> indices 1, 2, 0
		assert eta_indices_descending(eta) == [1, 2, 0]

	def test_values_descending(self):
		assert eta_values_descending([0.0, 2.0, 1.0]) == [2.0, 1.0, 0.0]

	def test_line_style_zero_is_dashed_black(self):
		clr, ls = eta_line_style(0, 5, 0.0)
		assert clr == "black" and ls == "--"


# ---------------------------------------------------------------------------
# Critical momentum
# ---------------------------------------------------------------------------

class TestCriticalMomentum:
	def test_no_crossing_returns_none(self, p_nat):
		q_grid   = np.linspace(0.0, 10.0, 11)
		eta_grid = np.array([0.0, 1.0])
		# Re Q is positive everywhere -> no sign change
		Q = np.ones((2, 11), dtype=complex)
		out = calculate_critical_momentum(q_grid, eta_grid, Q, p_nat, target_eta=1.0)
		assert out["critical_k_natural"] is None

	def test_linear_zero_crossing(self, p_nat):
		q_grid   = np.linspace(0.0, 10.0, 11)
		eta_grid = np.array([0.0, 1.0])
		# Re Q changes sign at q=4.0
		ReQ = q_grid - 4.0
		Q = np.zeros((2, 11), dtype=complex)
		Q[1] = ReQ
		out = calculate_critical_momentum(q_grid, eta_grid, Q, p_nat, target_eta=1.0)
		assert out["critical_eta"] == 1.0
		assert out["critical_k_natural"] == pytest.approx(4.0, abs=1e-12)

	def test_picks_nearest_target_eta(self, p_nat):
		eta_grid = np.array([0.0, 0.4, 1.6])
		q_grid   = np.linspace(0.0, 10.0, 6)
		Q = np.zeros((3, 6), dtype=complex)
		Q[2] = q_grid - 5.0
		out = calculate_critical_momentum(q_grid, eta_grid, Q, p_nat, target_eta=1.5)
		assert out["critical_eta"] == 1.6


# ---------------------------------------------------------------------------
# q_energy_values unit picker
# ---------------------------------------------------------------------------

class TestQEnergyValues:
	def test_picks_meV_for_small_values(self, p_nat):
		# Q in natural units of order 1e-3 -> ~1e-6 eV -> below threshold
		Q = np.array([1e-3, -2e-3])
		_, label, _ = q_energy_values(Q, p_nat, threshold_eV=0.1)
		assert label == "meV"

	def test_picks_eV_for_large_values(self, p_nat):
		# Q = 1000 natural -> ~4 eV -> above default 0.1 threshold
		Q = np.array([1000.0])
		_, label, _ = q_energy_values(Q, p_nat, threshold_eV=0.1)
		assert label == "eV"


# ---------------------------------------------------------------------------
# Smart tick formatter (requires matplotlib)
# ---------------------------------------------------------------------------

class TestSmartFormatter:
	def test_smart_uses_plain_in_band(self):
		pytest.importorskip("matplotlib")
		from polaritons.plotting import tick_formatter
		# max_abs in [1e-4, 1e5) -> plain number, no '×10'
		fmt = tick_formatter("smart", max_abs=1.5e2)
		assert "\u00d710" not in fmt(1234.0, 0)

	def test_smart_uses_scientific_below_low(self):
		pytest.importorskip("matplotlib")
		from polaritons.plotting import tick_formatter
		fmt = tick_formatter("smart", max_abs=1e-5, decimals=2)
		out = fmt(1e-5, 0)
		assert "\u00d710" in out

	def test_smart_uses_scientific_above_high(self):
		pytest.importorskip("matplotlib")
		from polaritons.plotting import tick_formatter
		fmt = tick_formatter("smart", max_abs=1e6, decimals=2)
		assert "\u00d710" in fmt(1e6, 0)
