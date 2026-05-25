"""
Plotting helpers (label formatters, colour selection, critical-momentum
detection) extracted from the visualisation notebooks so that they can be
unit-tested.

Matplotlib-specific helpers (`tick_formatter`, `format_plot_ticks`,
`apply_centered_critical_axis`, `hide_axis_offset`) import matplotlib lazily
inside the function body to keep the polaritons package importable in
environments without matplotlib.
"""

from __future__ import annotations
import numpy as np


SUPERSCRIPT = str.maketrans("0123456789-+", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b\u207a")


# ---------------------------------------------------------------------------
# Number / label formatting
# ---------------------------------------------------------------------------

def superscript_int(value: int) -> str:
	return str(int(value)).translate(SUPERSCRIPT)


def format_number(value) -> str:
	value = float(value)
	if not np.isfinite(value):
		return ""
	if abs(value) < 5e-13:
		value = 0.0
	value = round(value, 5)
	if abs(value - round(value)) < 5e-12:
		return f"{int(round(value)):,}" if abs(value) >= 1000 else str(int(round(value)))
	text = f"{value:,.5f}" if abs(value) >= 1000 else f"{value:.5f}"
	return text.rstrip("0").rstrip(".")


def format_power_value(value) -> str:
	value = float(value)
	if not np.isfinite(value):
		return ""
	if abs(value) < 1e-12:
		return "0"
	sign = "\u2212" if value < 0 else ""
	value = abs(value)
	exponent = int(np.floor(np.log10(value)))
	coefficient = value / 10**exponent
	return f"{sign}{format_number(coefficient)}\u00d710{superscript_int(exponent)}"


def format_power_value_fixed(value, *, decimals=2, zero_tol=1e-12) -> str:
	value = float(value)
	if not np.isfinite(value):
		return ""
	if value == 0.0 or (zero_tol is not None and abs(value) < zero_tol):
		return "0"
	sign = "\u2212" if value < 0 else ""
	value = abs(value)
	exponent = int(np.floor(np.log10(value)))
	coefficient = value / 10**exponent
	return f"{sign}{coefficient:.{decimals}f}\u00d710{superscript_int(exponent)}"


def clean_number_slug(value) -> str:
	text = f"{float(value):.6g}"
	return text.replace("-", "m").replace("+", "").replace(".", "p")


# ---------------------------------------------------------------------------
# Colour / line-style helpers
# ---------------------------------------------------------------------------

def eta_color(index: int, total: int, cmap=None):
	"""Sample a colormap at a fraction depending on (index, total).

	Defaults to matplotlib's 'viridis' colormap when `cmap` is None.
	"""
	if cmap is None:
		import matplotlib.pyplot as plt
		cmap = plt.get_cmap("viridis")
	denom = max(total - 1, 1)
	return cmap(0.12 + 0.83 * index / denom)


def eta_line_style(index: int, total: int, eta_value: float, *, linestyle: str = "-", cmap=None):
	"""Black/dashed for eta=0, otherwise (cmap colour, linestyle)."""
	if np.isclose(float(eta_value), 0.0, atol=1e-12):
		return "black", "--"
	return eta_color(index, total, cmap=cmap), linestyle


def eta_indices_descending(eta_grid, indices=None):
	eta_grid = np.asarray(eta_grid, dtype=float)
	candidates = range(len(eta_grid)) if indices is None else [int(idx) for idx in indices]
	return sorted(candidates, key=lambda idx: float(eta_grid[idx]), reverse=True)


def eta_values_descending(values):
	return sorted([float(v) for v in values], reverse=True)


# ---------------------------------------------------------------------------
# Cavity / kernel labels
# ---------------------------------------------------------------------------

def cavity_label(disorder_tuned: bool) -> str:
	return "disorder-tuned cavity" if disorder_tuned else "disorder-free cavity"


def cavity_slug(disorder_tuned: bool) -> str:
	return "disorder_tuned" if disorder_tuned else "disorder_free"


_KERNEL_DISPLAY = {
	"gaussian":    "Gaussian",
	"nongaussian": "Non-Gaussian",
	"ng":          "Non-Gaussian",
}


def display_kernel_type(kernel_type) -> str:
	return _KERNEL_DISPLAY.get(str(kernel_type), str(kernel_type))


def build_result_slug(kernel_type, xi_m, m_prime) -> str:
	kernel_part = str(kernel_type).replace(" ", "_").replace("-", "_").lower()
	m_part = f"mprime{clean_number_slug(m_prime)}"
	if xi_m is None:
		return f"{kernel_part}_{m_part}"
	return f"xi{clean_number_slug(1e9 * xi_m)}nm_{m_part}"


def build_result_label(kernel_type, xi_m, m_prime, *, show_m_prime: bool = True) -> str:
	parts = [display_kernel_type(kernel_type)] if xi_m is None else [f"\u03be={1e9 * xi_m:g} nm"]
	if show_m_prime:
		parts.append(f"m'={m_prime:g}")
	return ", ".join(parts)


# ---------------------------------------------------------------------------
# Critical momentum from sign change of Re[Q]
# ---------------------------------------------------------------------------

def calculate_critical_momentum(q_grid, eta_grid, Q_results, p_nat, *, target_eta: float = 1.0) -> dict:
	"""
	Return the smallest q>0 at which Re[Q(q, target_eta)] crosses zero.

	Returns
	-------
	dict with keys ``critical_eta``, ``critical_k_natural``, ``critical_k_cm_inv``
	(the last two are ``None`` if no crossing is detected).
	"""
	from .units import k_nat_to_cm  # local import to avoid hard dep at module load
	q          = np.asarray(q_grid, dtype=float)
	eta_values = np.asarray(eta_grid, dtype=float)
	Q          = np.asarray(Q_results)
	eta_idx    = int(np.argmin(np.abs(eta_values - target_eta)))
	eta_value  = float(eta_values[eta_idx])
	y          = np.real(Q[eta_idx])

	mask  = (q > 0.0) & np.isfinite(q) & np.isfinite(y)
	q_pos = q[mask]
	y_pos = y[mask]
	if q_pos.size < 2:
		return {"critical_eta": eta_value, "critical_k_natural": None, "critical_k_cm_inv": None}

	zero_hits = np.flatnonzero(np.isclose(y_pos, 0.0, rtol=1e-12, atol=1e-12))
	if zero_hits.size:
		k_crit = float(q_pos[int(zero_hits[0])])
	else:
		crossings = np.flatnonzero(np.signbit(y_pos[:-1]) != np.signbit(y_pos[1:]))
		if crossings.size == 0:
			return {"critical_eta": eta_value, "critical_k_natural": None, "critical_k_cm_inv": None}
		i = int(crossings[0])
		k0, k1 = float(q_pos[i]), float(q_pos[i + 1])
		y0, y1 = float(y_pos[i]), float(y_pos[i + 1])
		k_crit = k0 if y1 == y0 else k0 - y0 * (k1 - k0) / (y1 - y0)

	return {
		"critical_eta": eta_value,
		"critical_k_natural": float(k_crit),
		"critical_k_cm_inv": float(k_nat_to_cm(k_crit, p_nat)),
	}


def q_energy_values(Q_nat, p_nat, *, threshold_eV: float = 0.1):
	"""
	Return (Q_in_display_units, unit_label, max_abs_eV).

	Picks meV for small magnitudes, eV otherwise.
	"""
	from .units import energy_nat_to_eV, energy_nat_to_meV
	Q_eV = energy_nat_to_eV(Q_nat, p_nat)
	max_abs_eV = float(np.nanmax(np.abs(Q_eV))) if np.size(Q_eV) else 0.0
	if max_abs_eV < threshold_eV:
		return energy_nat_to_meV(Q_nat, p_nat), "meV", max_abs_eV
	return Q_eV, "eV", max_abs_eV


# ---------------------------------------------------------------------------
# Energy convention: library quantities are band-bottom relative (clean
# exciton minimum = 0). Plotting code adds the semiconductor offset
# (E_gap - E_bind) to recover absolute energies.
# ---------------------------------------------------------------------------

def semiconductor_offset_natural(p_nat) -> float:
	"""Semiconductor offset E_gap - E_bind in natural energy units."""
	return float(p_nat.E_gap - p_nat.E_bind)


def semiconductor_offset_eV(p_nat) -> float:
	"""Semiconductor offset E_gap - E_bind in eV (uses p_nat.E_unit)."""
	return float((p_nat.E_gap - p_nat.E_bind) * p_nat.E_unit)


def to_absolute_eV(E_nat, p_nat):
	"""Convert a band-bottom-relative natural energy to absolute eV."""
	from .units import energy_nat_to_eV
	return energy_nat_to_eV(E_nat, p_nat) + semiconductor_offset_eV(p_nat)


# ---------------------------------------------------------------------------
# Matplotlib tick / axis helpers (matplotlib imported lazily)
# ---------------------------------------------------------------------------

def axis_names(axes):
	return (axes,) if isinstance(axes, str) else tuple(axes)


def hide_axis_offset(ax, axes=("x", "y")):
	for axis_name in axis_names(axes):
		getattr(ax, f"{axis_name}axis").get_offset_text().set_visible(False)


def tick_formatter(style="number", *, decimals=2, zero_tol=1e-12, max_abs=None, low=1e-4, high=1e5):
	from matplotlib.ticker import FuncFormatter
	if style == "number":
		return FuncFormatter(lambda value, pos: format_number(value))
	if style == "power":
		return FuncFormatter(lambda value, pos: format_power_value(value))
	if style == "fixed_power":
		return FuncFormatter(lambda value, pos: format_power_value_fixed(value, decimals=decimals, zero_tol=zero_tol))
	if style == "smart":
		# Plain number unless the axis range sits outside [low, high); then
		# fall back to fixed-decimal scientific form (e.g. 1.23×10⁻⁵).
		use_sci = (
			max_abs is not None
			and np.isfinite(max_abs)
			and max_abs > 0.0
			and (abs(max_abs) < low or abs(max_abs) >= high)
		)
		if use_sci:
			return FuncFormatter(lambda value, pos: format_power_value_fixed(value, decimals=decimals, zero_tol=zero_tol))
		return FuncFormatter(lambda value, pos: format_number(value))
	if callable(style):
		return FuncFormatter(lambda value, pos: style(value))
	raise ValueError(f"Unknown tick formatter: {style!r}")


def _smart_axis_max_abs(ax, axis_name: str) -> float:
	lo, hi = getattr(ax, f"get_{axis_name}lim")()
	return max(abs(float(lo)), abs(float(hi)))


def format_plot_ticks(ax, *, axes=("x", "y"), ticks=None, nbins=None, formatter="number", decimals=2, zero_tol=1e-12, low=1e-4, high=1e5):
	from matplotlib.ticker import FixedLocator, MaxNLocator
	if ticks is not None and nbins is not None:
		raise ValueError("Pass either ticks or nbins, not both.")
	for axis_name in axis_names(axes):
		axis = getattr(ax, f"{axis_name}axis")
		if ticks is not None:
			axis.set_major_locator(FixedLocator(ticks))
		elif nbins is not None:
			axis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=2))
		max_abs = _smart_axis_max_abs(ax, axis_name) if formatter == "smart" else None
		axis.set_major_formatter(tick_formatter(formatter, decimals=decimals, zero_tol=zero_tol, max_abs=max_abs, low=low, high=high))
	hide_axis_offset(ax)


def apply_centered_critical_axis(ax, critical_k_cm, *, decimals=2, fallback_xmax=None, fallback_tick_count=5):
	from matplotlib.ticker import FixedLocator
	if critical_k_cm is None or not np.isfinite(critical_k_cm) or critical_k_cm <= 0.0:
		xmax = float(ax.get_xlim()[1] if fallback_xmax is None else fallback_xmax)
		if xmax > 0 and np.isfinite(xmax):
			format_plot_ticks(ax, axes="x", nbins=max(fallback_tick_count - 1, 1), formatter="power")
		return
	critical_k_cm = float(critical_k_cm)
	xmax = 2.0 * critical_k_cm
	ax.set_xlim(0.0, xmax)
	ticks = [0.0, 0.5 * critical_k_cm, critical_k_cm, 1.5 * critical_k_cm, xmax]
	labels = ["0.00"] + [format_power_value_fixed(tick, decimals=decimals) for tick in ticks[1:]]
	ax.xaxis.set_major_locator(FixedLocator(ticks))
	ax.set_xticklabels(labels)
	hide_axis_offset(ax)
