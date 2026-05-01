"""
Non-uniform quadrature grid construction for Picard iteration.

The momentum domain is split into four segments with different spacing
strategies to resolve both the IR behaviour and the resonance peak:

    A  [0,          q_A_end]  power-law clustered near zero
    B  [q_A_end,    q_B_end]  coarse linear
    C  [q_B_end,    q_C_end]  dense linear (resonance region)
    D  [q_C_end,    q_max ]   sparse log-spaced tail
"""

from __future__ import annotations
import numpy as np


def build_segmented_grid(
    total_points : int,
    q_A_end      : float,
    q_B_end      : float,
    q_C_end      : float,
    q_max        : float,
    frac_A       : float = 0.25,
    frac_B       : float = 0.10,
    frac_C       : float = 0.55,
    frac_D       : float = 0.10,
    power_A      : float = 2.0,
) -> tuple[np.ndarray, dict]:
    """
    Build a monotone 4-segment non-uniform momentum grid.

    Parameters
    ----------
    total_points : total number of grid points
    q_A_end      : end of IR segment
    q_B_end      : end of coarse linear segment
    q_C_end      : end of resonance segment
    q_max        : maximum momentum (end of log tail)
    frac_A/B/C/D : fraction of total_points allocated to each segment
    power_A      : power-law exponent for segment A

    Returns
    -------
    grid   : 1-D array of length total_points
    counts : dict with keys N_A, N_B, N_C, N_D
    """
    if abs(frac_A + frac_B + frac_C + frac_D - 1.0) > 1e-12:
        raise ValueError("frac_A + frac_B + frac_C + frac_D must equal 1")
    if not (0 < q_A_end < q_B_end < q_C_end < q_max):
        raise ValueError("Require 0 < q_A_end < q_B_end < q_C_end < q_max")

    N_A = max(2, round(frac_A * total_points))
    N_B = max(2, round(frac_B * total_points))
    N_C = max(2, round(frac_C * total_points))
    N_D = max(2, total_points - N_A - N_B - N_C)   # absorb rounding remainder

    # Segment A: power-law spacing near zero
    q_A = q_A_end * np.linspace(0.0, 1.0, N_A) ** power_A

    # Segment B: coarse linear (exclude shared boundary)
    q_B = np.linspace(q_A_end, q_B_end, N_B + 1)[1:]

    # Segment C: dense linear around resonance (exclude shared boundary)
    q_C = np.linspace(q_B_end, q_C_end, N_C + 1)[1:]

    # Segment D: log-spaced tail (exclude shared boundary)
    q_D = np.geomspace(q_C_end, q_max, N_D + 1)[1:]

    grid   = np.concatenate([q_A, q_B, q_C, q_D])
    counts = {"N_A": N_A, "N_B": N_B, "N_C": N_C, "N_D": N_D}
    return grid, counts


def trapz_weights(grid: np.ndarray) -> np.ndarray:
    """
    Trapezoid quadrature weights for a non-uniform 1-D grid.
    """
    w        = np.empty_like(grid)
    w[0]     = (grid[1]  - grid[0])  / 2.0
    w[-1]    = (grid[-1] - grid[-2]) / 2.0
    w[1:-1]  = (grid[2:] - grid[:-2]) / 2.0
    return w
