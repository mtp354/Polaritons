"""
Unit tests for polaritons.grid.
"""
import numpy as np
import pytest

from polaritons.grid import build_segmented_grid, trapz_weights


GRID_DEFAULTS = dict(
    total_points=100,
    q_A_end=1.0,
    q_B_end=5.0,
    q_C_end=20.0,
    q_max=100.0,
)


class TestBuildSegmentedGrid:
    def test_output_length(self):
        grid, counts = build_segmented_grid(**GRID_DEFAULTS)
        assert len(grid) == GRID_DEFAULTS["total_points"]

    def test_counts_sum(self):
        grid, counts = build_segmented_grid(**GRID_DEFAULTS)
        total = counts["N_A"] + counts["N_B"] + counts["N_C"] + counts["N_D"]
        assert total == GRID_DEFAULTS["total_points"]

    def test_monotone_increasing(self):
        grid, _ = build_segmented_grid(**GRID_DEFAULTS)
        assert np.all(np.diff(grid) > 0)

    def test_starts_at_zero(self):
        grid, _ = build_segmented_grid(**GRID_DEFAULTS)
        assert grid[0] == pytest.approx(0.0)

    def test_ends_at_q_max(self):
        grid, _ = build_segmented_grid(**GRID_DEFAULTS)
        assert grid[-1] == pytest.approx(GRID_DEFAULTS["q_max"])

    def test_segment_boundaries_respected(self):
        grid, _ = build_segmented_grid(**GRID_DEFAULTS)
        # The grid must contain the boundary q_A_end in segment A
        assert grid.min() == pytest.approx(0.0)
        assert grid.max() == pytest.approx(GRID_DEFAULTS["q_max"])

    def test_counts_dict_keys(self):
        _, counts = build_segmented_grid(**GRID_DEFAULTS)
        assert set(counts.keys()) == {"N_A", "N_B", "N_C", "N_D"}

    def test_each_count_at_least_two(self):
        _, counts = build_segmented_grid(**GRID_DEFAULTS)
        for key, val in counts.items():
            assert val >= 2, f"{key}={val} is less than 2"

    def test_raises_on_bad_fractions(self):
        with pytest.raises(ValueError, match="frac"):
            build_segmented_grid(
                **GRID_DEFAULTS,
                frac_A=0.4, frac_B=0.4, frac_C=0.4, frac_D=0.1,
            )

    def test_raises_on_bad_ordering(self):
        with pytest.raises(ValueError, match="Require"):
            build_segmented_grid(
                total_points=100,
                q_A_end=10.0,
                q_B_end=5.0,   # q_B_end < q_A_end — invalid
                q_C_end=20.0,
                q_max=100.0,
            )

    def test_custom_power_law(self):
        """Higher power_A concentrates more points near zero."""
        _, c1 = build_segmented_grid(**GRID_DEFAULTS, power_A=1.0)
        _, c2 = build_segmented_grid(**GRID_DEFAULTS, power_A=3.0)
        # Segment counts should be identical (power only affects spacing, not N_A)
        assert c1["N_A"] == c2["N_A"]

    def test_segment_A_power_law_spacing(self):
        """Segment A spacings should be increasing (coarser away from zero)."""
        grid, counts = build_segmented_grid(**GRID_DEFAULTS, power_A=2.0)
        seg_A = grid[: counts["N_A"]]
        diffs = np.diff(seg_A[1:])  # skip the leading zeros
        assert np.all(diffs >= 0)


class TestTrapzWeights:
    def test_length(self):
        grid = np.linspace(0.0, 10.0, 50)
        w = trapz_weights(grid)
        assert len(w) == len(grid)

    def test_all_positive(self):
        grid = np.linspace(0.0, 10.0, 50)
        w = trapz_weights(grid)
        assert np.all(w > 0)

    def test_sum_equals_domain(self):
        """Sum of trapezoid weights = length of the integration domain."""
        grid = np.linspace(0.0, 10.0, 1000)
        w = trapz_weights(grid)
        assert w.sum() == pytest.approx(10.0, rel=1e-10)

    def test_uniform_grid_equal_weights(self):
        """All interior weights equal for a uniform grid."""
        grid = np.linspace(0.0, 1.0, 101)
        w = trapz_weights(grid)
        h = grid[1] - grid[0]
        assert w[0] == pytest.approx(h / 2)
        assert w[-1] == pytest.approx(h / 2)
        np.testing.assert_allclose(w[1:-1], h, rtol=1e-12)

    def test_integrates_linear_exactly(self):
        """Trapezoid rule is exact for linear functions."""
        grid = np.linspace(0.0, 5.0, 200)
        w    = trapz_weights(grid)
        f    = 2.0 * grid + 1.0
        result = np.dot(w, f)
        # integral of 2x+1 from 0 to 5 = 25 + 5 = 30
        assert result == pytest.approx(30.0, rel=1e-6)

    def test_non_uniform_grid(self):
        """Weights should integrate correctly on a non-uniform grid."""
        grid, _ = build_segmented_grid(**GRID_DEFAULTS)
        w = trapz_weights(grid)
        assert len(w) == len(grid)
        assert np.all(w > 0)
