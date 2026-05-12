"""
Unit tests for polaritons.grid (uniform momentum grid).
"""
import numpy as np
import pytest

from polaritons.grid import uniform_grid_and_weights


class TestUniformGrid:
	def test_shape(self):
		q, w = uniform_grid_and_weights(25.0, 1000)
		assert q.shape == (1000,)
		assert w.shape == (1000,)

	def test_endpoints(self):
		q, _ = uniform_grid_and_weights(25.0, 1000)
		assert q[0] == 0.0
		assert q[-1] == pytest.approx(25.0)

	def test_monotonic(self):
		q, _ = uniform_grid_and_weights(25.0, 1000)
		assert np.all(np.diff(q) > 0)

	def test_weights_sum_to_K_max(self):
		_, w = uniform_grid_and_weights(25.0, 1000)
		assert w.sum() == pytest.approx(25.0)

	def test_endpoint_weights_half_interior(self):
		_, w = uniform_grid_and_weights(10.0, 11)   # dx = 1.0
		assert w[0]  == pytest.approx(0.5)
		assert w[-1] == pytest.approx(0.5)
		assert np.all(w[1:-1] == pytest.approx(1.0))

	def test_invalid_N_raises(self):
		with pytest.raises(ValueError):
			uniform_grid_and_weights(25.0, 1)

	def test_invalid_K_max_raises(self):
		with pytest.raises(ValueError):
			uniform_grid_and_weights(0.0, 100)
