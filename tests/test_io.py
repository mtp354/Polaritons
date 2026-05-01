"""
Unit tests for polaritons.io.

File-system operations use pytest's tmp_path fixture so tests are isolated.
"""
import json
import numpy as np
import pytest

from polaritons.parameters import Params
from polaritons.io import (
    _param_hash,
    _git_hash,
    save_result,
    load_result,
    make_sweep_stem,
    list_results,
    load_latest,
)


@pytest.fixture
def p():
    return Params()


@pytest.fixture
def sample_array():
    rng = np.random.default_rng(42)
    return rng.random((5, 10)).astype(np.float64)


@pytest.fixture
def complex_array():
    rng = np.random.default_rng(7)
    return (rng.random((4, 8)) + 1j * rng.random((4, 8))).astype(np.complex128)


# ---------------------------------------------------------------------------
# _git_hash
# ---------------------------------------------------------------------------

class TestGitHash:
    def test_returns_string(self):
        result = _git_hash()
        assert isinstance(result, str)

    def test_non_empty(self):
        result = _git_hash()
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _param_hash
# ---------------------------------------------------------------------------

class TestParamHash:
    def test_returns_six_hex_chars(self, p):
        h = _param_hash(p.to_dict())
        assert len(h) == 6
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self, p):
        d = p.to_dict()
        assert _param_hash(d) == _param_hash(d)

    def test_different_params_different_hash(self):
        p1 = Params(T=20.0)
        p2 = Params(T=30.0)
        assert _param_hash(p1.to_dict()) != _param_hash(p2.to_dict())

    def test_extra_dict_changes_hash(self, p):
        d = p.to_dict()
        h1 = _param_hash(d)
        h2 = _param_hash(d, extra={"mode": "kernel"})
        assert h1 != h2

    def test_extra_dict_deterministic(self, p):
        d = p.to_dict()
        extra = {"mode": "kernel", "N": 100}
        assert _param_hash(d, extra=extra) == _param_hash(d, extra=extra)


# ---------------------------------------------------------------------------
# save_result / load_result round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    def test_array_identical(self, tmp_path, sample_array, p):
        path = save_result(sample_array, tmp_path, "test_stem", p)
        loaded, meta = load_result(tmp_path, "test_stem")
        np.testing.assert_array_equal(loaded, sample_array)

    def test_complex_array_preserved(self, tmp_path, complex_array, p):
        save_result(complex_array, tmp_path, "cplx", p)
        loaded, _ = load_result(tmp_path, "cplx")
        np.testing.assert_array_equal(loaded, complex_array)

    def test_npy_file_created(self, tmp_path, sample_array, p):
        save_result(sample_array, tmp_path, "stem1", p)
        assert (tmp_path / "stem1.npy").exists()

    def test_json_file_created(self, tmp_path, sample_array, p):
        save_result(sample_array, tmp_path, "stem1", p)
        assert (tmp_path / "stem1_meta.json").exists()

    def test_meta_contains_required_keys(self, tmp_path, sample_array, p):
        save_result(sample_array, tmp_path, "meta_test", p)
        _, meta = load_result(tmp_path, "meta_test")
        for key in ("saved_at", "git_hash", "array_shape", "array_dtype", "params"):
            assert key in meta, f"Missing key: {key}"

    def test_meta_shape_matches(self, tmp_path, sample_array, p):
        save_result(sample_array, tmp_path, "shape_test", p)
        _, meta = load_result(tmp_path, "shape_test")
        assert meta["array_shape"] == list(sample_array.shape)

    def test_meta_dtype_matches(self, tmp_path, sample_array, p):
        save_result(sample_array, tmp_path, "dtype_test", p)
        _, meta = load_result(tmp_path, "dtype_test")
        assert meta["array_dtype"] == str(sample_array.dtype)

    def test_extra_meta_stored(self, tmp_path, sample_array, p):
        extra = {"note": "test run", "version": 2}
        save_result(sample_array, tmp_path, "extra_test", p, extra_meta=extra)
        _, meta = load_result(tmp_path, "extra_test")
        assert meta["note"] == "test run"
        assert meta["version"] == 2

    def test_returns_npy_path(self, tmp_path, sample_array, p):
        from pathlib import Path
        path = save_result(sample_array, tmp_path, "ret_test", p)
        assert isinstance(path, Path)
        assert path.suffix == ".npy"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_result(tmp_path, "nonexistent_stem")

    def test_directory_created_if_absent(self, tmp_path, sample_array, p):
        subdir = tmp_path / "new" / "subdir"
        save_result(sample_array, subdir, "nested", p)
        assert (subdir / "nested.npy").exists()

    def test_no_pickle(self, tmp_path, sample_array, p):
        """Arrays must be loadable with allow_pickle=False (no object arrays)."""
        save_result(sample_array, tmp_path, "nopickle", p)
        arr = np.load(tmp_path / "nopickle.npy", allow_pickle=False)
        np.testing.assert_array_equal(arr, sample_array)


# ---------------------------------------------------------------------------
# make_sweep_stem
# ---------------------------------------------------------------------------

class TestMakeSweepStem:
    def test_format(self, p):
        stem = make_sweep_stem("K", p)
        parts = stem.split("_")
        assert parts[0] == "K"
        assert len(parts[1]) == 6
        assert all(c in "0123456789abcdef" for c in parts[1])

    def test_deterministic(self, p):
        assert make_sweep_stem("K", p) == make_sweep_stem("K", p)

    def test_prefix_used(self, p):
        assert make_sweep_stem("Q", p).startswith("Q_")
        assert make_sweep_stem("g", p).startswith("g_")

    def test_different_params_different_stem(self):
        p1 = Params(xi=10e-9)
        p2 = Params(xi=20e-9)
        assert make_sweep_stem("K", p1) != make_sweep_stem("K", p2)


# ---------------------------------------------------------------------------
# list_results / load_latest
# ---------------------------------------------------------------------------

class TestListResults:
    def _save_n(self, tmp_path, n, p, prefix="K"):
        """Save n dummy arrays with sequential stems."""
        stems = []
        for i in range(n):
            arr  = np.array([float(i)])
            stem = f"{prefix}_stem{i:02d}"
            save_result(arr, tmp_path, stem, p)
            stems.append(stem)
        return stems

    def test_returns_list(self, tmp_path, p):
        self._save_n(tmp_path, 2, p)
        results = list_results(tmp_path)
        assert isinstance(results, list)

    def test_length(self, tmp_path, p):
        self._save_n(tmp_path, 3, p, prefix="K")
        results = list_results(tmp_path, prefix="K")
        assert len(results) == 3

    def test_stem_key_present(self, tmp_path, p):
        self._save_n(tmp_path, 1, p)
        results = list_results(tmp_path)
        assert "_stem" in results[0]

    def test_prefix_filtering(self, tmp_path, p):
        self._save_n(tmp_path, 2, p, prefix="K")
        self._save_n(tmp_path, 3, p, prefix="Q")
        assert len(list_results(tmp_path, prefix="K")) == 2
        assert len(list_results(tmp_path, prefix="Q")) == 3

    def test_empty_directory(self, tmp_path):
        results = list_results(tmp_path, prefix="X")
        assert results == []

    def test_json_path_present(self, tmp_path, p):
        self._save_n(tmp_path, 1, p)
        results = list_results(tmp_path)
        assert "_json_path" in results[0]


class TestLoadLatest:
    def test_returns_most_recent(self, tmp_path, p):
        """The most recently saved array should be loaded."""
        import time
        arr1 = np.array([1.0])
        arr2 = np.array([2.0])
        save_result(arr1, tmp_path, "latest_a", p)
        time.sleep(0.01)          # ensure different timestamps
        save_result(arr2, tmp_path, "latest_b", p)
        loaded, _ = load_latest(tmp_path)
        np.testing.assert_array_equal(loaded, arr2)

    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_latest(tmp_path, prefix="missing")
