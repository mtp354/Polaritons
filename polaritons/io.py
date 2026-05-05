"""
File I/O helpers for saving and loading computation results.

Conventions
-----------
Each result is stored as a pair of files in the same directory:

	<stem>.npy         the numerical array (float64 or complex128)
	<stem>_meta.json   metadata: parameters, grid info, timestamps, git hash

The `stem` is constructed from the result type and a short parameter hash so
that multiple sweeps can coexist without overwriting each other.

Directory layout (relative to project root)
-------------------------------------------
	Results/
		integrand_meshes/
			K_<hash>.npy
			K_<hash>_meta.json
		Q_results/
			Q_<hash>.npy
			Q_<hash>_meta.json
		interaction_strengths/
			g_<hash>.npy
			g_<hash>_meta.json
"""

from __future__ import annotations
import json
import hashlib
import datetime
import subprocess
from pathlib import Path

import numpy as np
from .parameters import Params


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _git_hash() -> str:
	"""Return current git commit short hash, or 'unknown'."""
	try:
		result = subprocess.run(
			["git", "rev-parse", "--short", "HEAD"],
			capture_output=True, text=True, timeout=3
		)
		return result.stdout.strip() or "unknown"
	except Exception:
		return "unknown"


def _param_hash(params_dict: dict, extra: dict | None = None) -> str:
	"""
	6-character hex hash of a parameter dict (plus any extra identifying info).
	Used to generate unique file stems.
	"""
	combined = {**(extra or {}), **params_dict}
	# Sort to make hash deterministic regardless of insertion order
	blob = json.dumps(combined, sort_keys=True, default=str)
	return hashlib.sha256(blob.encode()).hexdigest()[:6]


# ---------------------------------------------------------------------------
# Core save / load
# ---------------------------------------------------------------------------

def save_result(
	array      : np.ndarray,
	directory  : str | Path,
	stem       : str,
	params     : Params,
	extra_meta : dict | None = None,
) -> Path:
	"""
	Save `array` to `<directory>/<stem>.npy` and write accompanying metadata.

	Parameters
	----------
	array      : numpy array to save
	directory  : output directory (created if absent)
	stem       : file name stem (no extension)
	params     : Params instance (will be serialised via to_dict())
	extra_meta : any additional key-value pairs to include in the JSON

	Returns
	-------
	npy_path : Path to the saved .npy file
	"""
	directory = Path(directory)
	directory.mkdir(parents=True, exist_ok=True)

	npy_path  = directory / f"{stem}.npy"
	json_path = directory / f"{stem}_meta.json"

	np.save(npy_path, array)

	meta = {
		"saved_at"  : datetime.datetime.now(datetime.timezone.utc).isoformat(),
		"git_hash"  : _git_hash(),
		"array_shape"   : list(array.shape),
		"array_dtype"   : str(array.dtype),
		"params"    : params.to_dict(),
		**(extra_meta or {}),
	}
	with open(json_path, "w") as f:
		json.dump(meta, f, indent=2, default=str)

	print(f"Saved  {npy_path.name}  {list(array.shape)} {array.dtype}")
	return npy_path


def load_result(
	directory : str | Path,
	stem      : str,
) -> tuple[np.ndarray, dict]:
	"""
	Load array and metadata from `<directory>/<stem>.npy` + `_meta.json`.

	Returns
	-------
	array : numpy array
	meta  : metadata dict
	"""
	directory = Path(directory)
	npy_path  = directory / f"{stem}.npy"
	json_path = directory / f"{stem}_meta.json"

	array = np.load(npy_path, allow_pickle=False)
	with open(json_path) as f:
		meta = json.load(f)
	return array, meta


# ---------------------------------------------------------------------------
# Sweep storage helpers
# ---------------------------------------------------------------------------

def make_sweep_stem(prefix: str, params: Params, extra: dict | None = None) -> str:
	"""
	Build a file stem like  ``K_a3f2c1``  that uniquely identifies
	the combination of physical parameters + any extra sweep variables.
	"""
	h = _param_hash(params.to_dict(), extra=extra)
	return f"{prefix}_{h}"


def list_results(directory: str | Path, prefix: str = "") -> list[dict]:
	"""
	Return a list of metadata dicts for all saved results in `directory`
	whose stem starts with `prefix`.  Useful for browsing stored sweeps.
	"""
	directory = Path(directory)
	results = []
	glob_pat = f"{prefix}_*_meta.json" if prefix else "*_meta.json"
	for json_path in sorted(directory.glob(glob_pat)):
		with open(json_path) as f:
			meta = json.load(f)
		meta["_stem"] = json_path.stem.replace("_meta", "")
		meta["_json_path"] = str(json_path)
		results.append(meta)
	return results


def load_latest(
	directory : str | Path,
	prefix    : str = "",
) -> tuple[np.ndarray, dict]:
	"""
	Load the most recently saved result (by `saved_at` timestamp) whose
	stem starts with `prefix`.
	"""
	results = list_results(directory, prefix)
	if not results:
		raise FileNotFoundError(
			f"No results with prefix '{prefix}' found in {directory}"
	)
	results.sort(key=lambda r: r["saved_at"], reverse=True)
	stem = results[0]["_stem"]
	return load_result(directory, stem)
