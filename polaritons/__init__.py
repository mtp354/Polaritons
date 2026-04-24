from .parameters import Params, DEFAULT_PARAMS
from . import kernel, grid, solver, dispersion, many_body, io
# real_space is available as polaritons.real_space but not imported here
# to avoid the eager scipy.sparse dependency in the k-space notebooks.
