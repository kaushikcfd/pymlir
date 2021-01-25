from .affine import affine as affine_dialect
from .standard import standard as std_dialect
from .scf import scf as scf_dialect
from .linalg import linalg as linalg_dialect


STANDARD_DIALECTS = [affine_dialect, std_dialect, scf_dialect, linalg_dialect]
