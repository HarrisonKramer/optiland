from numpy.typing import ArrayLike

from optiland._types import BEArrayT
from optiland.backend import ndarray

def norm(x: BEArrayT, axis: int) -> BEArrayT: ...
def solve(a: ArrayLike, b: ArrayLike) -> ndarray: ...
