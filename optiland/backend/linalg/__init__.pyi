from numpy.typing import ArrayLike

from optiland._types import BEArray
from optiland.backend import ndarray

def norm(x: BEArray, axis: int) -> BEArray: ...
def solve(a: ArrayLike, b: ArrayLike) -> ndarray: ...
