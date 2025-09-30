"""Environmental Modeling Package

This package provides tools for calculating environmental parameters
relevant to optics, such as the refractive index of air under various
conditions using multiple models.

Modules:
    conditions: Defines the `EnvironmentalConditions` dataclass.
    ciddor: Implements the Ciddor (1996) model.
    kohlrausch: Implements a simplified Kohlrausch model.
    edlen: Implements the Edlén (1966) model.
    birch_downs: Implements the Birch & Downs (1993/1994) model.
    air_index: Provides a unified interface `refractive_index_air` for
               accessing all supported air refractive index models.

Key exports:
    EnvironmentalConditions: Dataclass for specifying environmental parameters.
    refractive_index_air: Unified function to calculate air refractive index.
    ciddor_refractive_index: Direct access to Ciddor model.
    kohlrausch_refractive_index: Direct access to Kohlrausch model.
    edlen_refractive_index: Direct access to Edlén model.
    birch_downs_refractive_index: Direct access to Birch & Downs model.
"""

from __future__ import annotations

from .air_index import refractive_index_air
from .conditions import EnvironmentalConditions
from .models.birch_downs import birch_downs_refractive_index
from .models.ciddor import ciddor_refractive_index
from .models.edlen import edlen_refractive_index
from .models.kohlrausch import kohlrausch_refractive_index

__all__ = [
    "EnvironmentalConditions",
    "refractive_index_air",
    "ciddor_refractive_index",
    "kohlrausch_refractive_index",
    "edlen_refractive_index",
    "birch_downs_refractive_index",
]
