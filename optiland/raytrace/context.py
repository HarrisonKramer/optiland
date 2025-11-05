from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.materials import BaseMaterial


@dataclass
class TracingContext:
    """A container for information about the current ray tracing context."""

    material: BaseMaterial
