"""Base class for non-sequential visualization viewers.

Kramer Harrison, 2026
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.nonsequential.scene import NonSequentialScene


class BaseNSQViewer(abc.ABC):
    """Abstract base for non-sequential scene viewers.

    Args:
        scene: The non-sequential scene to visualize.
    """

    def __init__(self, scene: NonSequentialScene) -> None:
        self.scene = scene

    @abc.abstractmethod
    def view(self, *args, **kwargs):
        """Render the scene."""
