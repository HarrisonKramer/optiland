"""
Base Viewer Module
This module defines the `BaseViewer` abstract base class, which serves as
the foundation for all visualization viewers within the Optiland library.
It establishes a common interface for initializing viewers with an optical
system and for generating visualizations.

Manuel Fragata Mendes, June 2025
"""

from __future__ import annotations

import abc


class BaseViewer(abc.ABC):
    """
    Abstract base class for all visualization viewers in Optiland.

    This class defines the standard interface for viewers, ensuring that
    each viewer is initialized with an Optic object and has a `view`
    method to generate the visualization.
    """

    def __init__(self, optic):
        """
        Initializes the BaseViewer.

        Args:
            optic (Optic): The optical system to be visualized.
        """
        self.optic = optic

    @abc.abstractmethod
    def view(self, *args, **kwargs):
        """
        The main method to generate and display the visualization.

        This method must be implemented by all concrete viewer subclasses.
        """
        pass
