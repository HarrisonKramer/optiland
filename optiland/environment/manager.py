"""Environment Manager module

This module defines the EnvironmentManager class, which manages the global
optical immersion environment for simulations. It allows setting and retrieving the
active environment.

The default environment is standard air, but users can configure it to other
environments as needed.

Kramer Harrison, 2025
"""

from __future__ import annotations

from optiland.materials.air import Air

from .conditions import EnvironmentalConditions
from .environment import Environment


class EnvironmentManager:
    """
    A singleton class to manage the global optical environment.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization
        if not hasattr(self, "_initialized"):
            self._default_conditions = EnvironmentalConditions()
            self._default_medium = Air(conditions=self._default_conditions)
            self._active_environment = Environment(
                medium=self._default_medium, conditions=self._default_conditions
            )
            self._initialized = True

    def set_environment(self, environment: Environment):
        """
        Sets the globally active optical environment.

        Args:
            environment (Environment): The environment configuration to activate.
        """
        self._active_environment = environment

    def get_environment(self) -> Environment:
        """
        Retrieves the currently active optical environment.
        """
        return self._active_environment

    def reset_to_default(self):
        """Resets the environment to standard air."""
        self.set_environment(
            Environment(self._default_medium, self._default_conditions)
        )


# Create the globally accessible instance
environment_manager = EnvironmentManager()
