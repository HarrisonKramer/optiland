"""Environment Manager module

This module defines the EnvironmentManager class, which manages the global
optical immersion environment for simulations. It allows setting and retrieving the
active environment.

The default environment is standard air, but users can configure it to other
environments as needed.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import Any

from typing_extensions import Self

from optiland.materials.air import Air

from .conditions import EnvironmentalConditions
from .environment import Environment


class EnvironmentManager:
    """A singleton class to manage the global optical environment.

    This manager ensures that a single, consistent environment is used across
    the simulation unless explicitly changed.

    Attributes:
        _instance (EnvironmentManager | None): The singleton instance.
    """

    _instance: EnvironmentManager | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        """Ensures that only one instance of EnvironmentManager is created."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initializes the EnvironmentManager, setting up the default environment.

        This method prevents re-initialization if the instance already exists.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._default_conditions = EnvironmentalConditions()
        self._default_medium = Air(conditions=self._default_conditions)
        self._active_environment = Environment(
            medium=self._default_medium, conditions=self._default_conditions
        )
        self._initialized = True

    def set_environment(self, environment: Environment) -> None:
        """Sets the globally active optical environment.

        Args:
            environment (Environment): The environment configuration to activate.
        """
        self._active_environment = environment

    def get_environment(self) -> Environment:
        """Retrieves the currently active optical environment.

        Returns:
            Environment: The active optical environment.
        """
        return self._active_environment

    def reset_to_default(self) -> None:
        """Resets the environment to standard air (20°C, 1 atm)."""
        self.set_environment(
            Environment(self._default_medium, self._default_conditions)
        )


# Create the globally accessible singleton instance
environment_manager = EnvironmentManager()
