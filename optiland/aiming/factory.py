"""Ray Aiming Strategies

This module provides various aiming strategies for ray tracing.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .strategies.cached import CachedAimingStrategy
from .strategies.fallback import FallbackAimingStrategy
from .strategies.iterative import IterativeAimingStrategy
from .strategies.paraxial import ParaxialAimingStrategy

if TYPE_CHECKING:
    from .base import AimingStrategy


class RayAimingStrategyFactory:
    """Factory for creating aiming strategies."""

    _strategies = {
        "iterative": IterativeAimingStrategy,
        "paraxial": ParaxialAimingStrategy,
        "cached": CachedAimingStrategy,
        "fallback": FallbackAimingStrategy,
    }

    @classmethod
    def create_strategy(cls, name: str, **kwargs) -> AimingStrategy:
        """
        Create an aiming strategy instance from a string identifier.

        Args:
            name (str): The name of the strategy to create.
            **kwargs: Additional keyword arguments for the strategy's constructor.

        Returns:
            AimingStrategy: An instance of the requested aiming strategy.

        Raises:
            ValueError: if the strategy name is unknown.
        """
        name = name.lower()
        if name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown aiming strategy: '{name}'. "
                f"Available strategies are: {available}"
            )

        strategy_class = cls._strategies[name]

        # Strategies like Cached and Fallback wrap other strategies.
        # We need to handle their instantiation specially.
        if name == "cached":
            base_strategy_name = kwargs.pop("base_strategy", "iterative")
            # separate kwargs for CachedAiming and the base strategy
            cached_kwargs = {}
            if "cache_size" in kwargs:
                cached_kwargs["cache_size"] = kwargs.pop("cache_size")

            base_strategy = cls.create_strategy(base_strategy_name, **kwargs)
            return strategy_class(base_strategy=base_strategy, **cached_kwargs)

        if name == "fallback":
            primary_strategy_name = kwargs.pop("primary_strategy", "iterative")
            fallback_strategy_name = kwargs.pop("fallback_strategy", "paraxial")

            primary_strategy = cls.create_strategy(primary_strategy_name, **kwargs)
            fallback_strategy = cls.create_strategy(fallback_strategy_name)

            return strategy_class(
                primary_strategy=primary_strategy, fallback_strategy=fallback_strategy
            )

        # For other strategies, we can instantiate directly.
        return strategy_class(**kwargs)
