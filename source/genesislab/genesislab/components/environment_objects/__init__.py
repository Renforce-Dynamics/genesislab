"""Environment objects system for interactive scene objects.

This module provides a system for managing interactive objects in the scene
that are separate from robots and terrain. These objects have their own
joint spaces and don't interfere with robot control.
"""

from .object_cfg import (
    EnvironmentObjectCfg,
    USDObjectCfg,
    PrimitiveObjectCfg,
    EnvironmentObjectsConfig,
)
from .object_manager import EnvironmentObjectManager

__all__ = [
    "EnvironmentObjectCfg",
    "USDObjectCfg",
    "PrimitiveObjectCfg",
    "EnvironmentObjectsConfig",
    "EnvironmentObjectManager",
]
