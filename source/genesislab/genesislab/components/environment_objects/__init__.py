"""Environment objects system for interactive scene objects.

This module provides a system for managing interactive objects in the scene
that are separate from robots and terrain. These objects have their own
joint spaces and don't interfere with robot control.

Configuration classes follow the same design pattern as RobotCfg/ArticulationCfg.
Objects are organized in SceneCfg as dict[str, ObjectCfg].
The ObjectManager is located in genesislab.managers.
"""

from .object_cfg import (
    ObjectCfg,
    InitialObjectPoseCfg,
    USDObjectCfg,
    PrimitiveObjectCfg,
)

__all__ = [
    "ObjectCfg",
    "InitialObjectPoseCfg",
    "USDObjectCfg",
    "PrimitiveObjectCfg",
]
