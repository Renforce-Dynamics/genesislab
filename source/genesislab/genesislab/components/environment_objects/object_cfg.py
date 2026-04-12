"""Configuration classes for environment objects.

Following the same design pattern as RobotCfg/ArticulationCfg.
All objects are configured individually and organized in SceneCfg as dict[str, ObjectCfg].
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from genesislab.utils.configclass import configclass


@configclass
class InitialObjectPoseCfg:
    """Initial pose configuration for an environment object."""

    pos: list[float] = [0.0, 0.0, 0.0]
    """Initial position (x, y, z)."""

    quat: list[float] = [1.0, 0.0, 0.0, 0.0]
    """Initial orientation quaternion [w, x, y, z] (wxyz format).

    Uses wxyz (scalar-first) format matching Genesis API and genesislab math utilities.
    Identity quaternion (no rotation) is [1, 0, 0, 0].
    """


@configclass
class ObjectCfg:
    """Base configuration for environment objects.

    Environment objects are interactive scene elements (furniture, props, etc.)
    that exist independently from robots and terrain. They can be dynamic
    (articulated) or static, and robots can interact with them.

    Similar to ArticulationCfg/RobotCfg design pattern.
    """

    name: str = MISSING
    """Logical name of the object (must be unique in scene)."""

    initial_pose: InitialObjectPoseCfg = InitialObjectPoseCfg()
    """Initial pose of the object."""

    scale: float = 1.0
    """Uniform scale factor."""

    fixed: bool = False
    """If True, object is fixed in place (cannot be moved)."""

    collision: bool = True
    """If True, object has collision geometry."""


@configclass
class USDObjectCfg(ObjectCfg):
    """Configuration for USD-based objects.

    Loads objects from USD files. Can include articulated objects
    (chairs, cabinets with drawers, doors, etc.).

    Example:
        >>> # Load a complete scene with furniture
        >>> scene_obj = USDObjectCfg(
        ...     name="office_scene",
        ...     usd_path="office_furniture.usd",
        ...     initial_pose=InitialObjectPoseCfg(pos=[0.0, 0.0, 0.0]),
        ... )

        >>> # Load a single chair
        >>> chair = USDObjectCfg(
        ...     name="chair_01",
        ...     usd_path="chair.usd",
        ...     initial_pose=InitialObjectPoseCfg(pos=[2.0, 1.0, 0.0]),
        ... )
    """

    usd_path: str = MISSING
    """Path to USD file."""

    load_articulation: bool = True
    """If True, load articulated joints from USD. If False, treat as static."""


@configclass
class PrimitiveObjectCfg(ObjectCfg):
    """Configuration for primitive geometric objects.

    Creates simple geometric shapes (boxes, spheres, etc.) as interactive objects.
    Useful for testing or simple props.

    Example:
        >>> # Create a movable box
        >>> box = PrimitiveObjectCfg(
        ...     name="box_01",
        ...     shape="box",
        ...     size=[0.5, 0.5, 0.5],
        ...     initial_pose=InitialObjectPoseCfg(pos=[1.0, 0.0, 0.25]),
        ... )
    """

    shape: Literal["box", "sphere", "cylinder", "capsule"] = "box"
    """Shape type: 'box', 'sphere', 'cylinder', 'capsule'."""

    size: list[float] = [1.0, 1.0, 1.0]
    """Size parameters (depends on shape):
    - box: [length, width, height]
    - sphere: [radius]
    - cylinder: [radius, height]
    - capsule: [radius, height]
    """

    mass: float = 1.0
    """Mass in kg (only if not fixed)."""

    friction: float = 0.5
    """Friction coefficient."""
