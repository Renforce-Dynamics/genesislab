"""Configuration classes for environment objects."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class EnvironmentObjectCfg:
    """Base configuration for environment objects.

    Environment objects are interactive scene elements (furniture, props, etc.)
    that exist independently from robots and terrain. They can be dynamic
    (articulated) or static, and robots can interact with them.
    """

    name: str = "object"
    """Name of the object (must be unique in scene)."""

    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position (x, y, z) in meters."""

    rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Rotation quaternion (w, x, y, z)."""

    scale: float = 1.0
    """Uniform scale factor."""

    fixed: bool = False
    """If True, object is fixed in place (cannot be moved)."""

    collision: bool = True
    """If True, object has collision geometry."""


@dataclass
class USDObjectCfg(EnvironmentObjectCfg):
    """Configuration for USD-based objects.

    Loads objects from USD files. Can include articulated objects
    (chairs, cabinets with drawers, doors, etc.).

    Example:
        >>> # Load a complete scene with furniture
        >>> scene_obj = USDObjectCfg(
        ...     name="office_scene",
        ...     usd_path="office_furniture.usd",
        ...     pos=(0.0, 0.0, 0.0),
        ... )

        >>> # Load a single chair
        >>> chair = USDObjectCfg(
        ...     name="chair_01",
        ...     usd_path="chair.usd",
        ...     pos=(2.0, 1.0, 0.0),
        ... )
    """

    usd_path: str = ""
    """Path to USD file."""

    load_articulation: bool = True
    """If True, load articulated joints from USD. If False, treat as static."""

    def __post_init__(self):
        """Validate configuration."""
        if not self.usd_path:
            raise ValueError("usd_path must be specified for USDObjectCfg")


@dataclass
class PrimitiveObjectCfg(EnvironmentObjectCfg):
    """Configuration for primitive geometric objects.

    Creates simple geometric shapes (boxes, spheres, etc.) as interactive objects.
    Useful for testing or simple props.

    Example:
        >>> # Create a movable box
        >>> box = PrimitiveObjectCfg(
        ...     name="box_01",
        ...     shape="box",
        ...     size=(0.5, 0.5, 0.5),
        ...     pos=(1.0, 0.0, 0.25),
        ... )
    """

    shape: str = "box"
    """Shape type: 'box', 'sphere', 'cylinder', 'capsule'."""

    size: Tuple[float, ...] = (1.0, 1.0, 1.0)
    """Size parameters (depends on shape):
    - box: (length, width, height)
    - sphere: (radius,)
    - cylinder: (radius, height)
    - capsule: (radius, height)
    """

    mass: float = 1.0
    """Mass in kg (only if not fixed)."""

    friction: float = 0.5
    """Friction coefficient."""


@dataclass
class EnvironmentObjectsConfig:
    """Collection of environment objects for a scene.

    Example:
        >>> from genesislab.components.environment_objects import (
        ...     EnvironmentObjectsConfig,
        ...     USDObjectCfg,
        ...     PrimitiveObjectCfg,
        ... )
        >>>
        >>> objects_cfg = EnvironmentObjectsConfig(
        ...     usd_objects=[
        ...         USDObjectCfg(
        ...             name="furniture",
        ...             usd_path="complete_scene.usd",
        ...         ),
        ...     ],
        ...     primitive_objects=[
        ...         PrimitiveObjectCfg(
        ...             name="box",
        ...             shape="box",
        ...             size=(0.3, 0.3, 0.3),
        ...             pos=(1.0, 0.0, 0.15),
        ...         ),
        ...     ],
        ... )
    """

    usd_objects: List[USDObjectCfg] = field(default_factory=list)
    """List of USD objects to load."""

    primitive_objects: List[PrimitiveObjectCfg] = field(default_factory=list)
    """List of primitive objects to create."""

    load_after_robots: bool = True
    """If True, load objects after robots are initialized (recommended).
    This prevents object joints from interfering with robot control."""

    enable_self_collision: bool = True
    """Enable collision between different environment objects."""

    enable_robot_collision: bool = True
    """Enable collision between environment objects and robots."""
