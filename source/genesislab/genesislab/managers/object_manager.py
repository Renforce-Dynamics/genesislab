"""Manager for environment objects in the scene.

Manages loading and lifecycle of interactive objects that are separate from robots.
Objects are loaded after robots to avoid joint indexing conflicts.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict

import genesis as gs

from genesislab.components.environment_objects.object_cfg import (
    ObjectCfg,
    USDObjectCfg,
    PrimitiveObjectCfg,
)

if TYPE_CHECKING:
    from genesislab.envs.manager_based_rl_env import ManagerBasedRlEnv

logger = logging.getLogger(__name__)


class ObjectManager:
    """Manages environment objects in the scene.

    This manager handles loading and lifecycle of interactive objects
    that are separate from robots. Objects are loaded after robots
    to avoid joint indexing conflicts.

    Unlike other managers (action, observation, etc.), this is not a term-based manager.
    It's a resource manager similar to how sensors or terrains are handled.

    Attributes:
        objects_cfg: Dictionary of object configurations keyed by object name.
        scene: Genesis scene instance.
        objects: Dictionary of loaded object entities keyed by name.
    """

    def __init__(
        self,
        objects_cfg: dict[str, ObjectCfg],
        scene: gs.Scene,
        env: "ManagerBasedRlEnv" = None,
    ):
        """Initialize object manager.

        Args:
            objects_cfg: Dictionary of object configurations keyed by object name.
            scene: Genesis scene instance (must be built).
            env: Optional environment instance (for compatibility with manager pattern).
        """
        self.objects_cfg = objects_cfg
        self.scene = scene
        self._env = env
        self.objects: Dict[str, Any] = {}

        logger.info("ObjectManager initialized")

    def load_objects(self) -> None:
        """Load all configured environment objects.

        This should be called AFTER robots are added and initialized,
        but BEFORE the first simulation step.

        The loading order ensures that object joints don't interfere
        with robot control space.
        """
        if not self.objects_cfg:
            logger.info("No environment objects to load")
            return

        logger.info(f"Loading {len(self.objects_cfg)} environment objects...")

        for obj_name, obj_cfg in self.objects_cfg.items():
            # Validate that config name matches dict key
            if obj_cfg.name != obj_name:
                logger.warning(
                    f"Object config name '{obj_cfg.name}' does not match "
                    f"dict key '{obj_name}'. Using dict key."
                )
                obj_cfg.name = obj_name

            # Load based on type
            if isinstance(obj_cfg, USDObjectCfg):
                self._load_usd_object(obj_cfg)
            elif isinstance(obj_cfg, PrimitiveObjectCfg):
                self._load_primitive_object(obj_cfg)
            else:
                logger.error(
                    f"Unknown object type for '{obj_name}': {type(obj_cfg)}. "
                    f"Supported: USDObjectCfg, PrimitiveObjectCfg"
                )
                continue

        logger.info(
            f"✅ Loaded {len(self.objects)} environment objects: "
            f"{list(self.objects.keys())}"
        )

    def _load_usd_object(self, cfg: USDObjectCfg) -> None:
        """Load a single USD object.

        Args:
            cfg: USD object configuration.
        """
        if not os.path.exists(cfg.usd_path):
            raise ValueError(f"USD file not found: {cfg.usd_path}")

        logger.info(f"Loading USD object '{cfg.name}' from {cfg.usd_path}")

        try:
            # Create USD morph with position, rotation, and scale
            morph = gs.morphs.USD(
                file=cfg.usd_path,
                pos=cfg.initial_pose.pos,
                quat=cfg.initial_pose.quat,
                scale=cfg.scale,
            )

            # Add to scene
            if cfg.load_articulation:
                # Load as articulated object (with joints)
                # Use add_stage to support mixed entities
                entity = self.scene.add_stage(
                    morph=morph,
                )
                logger.info(
                    f"  → Loaded as articulated object (may have joints)"
                )
            else:
                # Load as static entity (no joint control)
                entity = self.scene.add_entity(
                    morph=morph,
                    name=cfg.name,
                )
                logger.info(f"  → Loaded as static object")

            logger.info(
                f"     pos={cfg.initial_pose.pos}, "
                f"quat={cfg.initial_pose.quat}, "
                f"scale={cfg.scale}"
            )

            # Store reference
            self.objects[cfg.name] = entity

        except Exception as e:
            logger.error(
                f"Failed to load USD object '{cfg.name}': {e}"
            )
            raise

    def _load_primitive_object(self, cfg: PrimitiveObjectCfg) -> None:
        """Load a single primitive object.

        Args:
            cfg: Primitive object configuration.
        """
        logger.info(
            f"Loading primitive object '{cfg.name}' (shape: {cfg.shape})"
        )

        try:
            # Create morph based on shape
            if cfg.shape == "box":
                morph = gs.morphs.Box(size=cfg.size)
            elif cfg.shape == "sphere":
                morph = gs.morphs.Sphere(radius=cfg.size[0])
            elif cfg.shape == "cylinder":
                morph = gs.morphs.Cylinder(
                    radius=cfg.size[0],
                    height=cfg.size[1]
                )
            elif cfg.shape == "capsule":
                morph = gs.morphs.Capsule(
                    radius=cfg.size[0],
                    height=cfg.size[1]
                )
            else:
                raise ValueError(
                    f"Unknown shape '{cfg.shape}'. "
                    "Supported: box, sphere, cylinder, capsule"
                )

            # Add to scene with pose
            entity = self.scene.add_entity(
                morph=morph,
                name=cfg.name,
                pos=cfg.initial_pose.pos,
                quat=cfg.initial_pose.quat,
            )

            # Store reference
            self.objects[cfg.name] = entity

            logger.info(
                f"  → Created {cfg.shape} at {cfg.initial_pose.pos}"
            )

        except Exception as e:
            logger.error(
                f"Failed to create primitive object '{cfg.name}': {e}"
            )
            raise

    def get_object(self, name: str) -> Any | None:
        """Get an object by name.

        Args:
            name: Object name.

        Returns:
            Object entity, or None if not found.
        """
        return self.objects.get(name)

    def get_all_objects(self) -> Dict[str, Any]:
        """Get all loaded objects.

        Returns:
            Dictionary of objects keyed by name.
        """
        return self.objects.copy()

    def __len__(self) -> int:
        """Return number of loaded objects."""
        return len(self.objects)

    def __contains__(self, name: str) -> bool:
        """Check if an object exists."""
        return name in self.objects

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ObjectManager(num_objects={len(self.objects)})"
