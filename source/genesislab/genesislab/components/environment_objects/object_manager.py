"""Manager for environment objects in the scene."""

import logging
from typing import Dict, List, Optional, Any
import genesis as gs

from .object_cfg import (
    EnvironmentObjectsConfig,
    USDObjectCfg,
    PrimitiveObjectCfg,
)

logger = logging.getLogger(__name__)


class EnvironmentObjectManager:
    """Manages environment objects in the scene.

    This manager handles loading and lifecycle of interactive objects
    that are separate from robots. Objects are loaded after robots
    to avoid joint indexing conflicts.

    Attributes:
        cfg: Configuration for environment objects.
        scene: Genesis scene instance.
        objects: Dictionary of loaded objects keyed by name.
    """

    def __init__(
        self,
        cfg: EnvironmentObjectsConfig,
        scene: "gs.Scene",
    ):
        """Initialize environment object manager.

        Args:
            cfg: Environment objects configuration.
            scene: Genesis scene instance (must be built).
        """
        self.cfg = cfg
        self.scene = scene
        self.objects: Dict[str, Any] = {}

        logger.info("EnvironmentObjectManager initialized")

    def load_objects(self) -> None:
        """Load all configured environment objects.

        This should be called AFTER robots are added and initialized,
        but BEFORE the first simulation step.

        The loading order ensures that object joints don't interfere
        with robot control space.
        """
        logger.info("Loading environment objects...")

        # Load USD objects
        for usd_cfg in self.cfg.usd_objects:
            self._load_usd_object(usd_cfg)

        # Load primitive objects
        for prim_cfg in self.cfg.primitive_objects:
            self._load_primitive_object(prim_cfg)

        logger.info(
            f"✅ Loaded {len(self.objects)} environment objects: "
            f"{list(self.objects.keys())}"
        )

    def _load_usd_object(self, cfg: USDObjectCfg) -> None:
        """Load a single USD object.

        Args:
            cfg: USD object configuration.
        """
        import os
        if not os.path.exists(cfg.usd_path):
            raise ValueError(f"USD file not found: {cfg.usd_path}")

        logger.info(f"Loading USD object '{cfg.name}' from {cfg.usd_path}")

        try:
            # Create USD morph with position, rotation, and scale
            morph = gs.morphs.USD(
                file=cfg.usd_path,
                pos=cfg.pos,
                quat=cfg.rot,
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
                logger.info(f"     pos={cfg.pos}, rot={cfg.rot}, scale={cfg.scale}")
            else:
                # Load as static entity (no joint control)
                entity = self.scene.add_entity(
                    morph=morph,
                    name=cfg.name,
                )
                logger.info(f"  → Loaded as static object")
                logger.info(f"     pos={cfg.pos}, rot={cfg.rot}, scale={cfg.scale}")

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

            # Add to scene
            entity = self.scene.add_entity(
                morph=morph,
                name=cfg.name,
            )

            # Store reference
            self.objects[cfg.name] = entity

            logger.info(f"  → Created {cfg.shape} at {cfg.pos}")

        except Exception as e:
            logger.error(
                f"Failed to create primitive object '{cfg.name}': {e}"
            )
            raise

    def get_object(self, name: str) -> Optional[Any]:
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
