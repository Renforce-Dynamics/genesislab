"""Scene construction and entity management for LabScene."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import genesis as gs
import torch

if TYPE_CHECKING:
    from .lab_scene import LabScene
    from genesislab.engine.scene import SceneCfg
    from genesislab.components.sensors import SensorBaseCfg
    from genesislab.components.sensors.fake_sensors import FakeSensorBaseCfg
    from genesislab.components.sensors.genesis_sensors import GenesisSensorBaseCfg

from genesislab.components.sensors import SensorBase
from genesislab.engine.assets.articulation import ArticulationCfg
from genesislab.engine.assets.robot import Robot
from genesislab.engine.entity import LabEntity
from genesislab.engine.scene.terrain_runtime import TerrainRuntime

logger = logging.getLogger(__name__)

from genesislab.components.sensors.fake_sensors import FakeSensorBaseCfg
from genesislab.components.sensors.genesis_sensors import GenesisSensorBaseCfg


class SceneBuilder:
    """Helper class for building Genesis scenes and adding entities."""

    def __init__(self, scene: "LabScene"):
        """Initialize the scene builder.

        Args:
            scene: Reference to the LabScene instance.
        """
        self._scene = scene

    def create_scene(self) -> gs.Scene:
        """Create a Genesis Scene with appropriate options.

        Returns:
            The created Genesis Scene instance.
        """
        cfg: "SceneCfg" = self._scene.cfg
        
        # Create simulation options from SimOptionsCfg
        sim_options = gs.options.SimOptions(**cfg.sim_options.to_genesis_options())
        
        # Create viewer options from ViewerOptionsCfg
        viewer_options = gs.options.ViewerOptions(**cfg.viewer_options.to_genesis_options())
        
        # Create visualization options from VisOptionsCfg
        vis_options_kwargs = cfg.vis_options.to_genesis_options()
        vis_options = None
        if vis_options_kwargs is not None:
            vis_options = gs.options.VisOptions(**vis_options_kwargs)
        
        # Create rigid body options from RigidOptionsCfg
        rigid_options = gs.options.RigidOptions(
            **cfg.rigid_options.to_genesis_options(cfg.sim_options.dt)
        )
        
        # Create scene with all options
        scene_kwargs = {
            "sim_options": sim_options,
            "viewer_options": viewer_options,
            "show_viewer": cfg.viewer,
            "rigid_options": rigid_options,
        }
        if vis_options is not None:
            scene_kwargs["vis_options"] = vis_options
        
        scene = gs.Scene(**scene_kwargs)
        return scene

    def build_scene(self, scene: gs.Scene) -> None:
        """Build the scene with configured parameters.

        When terrain is present, ``env_spacing`` is forced to ``(0, 0)``.
        Genesis applies ``env_spacing`` as a purely visual offset
        (``envs_offset``) to **all** entities — including terrain.  Because
        terrain is world-only shared geometry (``batch_fixed_verts=False``),
        a non-zero offset causes it to appear duplicated at N positions in
        the viewer.  Robot placement in terrain scenes is already handled by
        ``TerrainRuntime.env_origins``, so ``envs_offset`` is redundant.

        Args:
            scene: The Genesis Scene instance to build.
        """
        env_spacing = self._scene.cfg.env_spacing
        if self._scene.cfg.terrain is not None:
            env_spacing = (0.0, 0.0)

        scene.build(
            n_envs=self._scene.cfg.num_envs,
            env_spacing=env_spacing,
            n_envs_per_row=self._scene.cfg.n_envs_per_row,
            center_envs_at_origin=self._scene.cfg.center_envs_at_origin,
        )

    def add_terrain(self, scene: gs.Scene) -> TerrainRuntime | None:
        """Add terrain entity to the scene based on the terrain configuration.

        Supports terrain modes via ``terrain_type``:

        * ``"plane"``        – Genesis built-in infinite plane.
        * ``"genesisbase"``  – Genesis native heightfield terrain (gs.morphs.Terrain).
        * ``"generator"``    – Procedural terrain via :class:`TerrainGenerator`.
        * ``"usd"``          – USD-based terrain (not yet implemented).

        Args:
            scene: The Genesis Scene instance.

        Returns:
            A :class:`TerrainRuntime` holding environment origins and
            curriculum state, or ``None`` if no terrain is configured.
        """
        terrain_cfg = self._scene.cfg.terrain
        if terrain_cfg is None:
            return None

        # Resolve the canonical terrain_type.  The deprecated ``type`` field
        # is already resolved to ``terrain_type`` by TerrainCfg.__post_init__.
        terrain_type = getattr(terrain_cfg, "terrain_type", "plane")

        if terrain_type == "plane":
            return self._add_plane_terrain(scene, terrain_cfg)
        elif terrain_type == "genesisbase":
            return self._add_genesisbase_terrain(scene, terrain_cfg)
        elif terrain_type == "generator":
            return self._add_generator_terrain(scene, terrain_cfg)
        elif terrain_type == "usd":
            raise NotImplementedError(
                "Terrain type 'usd' is not yet implemented.  "
                "This will be supported in a future release."
            )
        else:
            raise ValueError(
                f"Unknown terrain_type '{terrain_type}'.  "
                f"Accepted values: 'plane', 'genesisbase', 'generator', 'usd'."
            )

    # ------------------------------------------------------------------
    # Terrain mode helpers
    # ------------------------------------------------------------------

    def _add_plane_terrain(
        self, scene: gs.Scene, terrain_cfg,
    ) -> TerrainRuntime:
        """Add a flat plane and return a grid-based TerrainRuntime."""
        plane = gs.morphs.Plane()
        scene.add_entity(plane, name="terrain")

        # Plane mode uses grid-based environment origins.
        # env_spacing comes from SceneCfg (tuple) — use the first element.
        scene_cfg = self._scene.cfg
        env_spacing_val = getattr(terrain_cfg, "env_spacing", None)
        if env_spacing_val is None:
            # Fall back to the scene-level env_spacing
            raw = scene_cfg.env_spacing
            if isinstance(raw, (list, tuple)):
                env_spacing_val = float(raw[0])
            else:
                env_spacing_val = float(raw) if raw is not None else 2.5

        return TerrainRuntime(
            terrain_generator=None,
            terrain_origins=None,
            num_envs=scene_cfg.num_envs,
            env_spacing=env_spacing_val,
            max_init_terrain_level=None,
            device=self._scene.device,
        )

    def _add_genesisbase_terrain(
        self, scene: gs.Scene, terrain_cfg,
    ) -> TerrainRuntime:
        """Add a Genesis native heightfield terrain (gs.morphs.Terrain).

        Uses ``terrain_cfg.terrain_details_cfg`` (``GenesisTerrainMorphCfg``)
        and ``terrain_cfg.surface_cfg`` (``TerrainSurfaceCfg``) from master's
        ``TerrainCfg`` definition.
        """
        if terrain_cfg.terrain_details_cfg is None:
            raise ValueError(
                "terrain_type 'genesisbase' requires terrain_details_cfg to be set."
            )

        # Build surface from config
        surface = terrain_cfg.surface_cfg.build_surface()

        # Build the morph
        morph = gs.morphs.Terrain(
            **terrain_cfg.terrain_details_cfg.to_genesis_dict()
        )

        scene.add_entity(surface=surface, morph=morph, name="terrain")

        # GenesisBase terrain uses grid-based environment origins.
        scene_cfg = self._scene.cfg
        env_spacing_val = getattr(terrain_cfg, "env_spacing", None)
        if env_spacing_val is None:
            raw = scene_cfg.env_spacing
            if isinstance(raw, (list, tuple)):
                env_spacing_val = float(raw[0])
            else:
                env_spacing_val = float(raw) if raw is not None else 2.5

        return TerrainRuntime(
            terrain_generator=None,
            terrain_origins=None,
            num_envs=scene_cfg.num_envs,
            env_spacing=env_spacing_val,
            max_init_terrain_level=None,
            device=self._scene.device,
        )

    def _add_generator_terrain(
        self, scene: gs.Scene, terrain_cfg,
    ) -> TerrainRuntime:
        """Generate procedural terrain, add it to the scene, and return a TerrainRuntime.

        Tries Genesis-native terrain generation first (``gs.morphs.Terrain``) when all
        sub-terrains have a native equivalent.  Falls back to the trimesh mesh path via
        :class:`~genesislab.components.terrains.TerrainGenerator` otherwise.
        """
        gen_cfg = terrain_cfg.terrain_generator
        if gen_cfg is None:
            raise ValueError(
                "terrain_type is 'generator' but no terrain_generator config was provided."
            )

        from genesislab.components.terrains.genesis_terrain_converter import (
            can_convert,
            convert,
        )

        use_origins = getattr(terrain_cfg, "use_terrain_origins", True)
        env_spacing_val = getattr(terrain_cfg, "env_spacing", None)

        if can_convert(gen_cfg):
            # -----------------------------------------------------------
            # Native Genesis terrain path
            # -----------------------------------------------------------
            gs_kwargs = convert(gen_cfg)

            total_x = gen_cfg.num_rows * gen_cfg.size[0]
            total_y = gen_cfg.num_cols * gen_cfg.size[1]
            terrain_morph = gs.morphs.Terrain(
                **gs_kwargs,
                pos=(-total_x / 2.0, -total_y / 2.0, 0.0),
            )
            scene.add_entity(terrain_morph, name="terrain")
            logger.info(
                "Using Genesis-native terrain: %d sub-terrain type(s), grid %s",
                len(gen_cfg.sub_terrains),
                gs_kwargs["n_subterrains"],
            )

            if gen_cfg.border_width > 0.0:
                self._add_native_terrain_border(scene, gen_cfg)

            if use_origins:
                origins = self._compute_grid_terrain_origins(gen_cfg)
            else:
                origins = None
        else:
            # -----------------------------------------------------------
            # Mesh (trimesh) fallback path
            # -----------------------------------------------------------
            generator = gen_cfg.class_type(cfg=gen_cfg, device=self._scene.device)
            logger.info(
                "Generated terrain mesh (fallback): %d vertices, %d faces",
                len(generator.terrain_mesh.vertices),
                len(generator.terrain_mesh.faces),
            )
            terrain_morph = gs.morphs.MeshSet(files=[generator.terrain_mesh], fixed=True)
            scene.add_entity(terrain_morph, name="terrain")
            origins = generator.terrain_origins if use_origins else None

        # Fall back env_spacing for grid mode (when no per-sub-terrain origins)
        if env_spacing_val is None and origins is None:
            raw = self._scene.cfg.env_spacing
            if isinstance(raw, (list, tuple)):
                env_spacing_val = float(raw[0])
            else:
                env_spacing_val = float(raw) if raw is not None else 2.5

        return TerrainRuntime(
            terrain_generator=gen_cfg,
            terrain_origins=origins,
            num_envs=self._scene.cfg.num_envs,
            env_spacing=env_spacing_val,
            max_init_terrain_level=getattr(terrain_cfg, "max_init_terrain_level", None),
            device=self._scene.device,
        )

    def _compute_grid_terrain_origins(
        self,
        gen_cfg,
    ) -> "torch.Tensor":
        """Compute synthetic sub-terrain origins from the grid geometry.

        Used by the native Genesis terrain path, which does not expose
        per-sub-terrain origins directly.  Returns a tensor of shape
        ``(num_rows, num_cols, 3)`` where XY are the tile centres and Z is
        ``0.0`` (the native terrain is centred at ground level).

        Args:
            gen_cfg: The terrain generator configuration.

        Returns:
            Tensor of shape ``(num_rows, num_cols, 3)``.
        """
        import torch

        num_rows = gen_cfg.num_rows
        num_cols = gen_cfg.num_cols
        size_x, size_y = gen_cfg.size

        origins = torch.zeros(num_rows, num_cols, 3, device=self._scene.device)
        for r in range(num_rows):
            for c in range(num_cols):
                origins[r, c, 0] = (r - (num_rows - 1) / 2.0) * size_x
                origins[r, c, 1] = (c - (num_cols - 1) / 2.0) * size_y
                origins[r, c, 2] = 0.0
        return origins

    def _add_native_terrain_border(
        self,
        scene: "gs.Scene",
        gen_cfg,
    ) -> None:
        """Add a flat border mesh around the native-path terrain grid."""
        import numpy as np
        import trimesh
        from genesislab.components.terrains.trimesh.utils import make_border

        num_rows = gen_cfg.num_rows
        num_cols = gen_cfg.num_cols
        size_x, size_y = gen_cfg.size
        border_width = gen_cfg.border_width
        border_height = gen_cfg.border_height

        border_outer = (num_rows * size_x + 2 * border_width, num_cols * size_y + 2 * border_width)
        border_inner = (num_rows * size_x, num_cols * size_y)

        border_center = (0.0, 0.0, -abs(border_height) / 2.0)

        border_meshes = make_border(
            border_outer,
            border_inner,
            height=abs(border_height),
            position=border_center,
        )
        border_mesh = trimesh.util.concatenate(border_meshes)

        selector = ~(np.asarray(border_mesh.triangles)[:, :, 2] < -0.1).any(1)
        border_mesh.update_faces(selector)

        terrain_border_morph = gs.morphs.MeshSet(files=[border_mesh], fixed=True)
        scene.add_entity(terrain_border_morph, name="terrain_border")
        logger.info(
            "Added terrain border: outer=(%.1f, %.1f), inner=(%.1f, %.1f), height=%.2f",
            *border_outer, *border_inner, border_height,
        )

    def add_robot(self, scene: gs.Scene, entity_name: str, robot_cfg: ArticulationCfg, env: Any = None) -> LabEntity:
        """Add a robot entity to the scene using the Genesis-native asset layer.

        Args:
            scene: The Genesis Scene instance.
            entity_name: Name to assign to the entity.
            robot_cfg: Robot configuration (must be a GenesisArticulationCfg or subclass).
            env: Optional environment instance (ManagerBasedGenesisEnv). Required for full LabEntity functionality.

        Returns:
            LabEntity wrapper containing the raw entity and robot asset.
        """
        # Create a copy of the config with the entity name set
        asset_cfg = robot_cfg.replace(name=entity_name)
        # Use Robot class which provides name resolution support
        asset = Robot(asset_cfg, device=self._scene.device)
        raw_entity = asset.build_into_scene(scene)
        
        # Construct and return LabEntity directly
        lab_entity = LabEntity(env, entity_name, raw_entity, robot_asset=asset)
        return lab_entity

    def add_sensor(self, scene: "LabScene", sensor_name: str, sensor_cfg: "SensorBaseCfg") -> "SensorBase":
        """Add a sensor to the scene.

        Dispatches by config type:
        - **GenesisSensorBaseCfg**: Builds the underlying gs.sensors.* on gs_scene via
          build_genesis_sensor(), then creates the wrapper with genesis_sensor= and
          registers it on the lab scene.
        - **FakeSensorBaseCfg** (or other): Creates the sensor with optional entity
          from entity_name and registers it. No Genesis sensor is added.
        """

        # Set name if not set
        if sensor_cfg.name is None:
            sensor_cfg.name = sensor_name

        # Check that class_type is specified
        if sensor_cfg.class_type is None:
            raise ValueError(
                f"Sensor configuration '{type(sensor_cfg).__name__}' must have 'class_type' specified. "
                f"This should be set automatically when the sensor class is defined."
            )

        sensor_class = sensor_cfg.class_type
        sensor_kwargs = {}

        if isinstance(sensor_cfg, GenesisSensorBaseCfg):
            # Create the Genesis sensor on gs_scene and inject into wrapper
            gs_sensor = sensor_cfg.build_genesis_sensor(scene.gs_scene, scene)
            sensor_kwargs["genesis_sensor"] = gs_sensor
        elif isinstance(sensor_cfg, FakeSensorBaseCfg):
            # Fake sensors: inject entity when entity_name is set
            if getattr(sensor_cfg, "entity_name", None):
                entity_name = sensor_cfg.entity_name
                if entity_name not in scene.entities:
                    raise KeyError(
                        f"Entity '{entity_name}' not found in scene.entities. "
                        f"Sensor '{sensor_name}' requires the entity to exist. "
                        f"Available entities: {list(scene.entities.keys())}"
                    )
                sensor_kwargs["entity"] = scene.entities[entity_name].raw_entity
        else:
            # Legacy / other configs: same as before (e.g. entity_name -> entity)
            if hasattr(sensor_cfg, "entity_name") and sensor_cfg.entity_name:
                if sensor_cfg.entity_name not in scene.entities:
                    raise KeyError(
                        f"Entity '{sensor_cfg.entity_name}' not found in scene.entities. "
                        f"Sensor '{sensor_name}' requires the entity to exist. "
                        f"Available entities: {list(scene.entities.keys())}"
                    )
                lab_entity = scene.entities[sensor_cfg.entity_name]
                sensor_kwargs["entity"] = lab_entity.raw_entity

        sensor = sensor_class(
            cfg=sensor_cfg,
            num_envs=scene.num_envs,
            device=scene.device,
            **sensor_kwargs
        )

        scene.add_sensor(sensor_name, sensor)
        return sensor
