"""Scene construction and entity management for GenesisBinding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import genesis as gs

if TYPE_CHECKING:
    from genesislab.components.entities.scene_cfg import SceneCfg

from genesislab.engine.assets.articulation import GenesisArticulation, GenesisArticulationCfg


class SceneBuilder:
    """Helper class for building Genesis scenes and adding entities."""

    def __init__(self, binding: Any):
        """Initialize the scene builder.

        Args:
            binding: Reference to the GenesisBinding instance.
        """
        self._binding = binding

    def create_scene(self) -> gs.Scene:
        """Create a Genesis Scene with appropriate options.

        Returns:
            The created Genesis Scene instance.
        """
        # Create scene with simulation options
        sim_options = gs.options.SimOptions(**self._binding.cfg.to_genesis_options())
        # Viewer options: we respect ``SceneCfg.viewer`` to control whether a
        # window is shown. This keeps the default behaviour (viewer on) but
        # allows scripts to disable it for headless runs.
        viewer_options = gs.options.ViewerOptions()

        scene = gs.Scene(
            sim_options=sim_options,
            viewer_options=viewer_options,
            show_viewer=getattr(self._binding.cfg, "viewer", True),
        )
        return scene

    def build_scene(self, scene: gs.Scene) -> None:
        """Build the scene with configured parameters.

        Args:
            scene: The Genesis Scene instance to build.
        """
        # Build the scene
        scene.build(
            n_envs=self._binding.cfg.num_envs,
            env_spacing=self._binding.cfg.env_spacing,
            n_envs_per_row=self._binding.cfg.n_envs_per_row,
            center_envs_at_origin=self._binding.cfg.center_envs_at_origin,
        )

    def add_terrain(self, scene: gs.Scene) -> None:
        """Add terrain entity to the scene.

        Args:
            scene: The Genesis Scene instance.
        """
        terrain_cfg = self._binding.cfg.terrain
        if terrain_cfg is None:
            return

        # Support both dict-based and configclass-based terrain configs.
        if isinstance(terrain_cfg, dict):
            terrain_type = terrain_cfg.get("type", "plane")
        else:
            terrain_type = getattr(terrain_cfg, "type", "plane")
        if terrain_type == "plane":
            # Use Genesis' built-in infinite plane primitive so that the ground
            # is visually more obvious in renderings (with proper shading and
            # reflections) while still acting as a flat contact surface.
            plane = gs.morphs.Plane()
            scene.add_entity(plane, name="terrain")
        else:
            # Handle other terrain types (e.g., heightfield, mesh)
            raise NotImplementedError(f"Terrain type '{terrain_type}' not yet implemented")

    def add_robot(self, scene: gs.Scene, entity_name: str, robot_cfg: Any) -> Any:
        """Add a robot entity to the scene using the Genesis-native asset layer.

        Args:
            scene: The Genesis Scene instance.
            entity_name: Name to assign to the entity.
            robot_cfg: Robot configuration.

        Returns:
            The created entity object.
        """
        asset_cfg = GenesisArticulationCfg(
            name=entity_name,
            morph_type=robot_cfg.morph_type,
            morph_path=robot_cfg.morph_path,
            initial_pose=robot_cfg.initial_pose,
            fixed_base=robot_cfg.fixed_base,
            control_dofs=robot_cfg.control_dofs,
            morph_options=robot_cfg.morph_options,
        )
        asset = GenesisArticulation(asset_cfg, device=self._binding.device)
        entity = asset.build_into_scene(scene)

        return entity

    def add_sensor(self, scene: gs.Scene, sensor_name: str, sensor_cfg: Any) -> None:
        """Add a sensor to the scene.

        Currently, only simple Python-side sensors (like contact sensors) are
        supported. These do not create any engine primitives but expose data
        buffers that MDP terms can read.

        Args:
            scene: The Genesis Scene instance.
            sensor_name: Name to assign to the sensor.
            sensor_cfg: Sensor configuration (configclass or dict).
        """
        from genesislab.components.sensors import ContactSensor, ContactSensorCfg

        # Lazily attach a sensors dict to the Scene so that MDP code can access
        # ``env.scene.sensors[name]`` similar to IsaacLab.
        if not hasattr(scene, "sensors"):
            scene.sensors = {}

        cfg_obj: Any = sensor_cfg
        # Support both configclass instances and plain dict configs.
        if isinstance(sensor_cfg, dict):
            cfg_obj = ContactSensorCfg(**sensor_cfg)

        if isinstance(cfg_obj, ContactSensorCfg):
            if cfg_obj.name is None:
                cfg_obj.name = sensor_name
            # Get entity reference for contact sensor
            entity = None
            if hasattr(cfg_obj, "entity_name") and cfg_obj.entity_name:
                entity = self._binding._entities.get(cfg_obj.entity_name)
            sensor = ContactSensor(
                cfg=cfg_obj, num_envs=self._binding._num_envs, device=self._binding.device, entity=entity
            )
            scene.sensors[sensor_name] = sensor
